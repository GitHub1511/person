// Permission guard for the unattended Pi coder.  Loaded with `pi -e`.
//
// This is a tripwire, not a sandbox (Pi has no built-in one): it stops the obvious ways a coding run
// could leave the project or do something the owner did not authorise.  The real safety net is the
// helper's snapshot / verify / rollback, and that the real API key never enters this process.
//
// Every tool call is appended to state/pi_audit.jsonl (allowed or blocked, and why).
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { appendFileSync, mkdirSync } from "node:fs";
import * as path from "node:path";

const ROOT = path.resolve(process.env.HELPER_ROOT ?? process.cwd());
const STATE = path.resolve(process.env.HELPER_STATE ?? path.join(ROOT, "tools", "openrouter_helper", "state"));
const AUDIT = path.join(STATE, "pi_audit.jsonl");
const MAX_CALLS = Number(process.env.PI_MAX_TOOL_CALLS ?? 400);

// paths the coder may never write (and never read, for secrets)
const PROTECTED_WRITE = [
  path.join(ROOT, "tools", "openrouter_helper"),
  path.join(ROOT, ".git"),
  path.join(ROOT, ".gitignore"),
];
const SECRET_PATTERNS = [/\.env(\.|$)/i, /auth\.json$/i, /[\\/]\.ssh[\\/]/i, /id_rsa|id_ed25519/i,
  /credentials?(\.json)?$/i, /[\\/]\.aws[\\/]/i, /[\\/]\.config[\\/]gh[\\/]/i, /\.pem$/i, /\.pfx$/i];

// shell commands that are refused outright
const BLOCKED_CMD: [RegExp, string][] = [
  [/\bgit\s+(push|remote|credential|config\s+.*(credential|url|user))\b/i, "git push / remote / credentials are the owner's"],
  [/\bgh\s+(auth|repo|pr|release|api|secret)\b/i, "GitHub CLI is not allowed"],
  [/\b(rm|rmdir|del|erase|rd)\b[^\n;|&]*(\s-[a-z]*r|\s\/s\b|--recursive)/i, "recursive delete"],
  [/\bRemove-Item\b[^\n;|&]*-Recurse/i, "recursive delete"],
  [/\b(format|diskpart|mkfs|dd\s+if=|shutdown|reboot|halt|poweroff)\b/i, "system-level command"],
  [/\b(reg\s+(add|delete|import)|schtasks|sc\s+(create|config|delete)|netsh|bcdedit|icacls|takeown|net\s+(user|localgroup)|setx|crontab|systemctl)\b/i,
    "system / security setting"],
  [/\b(Set-ExecutionPolicy|Set-MpPreference|Add-MpPreference|New-ScheduledTask|Register-ScheduledTask|Set-Service|New-Service)\b/i,
    "system / security setting"],
  [/\b(pip|pip3|python\s+-m\s+pip|npm|pnpm|yarn|choco|winget|scoop|apt|apt-get|brew)\s+(install|i|add|uninstall|remove)\b/i,
    "installing packages (no new dependencies)"],
  [/\b(curl|wget|Invoke-WebRequest|iwr|Invoke-RestMethod|irm)\b[^\n]*(\s-X\s*(POST|PUT|PATCH|DELETE)|\s--data|\s-d\s|\s-F\s|\s--form|\s-T\s|--upload-file|\s-Method\s+(Post|Put|Patch|Delete)|\s-Body\b|\s-InFile\b)/i,
    "web requests that send data (read-only fetches are allowed)"],
  [/\b(nc|ncat|netcat|socat|ssh|scp|sftp|ftp|telnet|rsync)\b/i, "outbound network tools"],
  [/\b(curl|wget)\b[^\n|;]*\|\s*(ba|z|da)?sh\b/i, "piping a download into a shell"],
  [/\b(iex|Invoke-Expression)\b/i, "dynamic code execution from a string"],
  [/\bstart-process\b[^\n]*-verb\s+runas/i, "privilege elevation"],
  [/\b(sudo|runas)\b/i, "privilege elevation"],
  [/(\.env\b|auth\.json|\.ssh[\\/]|id_rsa|\.aws[\\/])/i, "touches secrets"],
  [/tools[\\/]openrouter_helper[\\/](?!tests[\\/]|README)[^\s]*\s*(>|>>)|(>|>>)\s*\S*openrouter_helper/i, "writes into the helper"],
  [/\b(sed\s+-i|tee|Set-Content|Add-Content|Out-File|mv|move|cp|copy|Move-Item|Copy-Item)\b[^\n]*openrouter_helper/i, "writes into the helper"],
  [/\bhelper\.sh\s+(run|stop|rollback|snapshot|serve)\b/i, "the loop controls itself; only `helper.sh verify|done|status` are for the coder"],
];

function inside(p: string, dir: string): boolean {
  const r = path.relative(dir, p);
  return r === "" || (!r.startsWith("..") && !path.isAbsolute(r));
}

function audit(rec: Record<string, unknown>): void {
  try {
    mkdirSync(STATE, { recursive: true });
    appendFileSync(AUDIT, JSON.stringify({ t: new Date().toISOString(), ...rec }) + "\n");
  } catch { /* auditing must never break the run */ }
}

function pathsIn(command: string): string[] {
  // absolute Windows / POSIX paths mentioned in a command
  const found = command.match(/[A-Za-z]:[\\/][^\s"'|;&<>]*|(?:^|[\s"'=])\/[A-Za-z0-9_.\-/]+/g) ?? [];
  return found.map((s) => s.trim().replace(/^["'=]/, ""));
}

export default function (pi: ExtensionAPI) {
  let calls = 0;

  pi.on("tool_call", async (event: any) => {
    const name: string = event.toolName ?? event.name ?? "?";
    const input: any = event.input ?? event.args ?? {};
    calls += 1;
    const block = (reason: string) => {
      audit({ tool: name, blocked: true, reason, input: JSON.stringify(input).slice(0, 400) });
      return { block: true, reason: `Blocked by the permission guard: ${reason}.` };
    };

    if (calls > MAX_CALLS) return block(`tool-call budget of ${MAX_CALLS} used up; finish with a report`);

    if (name === "bash" || name === "powershell") {
      const cmd: string = String(input.command ?? "");
      for (const [re, why] of BLOCKED_CMD) if (re.test(cmd)) return block(why);
      for (const p of pathsIn(cmd)) {
        const abs = path.resolve(p);
        if (SECRET_PATTERNS.some((re) => re.test(abs))) return block("touches secrets");
        const tmp = process.env.TEMP ? path.resolve(process.env.TEMP) : "/tmp";
        const sys = /^(\/usr|\/bin|\/etc|\/dev|\/proc|\/c\/windows|[a-z]:[\\/]windows|[a-z]:[\\/]program files)/i;
        const isWrite = /(>|>>|\btee\b|\bmv\b|\bcp\b|\brm\b|\bmkdir\b|\bSet-Content\b|\bOut-File\b|\bsed\s+-i)/.test(cmd);
        if (isWrite && !inside(abs, ROOT) && !inside(abs, tmp) && !sys.test(abs)) return block(`writes outside the project (${abs})`);
      }
    } else if (name === "write" || name === "edit") {
      const raw: string = String(input.path ?? input.file_path ?? input.file ?? "");
      const abs = path.resolve(ROOT, raw);
      if (!inside(abs, ROOT)) return block(`writes outside the project (${abs})`);
      if (PROTECTED_WRITE.some((d) => inside(abs, d))) return block(`protected path (${path.relative(ROOT, abs)})`);
      if (SECRET_PATTERNS.some((re) => re.test(abs))) return block("touches secrets");
    } else if (name === "read" || name === "grep" || name === "find" || name === "ls") {
      const raw: string = String(input.path ?? input.file_path ?? input.file ?? "");
      if (raw) {
        const abs = path.resolve(ROOT, raw);
        if (SECRET_PATTERNS.some((re) => re.test(abs))) return block("touches secrets");
      }
    }
    audit({ tool: name, blocked: false, input: JSON.stringify(input).slice(0, 300) });
    return undefined;
  });
}
