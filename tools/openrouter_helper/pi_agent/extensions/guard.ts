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

// Only these tool names are recognised. Anything else is blocked (renamed/aliased shells,
// apply_patch-style writers, notebook or MCP tools the coder was never granted).
const ALLOWED_TOOLS = new Set(["bash", "powershell", "read", "write", "edit", "grep", "find", "ls"]);

// Paths the coder may never write (mirrors prompts/rules.md: helper, git, secrets,
// orchestrator, AI_LOG; reads of secrets are also refused).
const PROTECTED_WRITE = [
  path.join(ROOT, "tools", "openrouter_helper"),
  path.join(ROOT, ".git"),
  path.join(ROOT, ".gitignore"),
  path.join(ROOT, ".env"),
  path.join(ROOT, "orchestrator"),
  path.join(ROOT, "AI_LOG.txt"),
];
const SECRET_PATTERNS = [/\.env(\.|$)/i, /auth\.json$/i, /[\\/]\.ssh[\\/]/i, /id_rsa|id_ed25519/i,
  /credentials?(\.json)?$/i, /[\\/]\.aws[\\/]/i, /[\\/]\.config[\\/]gh[\\/]/i, /\.pem$/i, /\.pfx$/i];

// Writable allow-list (prompts/rules.md): embodied_human/, tools/ except the helper,
// and the project-root run_*.py / diag_*.py / README.md. Everything else is denied
// for write/edit tools (deny by default).
function writableByPolicy(abs: string): boolean {
  const rel = path.relative(ROOT, abs).replace(/\\/g, "/");
  if (rel.startsWith("..") || path.isAbsolute(rel)) return false;
  if (rel === "" || rel === ".") return false;
  if (rel.startsWith("embodied_human/")) return true;
  if (rel === "tools" || rel.startsWith("tools/")) {
    if (rel === "tools/openrouter_helper" || rel.startsWith("tools/openrouter_helper/")) return false;
    return true;
  }
  if (/^(run_[A-Za-z0-9_]+\.py|diag_[A-Za-z0-9_]+\.py|README\.md)$/.test(rel)) return true;
  return false;
}

// shell commands that are refused outright
const BLOCKED_CMD: [RegExp, string][] = [
  [/\bgit\s+(push|remote|credential|config\s+.*(credential|url|user))\b/i, "git push / remote / credentials are the owner's"],
  [/\bgit\s+(reset|revert|clean|checkout|restore|stash|filter-branch|update-ref|update-index|reflog|rebase)\b/i, "git history/tree surgery would defeat snapshot+rollback"],
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
  [/(openrouter_helper|pi_audit\.jsonl|HARNESS_OVERRIDE)[^\n]*(^|\s(>|>>)\s*\S|\b(tee|Set-Content|Add-Content|Out-File)\b|open\s*\([^)]*['"][wax])/i, "writes into the helper"],
  [/((^|\s)(>|>>)\s*\S|\b(tee|Set-Content|Add-Content|Out-File)\b)[^\n]*openrouter_helper/i, "writes into the helper"],
  [/\b(sed\s+-i|tee|Set-Content|Add-Content|Out-File|\bmv\b|\bmove\b|\bcp\b|\bcopy\b|Move-Item|Copy-Item)\b[^\n]*openrouter_helper/i, "writes into the helper"],
  [/\b(python|python3|perl|node|ruby)\b[^\n]*(-c\b|-e\b|open\s*\([^)]*['"][wax]|os\s*\.\s*(write|replace|rename)|shutil\s*\.\s*(copy|move|rmtree)|pathlib[^;\n]*write_text|fs\s*\.\s*writeFile)/i, "scripted file write: use the read/edit/write tools so the guard can check the path"],
  [/\bhelper\.sh\s+(run|stop|rollback|snapshot|serve)\b/i, "the loop controls itself; only `helper.sh verify|done|status` are for the coder"],
  [/\bhlib\.py\b[^\n]*(rollback|snapshot)/i, "the loop controls snapshots; only `helper.sh verify` is for the coder"],
  [/\bproxy\.py\b/i, "the gateway is the loop's; the coder only uses it as an HTTP endpoint"],
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

function auditInput(input: unknown): string {
  try {
    const s = JSON.stringify(input);
    return s.length > 2000 ? s.slice(0, 2000) + "...[truncated]" : s;
  } catch {
    return "?";
  }
}

function pathsIn(command: string): string[] {
  // absolute Windows / POSIX paths plus repo-relative helper/tool paths.
  const abs = command.match(/[A-Za-z]:[\\/][^\s"'|;&<>]*|(?:^|[\s"'=])\/[A-Za-z0-9_.\-/]+/g) ?? [];
  const rel = command.match(/(?:^|[\s"'=])(tools\/openrouter_helper\/[^\s"'|;&<>]*|embodied_human\/[^\s"'|;&<>]*|orchestrator\/[^\s"'|;&<>]*|\.env[^\s"'|;&<>]*)/g) ?? [];
  return [...abs, ...rel].map((s) => s.trim().replace(/^["'=]/, ""));
}

// Split `cmd1 && cmd2; cmd3` and track `cd <dir>` so `cd helper && echo x > f` is caught.
function segmentsWithCwd(command: string): { seg: string; cwd: string }[] {
  const parts = command.split(/(?:&&|\|\||;|\n)/);
  let cwd = ROOT;
  const out: { seg: string; cwd: string }[] = [];
  for (const raw of parts) {
    const seg = raw.trim();
    const m = /^(?:cd|chdir|Set-Location|sl)\s+([^\s;|&]+)/i.exec(seg);
    if (m) {
      const target = m[1].replace(/^["']|["']$/g, "");
      cwd = path.resolve(cwd, target);
    }
    out.push({ seg, cwd });
  }
  return out;
}

function hasRedirection(seg: string): boolean {
  // `>` / `>>` / `2>` / `N>&M`, but not `->` or `=>`.
  return /(^|[\s;|&0-9])(>>?)\s*\S/.test(seg);
}

export default function (pi: ExtensionAPI) {
  let calls = 0;

  pi.on("tool_call", async (event: any) => {
    const name: string = event.toolName ?? event.name ?? "?";
    const input: any = event.input ?? event.args ?? {};
    calls += 1;
    const block = (reason: string) => {
      audit({ tool: name, blocked: true, reason, input: auditInput(input) });
      return { block: true, reason: `Blocked by the permission guard: ${reason}.` };
    };

    if (!ALLOWED_TOOLS.has(name)) return block(`unknown tool '${name}': only ${[...ALLOWED_TOOLS].join(",")} are granted`);
    if (calls > MAX_CALLS) return block(`tool-call budget of ${MAX_CALLS} used up; finish with a report`);

    if (name === "bash" || name === "powershell") {
      const cmd: string = String(input.command ?? input.script ?? "");
      for (const [re, why] of BLOCKED_CMD) if (re.test(cmd)) return block(why);
      // Resolve each segment against its `cd`-adjusted cwd; any write resolving
      // inside PROTECTED_WRITE is refused even when spelled relatively.
      for (const { seg, cwd } of segmentsWithCwd(cmd)) {
        const lower = seg.toLowerCase();
        const isWriteSeg = hasRedirection(seg) || /\b(tee|set-content|add-content|out-file|\bmv\b|\bcp\b|\brm\b|\bmkdir\b|sed\s+-i|open\s*\([^)]*['"][wax])/.test(seg);
        // Relative helper mentions with a write primitive (cd-bypass catch).
        if (/openrouter_helper/i.test(seg) && isWriteSeg) return block("writes into the helper");
        for (const p of pathsIn(seg)) {
          const abs = path.resolve(cwd, p);
          if (SECRET_PATTERNS.some((re) => re.test(abs))) return block("touches secrets");
          if (PROTECTED_WRITE.some((d) => inside(abs, d)) && isWriteSeg) return block(`protected path (${path.relative(ROOT, abs)})`);
          if (isWriteSeg && !inside(abs, ROOT)) {
            const tmp = process.env.TEMP ? path.resolve(process.env.TEMP) : "/tmp";
            const tmp2 = process.env.TMPDIR ? path.resolve(process.env.TMPDIR) : tmp;
            if (!inside(abs, tmp) && !inside(abs, tmp2)) {
              // No blanket exception for /usr, /etc, C:\Windows: outside the
              // project (and outside temp) is refused when writing.
              return block(`writes outside the project (${abs})`);
            }
          }
        }
        // ` lower` kept for future severity tiers; silence unused-var linters.
        void lower;
      }
    } else if (name === "write" || name === "edit") {
      const raw: string = String(input.path ?? input.file_path ?? input.file ?? "");
      const abs = path.resolve(ROOT, raw);
      if (!inside(abs, ROOT)) return block(`writes outside the project (${abs})`);
      if (PROTECTED_WRITE.some((d) => inside(abs, d))) return block(`protected path (${path.relative(ROOT, abs)})`);
      if (SECRET_PATTERNS.some((re) => re.test(abs))) return block("touches secrets");
      if (!writableByPolicy(abs)) return block(`outside the writable allow-list (see prompts/rules.md): ${path.relative(ROOT, abs)}`);
    } else if (name === "read" || name === "grep" || name === "find" || name === "ls") {
      const raw: string = String(input.path ?? input.file_path ?? input.file ?? input.pattern ?? "");
      if (raw) {
        const abs = path.resolve(ROOT, raw);
        if (SECRET_PATTERNS.some((re) => re.test(abs))) return block("touches secrets");
      }
    }
    audit({ tool: name, blocked: false, input: auditInput(input).slice(0, 300) });
    return undefined;
  });
}
