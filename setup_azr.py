#!/usr/bin/env python
"""
Getting the Absolute Zero Reasoner (AZR) behind the body.

Nothing is downloaded unless you pass ``--download``; this script otherwise
only checks and explains.

    python setup_azr.py                 # what you have, what to do next
    python setup_azr.py --check         # is an AZR server answering?
    python setup_azr.py --download 3b   # fetch the 3B GGUF (1.93 GB, Q4_K_M)

AZR (Zhao et al., "Absolute Zero: Reinforced Self-play Reasoning with Zero
Data", Tsinghua / BIGAI / Penn State; github.com/LeapLabTHU/Absolute-Zero-Reasoner)
is a Qwen2.5-Coder model.  The released checkpoints are on Hugging Face:

    andrewzh/Absolute_Zero_Reasoner-Coder-3b        (safetensors, ~6 GB in fp16)
    andrewzh/Absolute_Zero_Reasoner-Coder-7b
    andrewzh/Absolute_Zero_Reasoner-Coder-14b

Community GGUF quantisations (what you want on a 6 GB GPU):

    bartowski/andrewzh_Absolute_Zero_Reasoner-Coder-3b-GGUF
        ...-Q4_K_M.gguf   1.93 GB    <- recommended
        ...-Q5_K_M.gguf   2.22 GB
        ...-Q8_0.gguf     3.29 GB
    bartowski/andrewzh_Absolute_Zero_Reasoner-Coder-7b-GGUF
        ...-Q4_K_M.gguf   4.68 GB    (fits a 6 GB card with a short context,
                                      but leaves little room for anything else)

Serving it
----------
The simplest route on Windows needs no Python packages and no compiler:

1. Download a prebuilt ``llama-server`` from the llama.cpp releases page
   (github.com/ggml-org/llama.cpp/releases; the ``win-cuda`` zip for an NVIDIA
   card, or the ``win-vulkan`` / ``win-cpu`` zip otherwise).
2. Start it::

       llama-server -m andrewzh_Absolute_Zero_Reasoner-Coder-3b-Q4_K_M.gguf ^
                    -ngl 99 -c 3072 --port 8080

3. Run the person with it::

       python run_mind.py --backend server --url http://127.0.0.1:8080/v1

vLLM (which the AZR repository itself uses) or Ollama work the same way, since
the backend only needs an OpenAI-style ``/v1/completions`` endpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

FILES = {
    "3b": ("bartowski/andrewzh_Absolute_Zero_Reasoner-Coder-3b-GGUF",
           "andrewzh_Absolute_Zero_Reasoner-Coder-3b-Q4_K_M.gguf", "1.93 GB"),
    "7b": ("bartowski/andrewzh_Absolute_Zero_Reasoner-Coder-7b-GGUF",
           "andrewzh_Absolute_Zero_Reasoner-Coder-7b-Q4_K_M.gguf", "4.68 GB"),
}


def check(url: str) -> int:
    import requests
    try:
        r = requests.get(url.rstrip("/") + "/models", timeout=3)
        print(f"{url}: HTTP {r.status_code}")
        print(r.text[:400])
        return 0 if r.ok else 1
    except Exception as exc:
        print(f"no server at {url}: {exc}")
        return 1


def download(size: str, dest: Path) -> int:
    import requests
    repo, name, human = FILES[size]
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / name
    print(f"downloading {name} ({human}) from {url}\n  -> {out}")
    with requests.get(url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        got = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if total:
                    print(f"\r  {got / 1e9:5.2f} / {total / 1e9:.2f} GB", end="", flush=True)
    print("\ndone.  Start llama-server with:")
    print(f"  llama-server -m {out} -ngl 99 -c 3072 --port 8080")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--download", choices=list(FILES), default=None,
                    help="download the Q4_K_M GGUF of this size (asks nothing further)")
    ap.add_argument("--dest", type=Path, default=Path("models"))
    a = ap.parse_args()
    if a.download:
        return download(a.download, a.dest)
    if a.check:
        return check(a.url)
    print(__doc__)
    ggufs = list(Path("models").glob("*.gguf")) if Path("models").exists() else []
    print("GGUF files in ./models :", [str(p) for p in ggufs] or "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
