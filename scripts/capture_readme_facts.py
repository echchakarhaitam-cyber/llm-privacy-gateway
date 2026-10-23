"""Run the README's quick start and the test suites for real and store what they print.

Usage:  python scripts/capture_readme_facts.py [--tests]

Writes results/readme_facts.json, which ``scripts/make_results.py --readme`` embeds into
README.md, so no output shown in the README is typed by hand:

* ``quickstart_detect``: a Python snippet (detect + mask + unmask with the regex detector,
  which needs no model) and its real stdout;
* ``quickstart_gateway``: the gateway started with ``--mock-upstream``, one ``curl`` request
  and its real response (status line, request-id header, JSON body);
* ``tests`` (with ``--tests``): the last line of ``pytest -q``, ``pytest -q -m "data or
  models"`` and ``pytest -q -m slow``, and the wall time of each.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OUT = Path("results/readme_facts.json")

DETECT_SNIPPET = '''\
from privgate.detectors.regex import RegexDetector
from privgate.masking import mask, unmask

text = "Contact Anna Lee at anna.lee@example.org or +33 6 12 34 56 78. Card: 4111 1111 1111 1111."
spans = RegexDetector().detect(text)
print([(s.label.value, text[s.start:s.end]) for s in spans])

masked = mask(text, spans, "typed")                 # reversible placeholders
print(masked.text)
print(dict(masked.mapping))
print(unmask("Please write to <EMAIL_1>.", masked.mapping))
'''

REQUEST = {"model": "demo", "messages": [
    {"role": "user", "content": "Email anna.lee@example.org and call +33 6 12 34 56 78."}]}


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", env=env,
                          timeout=900, check=False, **kw)


def capture_detect() -> dict:
    result = run([sys.executable, "-c", DETECT_SNIPPET])
    if result.returncode:
        raise SystemExit(f"snippet failed:\n{result.stderr}")
    return {"code": DETECT_SNIPPET, "output": result.stdout.rstrip("\n")}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def capture_gateway() -> dict:
    curl = shutil.which("curl")
    if not curl:
        raise SystemExit("curl is needed for the gateway quick start")
    port = free_port()
    server = subprocess.Popen(
        [sys.executable, "-m", "privgate.gateway", "--mock-upstream", "--detector", "regex",
         "--port", str(port)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                if urllib.request.urlopen(base + "/healthz", timeout=1).status == 200:
                    break
            except OSError:
                if server.poll() is not None:
                    raise SystemExit(f"gateway exited early: {server.stderr.read()}") from None
                time.sleep(0.5)
        body = json.dumps(REQUEST)
        reply = run([curl, "-s", "-i", f"{base}/v1/chat/completions", "-H",
                     "Content-Type: application/json", "-d", body])
        health = run([curl, "-s", f"{base}/healthz"])
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
    head, _, payload = reply.stdout.replace("\r\n", "\n").partition("\n\n")
    lines = [ln for ln in head.split("\n")
             if ln.startswith("HTTP/") or ln.lower().startswith("x-privgate-request-id")]
    return {
        "command": "python -m privgate.gateway --mock-upstream --detector regex --port 8088",
        "curl": ("curl -s -i http://127.0.0.1:8088/v1/chat/completions "
                 "-H 'Content-Type: application/json' -d '" + body + "'"),
        "response_headers": lines, "response_json": json.loads(payload),
        "healthz": json.loads(health.stdout),
    }


def capture_tests() -> dict:
    out = {}
    for name, extra in (("default", []), ("data_or_models", ["-m", "data or models"]),
                        ("slow", ["-m", "slow"])):
        t0 = time.perf_counter()
        result = run([sys.executable, "-m", "pytest", "-q", "--show-capture=no", *extra])
        last = [ln for ln in result.stdout.strip().splitlines() if ln.strip()][-1]
        shown = " ".join(f'"{x}"' if " " in x else x for x in extra)
        out[name] = {"command": "pytest -q" + (" " + shown if extra else ""),
                     "summary": last, "returncode": result.returncode,
                     "seconds": round(time.perf_counter() - t0, 1)}
        print(name, "->", last, flush=True)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tests", action="store_true", help="also run the three test suites")
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args(argv)
    facts = json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else {}
    facts["quickstart_detect"] = capture_detect()
    facts["quickstart_gateway"] = capture_gateway()
    if args.tests:
        facts["tests"] = capture_tests()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8",
                        newline="\n")
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
