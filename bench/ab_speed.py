#!/usr/bin/env python3
"""A/B speed sweep for this machine, run by you, not by the agent.

    python bench/ab_speed.py --list                          show the variants
    python bench/ab_speed.py --check-only                    is the card free?
    python bench/ab_speed.py                                 baseline + the promising ones, 1 round
    python bench/ab_speed.py --pairs baseline pcie-auto      interleaved A/B/A/B of two variants
    python bench/ab_speed.py --variants pcie-auto,draft-en   only these (baseline always runs first)
    python bench/ab_speed.py --rounds 3                      three engine starts per variant

The sweep starts its own engine on its own port and never touches a running server.
One graphics card holds one engine, so unload whatever is using Strata first
(the web page's unload button, or Ctrl+C on the server window). The sweep checks this
and stops with a clear message if the card is busy.

Measured per variant, from the engine's own timings in the answer's `timings` block:
  decode tok/s      three short greedy chats, 200 tokens each
  prompt tok/s      one fresh 4K and one fresh 16K prompt (unique tag, nothing cached)
  agent re-reads    a six-turn session that grows like an agent client: how many prompt
                    tokens each turn had to read again, and the session's wall time
It also keeps the engine's start-up lines (expert cache slots, VRAM free, PCIe share,
tail skip), because those explain the differences.

Results: bench/ab/<date>-rx9070-sweep/results.json and summary.md.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "strata-q2_0.json"
OUT_ROOT = ROOT / "bench" / "ab"

FILLER = (
    "The lighthouse keeper logged the weather every evening: wind from the east, a low "
    "tide, two ships in the harbour. She wrote the numbers in a ledger, and the ledger "
    "filled page by page through the long autumn, one line for every hour. "
)
SYSTEM = "You are a careful code assistant. Answer in two sentences. Do not call tools."
AGENT_FILES = [
    ("src/app.py", "The entry point. It parses arguments, opens the config, starts the "
                   "server and hands requests to the engine. Error handling is in a helper "
                   "module. Logging goes to a rotating file." * 25),
    ("src/engine.py", "Wraps the engine process. It writes the GEN line, reads tokens until "
                      "the stop token, and reports timings. It restarts the engine if it dies." * 25),
    ("src/cache.py", "Keeps the expert cache. It ranks experts by use, swaps them in and out, "
                     "and pins the buffers the driver allows. The swap policy is one function." * 25),
    ("src/agent.py", "The agent loop. It sends the conversation, reads the tool call, runs it, "
                     "and appends the result. It caps the context and rewinds when the cap is hit." * 25),
    ("src/monitor.py", "The monitor tab. It samples token rates, cache hits and draft "
                       "acceptance, and writes them to a per-request table." * 25),
    ("src/web.py", "The web app. It serves the page, the model list and the sampling drawer, "
                   "and talks to the same API the other clients use." * 25),
]

# Three decode prompts of the kind the project's own gate uses: a story, a code answer,
# an explanation. Repetitive output (a list of numbers) flatters draft acceptance, so it
# is not used here.
DECODE_PROMPTS = [
    "Write 200 words about a lighthouse keeper who finds a locked chest on the shore at low tide.",
    "Write a Python function that reads a JSONL file and returns the ten records with the largest "
    "'score' field, with type hints and a docstring, then a short usage example.",
    "Explain in 200 words how a mixture-of-experts model routes a token, and what happens when the "
    "expert cache on the graphics card misses.",
]

# name -> (args to add, args to remove, env to add, what it does)
VARIANTS = {
    "baseline": ({}, [], {}, "the config as it is now"),
    "pcie-auto": ({}, ["--pcie-frac", "0.00"], {},
                  "let the engine probe the link and refine the PCIe share (PcieAuto) "
                  "instead of forcing every missed expert through the CPU pool"),
    "pcie-0.25": ({"--pcie-frac": "0.25"}, ["--pcie-frac", "0.00"], {},
                  "a fixed quarter of the missed experts on the GPU, the rest on the CPU pool"),
    "draft-en": ({"--draft-vocab": "en"}, [], {},
                 "the small English/code draft vocabulary: ~110 MiB less VRAM, English answers 1-2% faster"),
    "cache-every-4096": ({"--prompt-cache-every": "4096"}, [], {},
                        "checkpoints every 4K tokens instead of 16K: a follow-up turn resumes closer to the end"),
    "conv-cache": ({"--conversation-cache-mib": "8192", "--conversation-cache-slots": "4"}, [], {},
                   "park up to four conversations in 8 GiB of host RAM (off by default)"),
    "select-wmma": ({}, [], {"STRATA_SELECT_WMMA": "1"},
                    "matrix-core block scorer for the prompt's QSA top-k (+1.5% at 16K/262K on gfx12)"),
    "prefill-16k": ({}, [], {"STRATA_PREFILL_AUTO_MAX": "16384"},
                    "bigger prompt chunks on Windows (the 16K default is Linux-only today)"),
    "host-core-last": ({"--host-core": "last"}, [], {},
                      "the host thread on the last physical core: Windows sends GPU interrupts to core 0"),
    "reserve-1000": ({"--vram-reserve-mib": "1000"}, ["--vram-reserve-mib", "700"], {},
                     "more slack on the card; the docs show slack beats extra cache slots"),
    "adapt-overlap": ({}, [], {"STRATA_ADAPT_OVERLAP": "1"},
                      "the adaptive tier's copies land while the next window runs (opt-in on Windows/HIP)"),
    "adapt-async": ({"--adapt-async": "1"}, [], {},
                    "cache swaps advance between windows on a helper thread"),
    "no-pcie-group": ({}, [], {"STRATA_NO_PCIE_GROUP": "1"},
                      "with --pcie-frac 0: drop the empty PCIe stage from the window graph "
                      "(PR #2223, NOT in the 0.1.43 engine - only means something if you "
                      "build the engine with that PR)"),
    "hc-persist": ({}, [], {"STRATA_HC_PERSIST": "1"},
                   "the verifier's hyper-connection read as one kernel (PR #2226, NOT in the "
                   "0.1.43 engine - only means something if you build with that PR)"),
}

ORDER = ["baseline", "pcie-auto", "pcie-0.25", "draft-en", "cache-every-4096", "conv-cache",
         "select-wmma", "prefill-16k", "host-core-last", "reserve-1000", "adapt-overlap",
         "adapt-async", "no-pcie-group", "hc-persist"]
DEFAULT_SET = ["baseline", "pcie-auto", "pcie-0.25", "draft-en", "cache-every-4096",
               "select-wmma", "host-core-last"]


def load_config() -> dict:
    try:
        return json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read {CONFIG}: {exc}") from exc


def build_args(name: str) -> list[str]:
    add, drop, _, _ = VARIANTS[name]
    args = list(load_config()["args"])
    out: list[str] = []
    skipping = False
    for token in args:
        if skipping and not token.startswith("--"):
            skipping = False
            continue
        if token in drop:
            skipping = token.startswith("--")
            continue
        out.append(token)
    for flag, value in add.items():
        if flag in out:
            out[out.index(flag) + 1] = value
        else:
            out.extend([flag, value])
    return out


def build_env(name: str) -> dict:
    env = dict(load_config().get("env", {}))
    env.update(VARIANTS[name][2])
    return env


def server_config(name: str, port: int, out_dir: Path) -> Path:
    cfg = load_config()
    cfg["args"] = build_args(name)
    cfg["env"] = build_env(name)
    cfg["port"] = port
    cfg["log"] = str(out_dir / f"{name}.log")
    path = out_dir / f"{name}.json"
    path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return path


def api(port: int, path: str, body: dict | None = None, timeout: float = 900) -> dict | None:
    """One call to the local engine API: 127.0.0.1 only, http only."""
    conn = HTTPConnection("127.0.0.1", port, timeout=timeout)
    try:
        if body is None:
            conn.request("GET", path)
        else:
            conn.request("POST", path, body=json.dumps(body).encode(),
                         headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = resp.read().decode()
        if resp.status >= 400:
            return {"error": f"HTTP {resp.status}: {data[:200]}"}
        return json.loads(data)
    except Exception as exc:
        return {"error": str(exc)}
    finally:
        conn.close()


def card_free(port: int) -> bool:
    """True when no Strata server on this port is holding the model."""
    health = api(port, "/health", timeout=10)
    return not (health and health.get("loaded"))


def chat(port: int, messages: list, max_tokens: int) -> dict:
    body = {"model": "strata", "temperature": 0, "max_tokens": max_tokens,
            "reasoning_effort": "none", "messages": messages}
    t0 = time.time()
    answer = api(port, "/v1/chat/completions", body)
    if not answer or "error" in answer:
        return {"ok": False, "err": (answer or {}).get("error", "no response")}
    timings = answer.get("timings") or {}
    return {"ok": True, "wall_s": time.time() - t0,
            "prompt_n": timings.get("prompt_n"), "prompt_ms": timings.get("prompt_ms"),
            "decode_ms": timings.get("decode_ms"),
            "prompt_tps": timings.get("prompt_per_second"),
            "decode_tps": timings.get("predicted_per_second"),
            "drafts": f"{timings.get('drafts_accepted')}/{timings.get('drafts_offered')}"}


def startup_lines(log_path: Path) -> dict:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    found = {}
    for key, needle in (("cache_slots", "expert cache"), ("vram_free", "MiB of VRAM free"),
                        ("pcie", "PCIe"), ("tail_skip", "tail skip"), ("pool", "pool workers")):
        hits = [line.strip() for line in text.splitlines() if needle.lower() in line.lower()]
        if hits:
            found[key] = hits[-1][:200]
    return found


def measure(port: int) -> dict:
    """The suite: three short chats, a fresh 4K and a fresh 16K prompt, a six-turn agent session."""
    decodes = [chat(port, [{"role": "user", "content": f"[dec {i}] {p}"}], 200)
               for i, p in enumerate(DECODE_PROMPTS)]
    prompts = [chat(port, [{"role": "user",
                            "content": f"[p4k {i}]\n" + FILLER * 60 + "\nSummarise in one sentence."}], 24)
               for i in range(2)]
    big = chat(port, [{"role": "user",
                       "content": f"[p16k {int(time.time())}]\n" + FILLER * 240 + "\nSummarise in one sentence."}], 24)

    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "I will show you project files one at a time."}]
    agent = []
    for rel, file_body in AGENT_FILES:
        messages.append({"role": "user", "content": f"File {rel}:\n{file_body}\n\nWhat is this file for?"})
        reply = chat(port, messages, 64)
        agent.append(reply)
        messages.append({"role": "assistant", "content": "A module file." if reply.get("ok") else "n/a"})

    wall = sum(a.get("wall_s", 0.0) for a in agent)
    read_tokens = [a.get("prompt_n") or 0 for a in agent]
    return {
        "ok": True,
        "decode_tps": median([d["decode_tps"] for d in decodes if d.get("decode_tps")] or [0.0]),
        "decode_runs": [d.get("decode_tps") for d in decodes],
        "decode_drafts": [d.get("drafts") for d in decodes],
        "prompt4k_tps": median([p["prompt_tps"] for p in prompts if p.get("prompt_tps")] or [0.0]),
        "prompt16k_tps": big.get("prompt_tps"),
        "agent_wall_s": round(wall, 1),
        "agent_read_tokens": read_tokens,
        "agent_tps": round(sum(read_tokens) / max(1.0, wall), 1),
        "agent": agent,
    }


def run_variant(name: str, port: int, out_dir: Path) -> dict:
    cfg_path = server_config(name, port, out_dir)
    proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"),
                             "--engine", "strata", "--config", str(cfg_path), "--port", str(port)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    result = {"variant": name, "ok": False}
    try:
        deadline = time.time() + 420
        while time.time() < deadline:
            health = api(port, "/health", timeout=10)
            if health and health.get("loaded"):
                break
            time.sleep(3)
        else:
            result["err"] = "engine did not reach READY within 420 s"
            return result

        result.update(startup_lines(out_dir / f"{name}.log"))

        chat(port, [{"role": "user", "content": "What is 17 times 3?"}], 40)   # warm-up
        result.update(measure(port))
    finally:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            proc.terminate()
            try:
                proc.wait(timeout=25)
            except Exception:
                proc.kill()
        time.sleep(12)
    return result


def write_summary(out_dir: Path, date: str, results: list) -> None:
    lines = [f"# A/B sweep {date} - RX 9070 (gfx1201), Q2_0, engine 0.1.43", "",
             "| variant | decode tok/s | 4K prompt tok/s | 16K prompt tok/s | agent tok/s | agent wall s | cache slots | VRAM free |",
             "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |"]
    for r in results:
        if not r.get("ok"):
            lines.append(f"| {r['variant']} | failed | | | | | | {r.get('err', '')} |")
            continue
        lines.append(f"| {r['variant']} | {r['decode_tps']:.1f} | {r['prompt4k_tps']:.1f} | "
                     f"{r['prompt16k_tps'] or 0:.1f} | {r['agent_tps']:.1f} | {r['agent_wall_s']:.0f} | "
                     f"{r.get('cache_slots', '')[-45:]} | {r.get('vram_free', '')[-45:]} |")
    lines.append("")
    lines.append("Agent column: tokens read (not reused) per turn summed over six turns, divided by the session's wall time.")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the variants and stop")
    ap.add_argument("--variants", default="", help="comma separated names (default: the promising set)")
    ap.add_argument("--all", action="store_true", help="run every variant in --list")
    ap.add_argument("--live", action="store_true",
                    help="measure the server that is already running, without restarting anything")
    ap.add_argument("--pairs", nargs=2, metavar=("A", "B"), help="interleaved A/B of two variants")
    ap.add_argument("--rounds", type=int, default=1, help="engine starts per variant (default 1)")
    ap.add_argument("--port", type=int, default=12356, help="port for the sweep's own server")
    ap.add_argument("--check-only", action="store_true", help="only report whether the card is free")
    ap.add_argument("--force", action="store_true", help="run even if a server holds the card")
    args = ap.parse_args()

    if args.list:
        for name in ORDER:
            print(f"{name:<16} {VARIANTS[name][3]}")
        return 0

    base_port = load_config().get("port", 12345)
    busy = not card_free(base_port)
    if args.check_only:
        print(f"port {base_port}: model loaded={busy} -> card {'busy' if busy else 'free'}")
        return 0

    if args.live:
        if busy is False:
            print(f"nothing is loaded on port {base_port}; start your server first, or run the sweep.")
            return 1
        date = datetime.date.today().isoformat()
        out_dir = OUT_ROOT / f"{date}-live"
        out_dir.mkdir(parents=True, exist_ok=True)
        chat(base_port, [{"role": "user", "content": "What is 17 times 3?"}], 40)
        r = measure(base_port)
        r["variant"] = "live (current config, no restart)"
        (out_dir / "results.json").write_text(json.dumps({"date": date, "results": [r]}, indent=2),
                                             encoding="utf-8")
        write_summary(out_dir, date, [r])
        print(f"decode {r['decode_tps']:.1f} tok/s | 4K {r['prompt4k_tps']:.1f} | "
              f"16K {r['prompt16k_tps'] or 0:.1f} | agent {r['agent_tps']:.1f} tok/s "
              f"({r['agent_wall_s']:.0f} s)")
        print(f"per turn, tokens read again: {r['agent_read_tokens']}")
        print(f"summary: {out_dir / 'summary.md'}")
        return 0

    if busy and not args.force:
        print(f"A Strata server is loaded on port {base_port}; one card holds one engine.")
        print("Unload it first (the web page's unload button, or Ctrl+C on the server window),")
        print("then run this again. --force runs anyway if you know that server is idle.")
        return 1

    if args.pairs:
        names = list(args.pairs) * max(1, args.rounds)
    else:
        wanted = [v.strip() for v in args.variants.split(",") if v.strip()]
        if wanted:
            chosen = [v for v in ORDER if v in wanted]
            if "baseline" not in chosen:
                chosen = ["baseline"] + chosen
        else:
            chosen = ORDER if args.all else DEFAULT_SET
        names = [n for n in chosen for _ in range(max(1, args.rounds))]

    for name in names:
        if name not in VARIANTS:
            print(f"unknown variant: {name}")
            return 1

    date = datetime.date.today().isoformat()
    out_dir = OUT_ROOT / f"{date}-rx9070-sweep"
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    print(f"{len(names)} engine starts, results in {out_dir}")
    for i, name in enumerate(names, 1):
        print(f"[{i}/{len(names)}] {name} ...", flush=True)
        r = run_variant(name, args.port, out_dir)
        results.append(r)
        if r.get("ok"):
            print(f"    decode {r['decode_tps']:.1f} tok/s | 4K {r['prompt4k_tps']:.1f} | "
                  f"16K {r['prompt16k_tps'] or 0:.1f} | agent {r['agent_tps']:.1f} tok/s "
                  f"({r['agent_wall_s']:.0f} s)", flush=True)
        else:
            print(f"    failed: {r.get('err')}", flush=True)

    (out_dir / "results.json").write_text(json.dumps({"date": date, "results": results}, indent=2),
                                          encoding="utf-8")
    write_summary(out_dir, date, results)
    print(f"\nsummary: {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
