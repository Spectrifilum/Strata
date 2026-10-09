#!/usr/bin/env python3
"""setuphip.py - Strata's image encoder on an AMD GPU, as a second path beside the one you already run.

setup.py (START-HERE.bat) keeps images on the CPU when the card is AMD: its `--vision gpu` builds the encoder with
CUDA.  The same encoder builds for AMD too - the identical ggml-cuda sources compiled for ROCm
(tools/vision/CMakeLists.txt with -DSTRATA_VISION_HIP=ON, docs/AMD_HIP.md), which
tools/vision/build_windows_hip.bat knows how to configure.  This script runs that second path and leaves the first
one alone: it never writes inside engine\\, never edits the model's own config, and never touches its start script.
It builds the encoder, installs it in engine-hip\\, and writes a config of its own, strata-<model>-hip.json, plus
run-<model>-hip.bat, whose server listens on the model's own port - the number its clients are already configured
for. The two paths therefore take turns rather than running side by side (only one can hold that port, and only one
engine fits in RAM anyway); --port moves this one if you want both. The original bootstrap stays exactly as it was.

    START-HERE-HIP.bat                       build the encoder, write the config, print how to start it
    START-HERE-HIP.bat --run                 ... and start that server here
    START-HERE-HIP.bat --config strata-<model>.json
                                             the model to add images to (found, or asked, if left out)
    START-HERE-HIP.bat --rocm <dir>         the ROCm install WITH a compiler (%ROCM_PATH%/%HIP_PATH%/.rocm-win)
    START-HERE-HIP.bat --archs gfx1100      the card to build for (default: the card in this PC)
    START-HERE-HIP.bat --rebuild            compile again although the encoder's source did not change

Nothing here is needed to go back: engine-hip\\, strata-<model>-hip.json and run-<model>-hip.bat are the only
files it adds; delete those three and the PC runs exactly what it ran before.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import setup as S            # the installer's helpers, read-only: say/ok/warn/fail, source_hash, VEXE, VISION_SOURCES

ENG = ROOT / "engine"                    # the installed engine: this script only reads it
HIP = ROOT / "engine-hip"                # this path's program and the libraries it loads
BUILD_DIR = ROOT / "build-vision-hip"   # its build tree (gitignored, like every build*/)
BUILD_SH = ROOT / "tools" / "vision" / "build_windows_hip.bat"
RESERVE_NOTE = ("the image encoder's VRAM on AMD has not been measured; if the engine gets slower after the encoder "
                "has warmed up, add \"--vram-reserve-mib\", 1400 to the \"args\" list of this file")


def config_paths():
    """The model configs of this folder: the ones setup.py writes, strata-<model>.json.  Its shared-settings files
    sit beside them and are not configs, neither is this path's own output (strata-<model>-hip.json), and a
    hand-named config is not looked for here - pass it with --config."""
    found = []
    for f in sorted(ROOT.glob("strata-*.json")):
        if f.name.endswith((".shared-settings.json", "-hip.json")):
            continue
        try:
            cfg = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(cfg, dict) and cfg.get("exe") and isinstance(cfg.get("args"), list):
            found.append(f)
    return found


def pick_config(want, yes):
    """The model to add images to: --config, the one config this folder has, or a question."""
    if want:
        p = Path(want)
        if not p.is_absolute():
            p = ROOT / p
        if not p.exists():
            S.fail(f"no such config: {p}")
        return p
    found = config_paths()
    if not found:
        S.fail("this folder has no model config yet", "run START-HERE.bat first - it installs the model and writes one")
    if len(found) == 1:
        S.ok(f"the model of {found[0].name}")
        return found[0]
    if yes:
        newest = max(found, key=lambda f: f.stat().st_mtime)
        S.ok(f"{len(found)} configs here, using the newest: {newest.name}")
        return newest
    say("\n  This folder holds more than one model:")
    for i, f in enumerate(found, 1):
        say(f"    {i}. {f.name}    (last changed {time.ctime(f.stat().st_mtime)})")
    try:
        n = input("  Which model gets the AMD image encoder? [1] ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        S.fail("this run has no terminal to answer in",
               "pass --config strata-<model>.json for the model to use, or --yes for the newest one")
    try:
        return found[int(n) - 1]
    except (ValueError, IndexError):
        S.fail(f"not a choice from that list: {n}")


def say(msg=""):
    print(msg, flush=True)


def model_tag(base):
    """The model's short name, the one setup.py names its files by (strata-<tag>.json, run-<tag>.bat), taken from
    the config it wrote.  This path's files keep that same name with -hip on the end."""
    return base.stem[len("strata-"):] if base.stem.startswith("strata-") else base.stem


def mmproj_for(model):
    """The model's mmproj file: it sits beside its shards (setup.py downloads it there)."""
    for d in (model.parent, model.parent.parent):
        hits = sorted(d.glob("mmproj-*.gguf"))
        if hits:
            return hits[0]
    S.fail(f"no mmproj-*.gguf next to {model.name}", "run START-HERE.bat --vision yes once: it downloads the "
                                                    "vision encoder, or pass --mmproj <path>")


def build_cache():
    """What the build's CMakeCache.txt says about the ROCm and the card it built for - the build script owns that
    choice, and placing the runtime libraries correctly needs to know it."""
    out = {"rocm": None, "archs": []}
    cache = BUILD_DIR / "CMakeCache.txt"
    if not cache.exists():
        return out
    for line in cache.read_text(encoding="utf-8", errors="replace"):
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        val = val.strip()
        if key.startswith("ROCM_ROOT"):
            out["rocm"] = Path(val) if val else None
        elif key.startswith(("GPU_TARGETS", "CMAKE_HIP_ARCHITECTURES")):
            out["archs"] += [a for a in re.split(r"[;,]", val) if a and a not in out["archs"]]
    return out


def libraries(base_cfg, exe_dir):
    """hipBLAS and rocBLAS for the encoder, with the kernel data they read from next to themselves.

    The engine's own ROCm folder (the model config's lib_dirs) already carries all of it, and the encoder was
    measured working with that: what has to sit next to the program is only the HIP runtime of the ROCm it was
    compiled with, which CMake copies there - Windows looks in the program's own folder and then in System32 before
    it looks at the PATH, so the driver's runtime would otherwise be used and it does not match a newer compiler
    (measured: access violation on the first weight copy).  When that folder has no hipBLAS, copy it from the ROCm
    the encoder was built with; hipBLASLt's kernels are per card, so only the built card's part of it."""
    for d in (Path(x) for x in base_cfg.get("lib_dirs") or []):
        if (d / "hipblas.dll").exists() and (d / "rocblas" / "library").is_dir():
            S.ok(f"hipBLAS/rocBLAS: the engine's runtime in {d}, its kernel data stays there")
            return
    src = build_cache()["rocm"]
    src = src / "bin" if src else None
    if not src or not src.is_dir():
        S.warn("the engine's runtime has no hipBLAS and the build's ROCm is unknown: the encoder may not find "
               "hipBLAS when it starts")
        return
    archs = build_cache()["archs"]
    copied = 0
    for f in ("hipblas.dll", "libhipblaslt.dll", "rocblas.dll", "rocsolver.dll", "amd_comgr.dll",
              "libtensilelite-host.dll", "rocm_kpack.dll", "origami.dll"):
        if (src / f).exists():
            shutil.copy2(src / f, exe_dir / f)
            copied += 1
    if (src / "rocblas" / "library").is_dir():              # rocBLAS reads this beside its own DLL
        shutil.copytree(src / "rocblas", exe_dir / "rocblas", dirs_exist_ok=True)
    for arch in archs:                                      # hipBLASLt: one directory per card
        lib = src / "hipblaslt" / "library" / arch
        if lib.is_dir():
            shutil.copytree(lib, exe_dir / "hipblaslt" / "library" / arch, dirs_exist_ok=True)
    S.ok(f"hipBLAS/rocBLAS: copied from {src} ({copied} libraries, kernel data for "
         f"{', '.join(archs) or 'no listed card'})")


def build_encoder(rocm, archs, force, base_cfg):
    """Compile the encoder for AMD through tools/vision/build_windows_hip.bat - that script owns the ROCm search,
    the card and the choice of MSVC toolset, and this script does not repeat it.  The build tree keeps a stamp of
    the source it was made from (setup.source_hash), so a second run does not spend the minutes again."""
    stamp = HIP / "BUILD.json"
    tstamp = BUILD_DIR / "strata-vision-stamp.json"
    vsrc = S.source_hash(S.VISION_SOURCES)
    have = {}
    if stamp.exists():
        try:
            have = json.loads(stamp.read_text(encoding="utf-8"))
        except ValueError:
            have = {}
    if not force and (HIP / S.VEXE).exists() and have.get("vision_src") == vsrc:
        S.ok(f"image encoder already built for this PC: {HIP / S.VEXE}")
        return
    exe = BUILD_DIR / "bin" / S.VEXE
    if not BUILD_SH.exists():
        S.fail(f"missing {BUILD_SH}")
    made_here = False
    if not force and exe.exists() and tstamp.exists():
        try:
            made_here = json.loads(tstamp.read_text(encoding="utf-8")).get("vision_src") == vsrc
        except ValueError:
            made_here = False
    if made_here:
        S.ok("the build tree holds this source's encoder: reusing it")
    elif not force and exe.exists():
        # build_windows_hip.bat run by hand leaves no stamp: take what is there and stamp it from now on, so the
        # next run knows.  Say that the source was not compared - --rebuild is the answer if it might be older.
        S.warn(f"reusing the encoder in {BUILD_DIR}: its source was not compared (it was built by "
               f"{BUILD_SH.name}); pass --rebuild to compile it again")
    else:
        say("  Compiling the image encoder for your AMD GPU (tens of minutes, once; docs/AMD_HIP.md) ...")
        env = {**os.environ, "ROCM": str(rocm) if rocm else "", "BUILD_DIR": str(BUILD_DIR)}
        if archs:
            env["STRATA_HIP_ARCHS"] = archs
        S.run(["cmd", "/c", str(BUILD_SH)], env={k: v for k, v in env.items() if v})
    if not exe.exists():
        S.fail("the build did not leave an encoder where its own script said it would", str(exe))
    HIP.mkdir(exist_ok=True)
    # The AMD build links llama/ggml shared, so the libraries travel with the program; the ROCm runtime itself comes
    # from the engine's own engine\rocm\bin, which the server already puts on the PATH through the config's lib_dirs.
    for f in [exe, *(BUILD_DIR / "bin").glob("*.dll")]:
        shutil.copy2(f, HIP / f.name)
    libraries(base_cfg, HIP)
    made = {"backend": "hip-vision", "version": S.source_version(),
            "rocm": str(rocm or "found by build_windows_hip.bat"), "archs": archs or "this PC's card",
            "vision_src": vsrc}
    record = json.dumps(made, indent=1)
    stamp.write_text(record, encoding="utf-8")
    tstamp.write_text(record, encoding="utf-8")
    S.ok(f"image encoder: {HIP / S.VEXE}")


def port_open(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1.5) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(description="Strata's image encoder on an AMD GPU, as a separate path")
    ap.add_argument("--setup", action="store_true", help="(START-HERE-HIP.bat --setup: run the same steps again)")
    ap.add_argument("--config", help="the model's server config to add images to (default: the strata-<model>.json "
                                     "of this folder)")
    ap.add_argument("--mmproj", help="the model's mmproj file (default: the one beside its model shards)")
    ap.add_argument("--rocm", help="the ROCm install with a compiler (default: the ROCM_PATH / HIP_PATH "
                                   "variables, or .rocm-win)")
    ap.add_argument("--archs", help="the card to build for, e.g. gfx1201 (default: the card in this PC)")
    ap.add_argument("--vision", choices=["gpu", "cpu"], default="gpu",
                    help="where pictures are read: on the card (the default, and the point of this path) or on the "
                         "CPU - the CPU mode starts the same encoder without --gpu, which leaves the card and its "
                         "VRAM to the engine and costs about 10-30 s a picture")
    ap.add_argument("--rebuild", action="store_true", help="compile again although the source did not change")
    ap.add_argument("--port", type=int, help="move this path's server off that port (default: the model config's "
                                            "own port, so existing clients keep working unchanged)")
    ap.add_argument("--run", action="store_true", help="start that server when everything is in place")
    ap.add_argument("--yes", action="store_true", help="no questions: the one config here, or the newest")
    a = ap.parse_args()

    if os.name != "nt":
        S.fail("this path is the Windows AMD image encoder; on Linux build it with the cmake lines in docs/AMD_HIP.md")
    if not (ENG / S.EXE).exists():
        S.fail("there is no engine in engine\\ yet", "run START-HERE.bat first - it installs the engine and the model")
    base = pick_config(a.config, a.yes)
    cfg = json.loads(base.read_text(encoding="utf-8"))
    if Path(cfg["exe"]).resolve() != (ENG / S.EXE).resolve():
        S.warn(f"{base.name}'s engine is not the one in engine\\: the new config keeps that same exe line")

    # The pieces the vision entry needs are already in the model's config: its first shard is --native, and the
    # mmproj sits beside the shards.
    args = cfg["args"]
    shard = None
    for i, one in enumerate(args):
        if one in ("--native", "--gguf", "-m") and i + 1 < len(args):
            shard = Path(args[i + 1])
    if not shard or not shard.exists():
        S.fail(f"{base.name} does not name a model file this script can read (--native ...)")
    mmproj = Path(a.mmproj) if a.mmproj else mmproj_for(shard)
    if not mmproj.exists():
        S.fail(f"no such mmproj file: {mmproj}")

    build_encoder(a.rocm, a.archs, a.rebuild, cfg)

    # A config of our own: the model's, plus the encoder, on this path's own port, with its own engine log.  Named
    # the way setup.py names its files, with -hip on the end, so it is clear which model and which path it belongs
    # to.  The model's own config is only ever read.
    tag = model_tag(base)
    out = ROOT / f"strata-{tag}-hip.json"
    base_port = int(cfg.get("port") or 8095)
    port = a.port or base_port          # the model's own port, so a client keeps one address: this path answers the
                                        # number it already uses, and takes its turn rather than running beside it
    new = dict(cfg)
    # The same keys setup.py writes for an image encoder, so one reader handles both paths; "gpu" here is the
    # encoder's own switch (serve/server.py adds --gpu), not the top-level "gpu" that names a card.
    mode = a.vision
    new["vision"] = {"exe": str(HIP / S.VEXE), "mmproj": str(mmproj), "model": str(shard),
                     "gpu": mode == "gpu", "max_tokens": S.VISION[mode]["max_tokens"]}
    if mode == "cpu":
        new["vision"]["threads"] = max(1, (os.cpu_count() or 8) // 2)
    # The engine has to be told to take image embeddings at all: src/program/generate.cpp answers "this engine was
    # started without --vision" to a prompt that carries one.  The model's own config was written with images off,
    # so its args have no such flag; the first run of this path died on exactly that.  Same rule as setup.py: the
    # flag, and the VRAM the encoder is given.
    reserve = str(S.VISION[mode]["reserve_mib"])
    if "--vision" not in new["args"]:
        new["args"] += ["--vision", "--vram-reserve-mib", reserve]
        S.ok(f"engine args: --vision --vram-reserve-mib {reserve} added ({base.name} was written without images)")
    new["port"] = port
    new["log"] = str(ROOT / f"strata-{tag}-hip.log")   # its own engine log: the model's stays the one you read
    out.write_text(json.dumps(new, indent=1) + "\n", encoding="utf-8")
    S.ok(f"config: {out.name} (port {port}, pictures on the {mode}; {base.name} is only ever read)")

    # the start script of this path, the way setup.py writes its own (serve/server.py, the new config, the new port)
    serve_cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                 "--config", str(out), "--port", str(port), "--open"]
    bat = ROOT / f"run-{tag}-hip.bat"
    # The mode of the pictures is the config's, not a command line of this script - serve/server.py has no --vision.
    # Someone will type one here anyway, so say where it belongs instead of quietly ignoring it.
    bat.write_text("@echo off\r\ntitle Strata " + tag.upper() + " (AMD image encoder)\r\ncd /d \"" + str(ROOT) +
                   "\"\r\n"
                   "rem This script takes no options: where pictures are read is written in the config.\r\n"
                   "rem Ask for it with START-HERE-HIP.bat --vision cpu or --vision gpu, which rewrites strata-" +
                   tag + "-hip.json.\r\n"
                   "if not \"%~1\"==\"\" (\r\n"
                   "  echo This start script takes no options, so what you typed is ignored.\r\n"
                   "  echo To choose where pictures are read:  START-HERE-HIP.bat --vision cpu   or --vision gpu\r\n"
                   "  echo Starting as configured now: pictures on the " + mode + ".\r\n" +
                   ")\r\n"
                   + " ".join(f'"{x}"' for x in serve_cmd) + "\r\nif errorlevel 1 pause\r\n",
                   encoding="utf-8")
    S.ok(f"start it: {bat.name}")
    if port == base_port:
        say(f"  (this server answers on port {port}, the number {base.name}'s own start script uses: the two paths "
            "take turns, they cannot both run at once - stop the other one first, or give this one --port)")
    say(f"  (to go back: delete {HIP.name}\\, {out.name} and {bat.name}; {ENG.name}\\ was not touched)")
    if mode == "gpu":
        say(f"  (and note: {RESERVE_NOTE})")
    else:
        say("  (the card keeps its whole VRAM to the engine this way; START-HERE-HIP.bat without --vision cpu puts "
            "the encoder back on the card - the same build serves both)")

    if not a.run:
        say(f"\n  Double-click {bat.name}, or start it here with START-HERE-HIP.bat --run.")
        return
    if port_open(port):
        # One port, one server: the model's own path may be holding it, and only one engine fits in RAM anyway.  The
        # answer is to stop that server, or move this path with --port.
        S.fail(f"something already answers on port {port}"
               + (f" - that is the model's own server, {base.name}" if port == base_port else ""),
               "stop it first and start this path again, or give this path another port with --port")
    say(f"\n  Starting the server with the AMD image encoder on port {port} (Ctrl+C stops it) ...")
    subprocess.run(serve_cmd, cwd=str(ROOT))


if __name__ == "__main__":
    main()
