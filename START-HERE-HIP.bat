@echo off
rem Strata's second path: the image encoder on an AMD GPU (ROCm), beside the runtime you already have.
rem It builds tools\vision for HIP and writes its own config and start script; engine\, the model's config and its
rem start script are only read.  To go back: delete engine-hip, *-hip.json and run-*-hip.bat.  Nothing here installs
rem Python or the engine - run START-HERE.bat for that, and this path uses what it put there.
rem
rem Needs, for the build only: a ROCm install WITH a compiler (a full SDK, or the .rocm-win of the engine's own
rem Windows HIP build) and Visual Studio Build Tools with the C++ workload.  docs/AMD_HIP.md, "Images: the encoder
rem on an AMD card", says what it does and what has not been measured.
setlocal
title Strata (AMD image encoder)
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo  There is no Python environment here yet.
  echo  Run START-HERE.bat first - it installs Python, the engine and the model; this path adds the
  echo  image encoder to what they left, and changes none of it.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" setuphip.py %*
if errorlevel 1 pause
exit /b
