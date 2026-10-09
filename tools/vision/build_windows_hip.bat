@echo off
rem strata-vision on an AMD card, on Windows: the image encoder of Strata's multimodal path built with ROCm's clang
rem and llama.cpp's ggml-hip (the same ggml-cuda kernels the NVIDIA build compiles, compiled for AMD).  This script
rem builds it; running it is the server's job - a "vision" entry in the server config starts it (docs/DETAILS.md).
rem
rem   tools\vision\build_windows_hip.bat              -> build-vision-hip\bin\strata-vision.exe + the .dll files it needs
rem   tools\vision\build_windows_hip.bat install      -> also copy them into engine-hip\, the AMD image folder
rem
rem Needs: Visual Studio Build Tools with the C++ workload (the linker, the CRT, the Windows SDK), cmake and ninja
rem (setup's .venv has both), and a ROCm install WITH a compiler: a full SDK, or the one setup installs in .rocm-win.
rem Not the engine's engine\rocm\bin - that one carries runtime DLLs only, no compiler.
rem
rem Which MSVC toolset can be used is decided by a compile, not by a version number.  <cmath> of the newest toolsets
rem (14.51, Visual Studio 18) declares isgreater and its five friends as host+device builtins, and the __device__
rem overloads in ROCm's HIP headers cannot overload those: every ggml-cuda file then fails with twenty errors that say
rem nothing about ggml.  14.44 (Build Tools 2022) compiles that same line.  It is a combination of the two sides, so
rem the script compiles that one line for this PC's card against every installed toolset and builds with the first
rem one that passes (Visual Studio's own vcvars scripts cannot be called twice in one environment, so each try runs in
rem its own command process, and the build itself runs in one that started with the vcvars call that worked - in
rem build-vision-hip\run-build.bat, which can be re-run by hand).
rem
rem Settings (environment variables, all optional):
rem   ROCM               the ROCm root (default: %ROCM_PATH% / %HIP_PATH%, then .rocm-win's rocm-sdk.exe)
rem   STRATA_HIP_ARCHS   the card to build for, e.g. gfx1201 (default: the card in this PC, read with amdgpu-arch)
rem   BUILD_DIR          default <repo>\build-vision-hip
rem   STRATA_GGML_DIR    a llama.cpp checkout at the pinned commit (default: third_party\llama.cpp)
setlocal EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "SRC=%%~fI"
if not defined BUILD_DIR set "BUILD_DIR=%SRC%\build-vision-hip"
set "INSTALL="
if /i "%~1"=="install" set "INSTALL=1"

rem ---- 1. ROCm: the HIP compiler, the runtime, hipBLAS
if not defined ROCM set "ROCM=%ROCM_PATH%"
if not defined ROCM set "ROCM=%HIP_PATH%"
if not defined ROCM if exist "%SRC%\.rocm-win\Scripts\rocm-sdk.exe" for /f "delims=" %%R in ('"%SRC%\.rocm-win\Scripts\rocm-sdk.exe" path --root') do set "ROCM=%%R"
if not defined ROCM echo no ROCm: set ROCM=^<the ROCm root^>, or run setup.py --backend hip once& exit /b 1
if not exist "%ROCM%\lib\llvm\bin\clang.exe" echo no C++ compiler in "%ROCM%" - that ROCm has no compiler& exit /b 1
set "ROCM_F=%ROCM:\=/%"
set "CLANG=%ROCM%\lib\llvm\bin\clang++.exe"

rem ---- the card to build for: the same amdgpu-arch CMake uses, so the try below asks a real question
set "ARCH=%STRATA_HIP_ARCHS%"
if defined ARCH for /f "delims=;" %%A in ("%ARCH%") do set "ARCH=%%A"
if not defined ARCH if exist "%ROCM%\lib\llvm\bin\amdgpu-arch.exe" for /f "delims=" %%A in ('"%ROCM%\lib\llvm\bin\amdgpu-arch.exe"') do if not defined ARCH set "ARCH=%%A"
if not defined ARCH if exist "%ROCM%\bin\amdgpu-arch.exe" for /f "delims=" %%A in ('"%ROCM%\bin\amdgpu-arch.exe"') do if not defined ARCH set "ARCH=%%A"
if not defined ARCH echo no AMD GPU found: set STRATA_HIP_ARCHS=gfx1100 (the card this build is for)& exit /b 1

rem ---- 2. the MSVC toolset: one line of <cmath> code, compiled as HIP for that card, decides it (see above)
set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
set "PROBE=%TEMP%\strata-hip-cmath-probe.cpp"
set "PBAT=%TEMP%\strata-hip-cmath-probe.bat"
> "%PROBE%" echo #include ^<cmath^>
>> "%PROBE%" echo int main() { return (int) isgreater(1.0, 2.0); }
set "VSDIR="
set "VCARGS="
rem vswhere lists the installations newest first: the first one with a toolset that passes is the one to build with.
if exist "%VSWHERE%" for /f "delims=" %%V in ('"%VSWHERE%" -all -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath') do if not defined VSDIR call :scan "%%V"
if not exist "%VSWHERE%" if not defined VSDIR call :scan "%ProgramFiles(x86)%\Microsoft Visual Studio"
if not exist "%VSWHERE%" if not defined VSDIR call :scan "%ProgramFiles%\Microsoft Visual Studio"
if defined VSDIR goto :toolset_found
echo No installed MSVC toolset compiles ^<cmath^> as HIP code for %ARCH%, and ggml-cuda would not compile with it
echo either.  Build Tools 2022 (MSVC 14.4x) do: the 14.5x headers declare isgreater and its five friends as
echo host+device builtins, which ROCm's HIP headers cannot overload.
exit /b 1
:toolset_found
echo Building with %VSDIR% %VCARGS% for %ARCH%, with ROCm from %ROCM%

rem ---- 3. configure + build, in one command process that starts with the vcvars call that worked
set "CMAKE=%SRC%\.venv\Scripts\cmake.exe"
if not exist "%CMAKE%" set "CMAKE=cmake"
set "NINJA=%SRC%\.venv\Scripts\ninja.exe"
set "EXTRA="
if exist "%NINJA%" set "EXTRA=!EXTRA! -DCMAKE_MAKE_PROGRAM=!NINJA!"
if defined STRATA_HIP_ARCHS set "EXTRA=!EXTRA! -DGPU_TARGETS=%STRATA_HIP_ARCHS:;= %"
if defined STRATA_GGML_DIR set "EXTRA=!EXTRA! -DLLAMA_DIR=%STRATA_GGML_DIR:\=/%"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"
set "RUN=%BUILD_DIR%\run-build.bat"
> "%RUN%" echo @echo off
>> "%RUN%" echo rem Generated by tools\vision\build_windows_hip.bat - re-run this file to build again.
>> "%RUN%" echo call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" %VCARGS% ^|^| exit /b 1
>> "%RUN%" echo set "HIP_PLATFORM=amd"
>> "%RUN%" echo set "ROCM_PATH=%ROCM%"
>> "%RUN%" echo set "HIP_PATH=%ROCM%"
>> "%RUN%" echo set "PATH=%ROCM%\bin;%ROCM%\lib\llvm\bin;%%PATH%%"
>> "%RUN%" echo "%CMAKE%" -G Ninja -S "%SRC%\tools\vision" -B "%BUILD_DIR%" -DCMAKE_BUILD_TYPE=Release !EXTRA! -DSTRATA_VISION_HIP=ON -DROCM_ROOT="%ROCM_F%" "-DCMAKE_C_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_CXX_COMPILER=%ROCM_F%/lib/llvm/bin/clang.exe" "-DCMAKE_PREFIX_PATH=%ROCM_F%" ^|^| exit /b 1
>> "%RUN%" echo "%CMAKE%" --build "%BUILD_DIR%" --target strata-vision ^|^| exit /b 1
call "%RUN%" || exit /b 1

rem ---- 4. install: the program and the libraries it loads, in the AMD image folder (engine\ is never written: the
rem engine that is already installed stays the one it was installed as; setuphip.py writes the config that uses this)
set "BIN=%BUILD_DIR%\bin"
echo Built: %BIN%\strata-vision.exe
if not defined INSTALL goto :done
if not exist "%SRC%\engine-hip" mkdir "%SRC%\engine-hip"
copy /y "%BIN%\strata-vision.exe" "%SRC%\engine-hip\" >nul || exit /b 1
for %%F in ("%BIN%\*.dll") do copy /y "%%F" "%SRC%\engine-hip\" >nul
echo Installed: engine-hip\strata-vision.exe and the llama/ggml libraries it loads.  It reaches the ROCm runtime
echo through the same engine\rocm\bin the engine uses; the server puts that on the PATH from the config's lib_dirs.
echo To run it: START-HERE-HIP.bat writes the server config and the start script that point at it.
:done
endlocal
goto :eof

rem ---- the two tries: the toolsets of one Visual Studio installation (:scan), then one toolset (:probe).
rem The default toolset first: -vcvars_ver needs version files a single-toolset install does not have.
:scan
call :probe "%~1" ""
if defined VSDIR goto :eof
for /d %%T in ("%~1\VC\Tools\MSVC\*") do if not defined VSDIR call :probe "%~1" "%%~nxT"
goto :eof

:probe
set "VCARGS="
if not "%~2"=="" set "VCARGS=-vcvars_ver=%~2"
> "%PBAT%" echo @echo off
>> "%PBAT%" echo call "%~1\VC\Auxiliary\Build\vcvars64.bat" %VCARGS% ^>nul 2^>nul ^&^& "%CLANG%" -x hip --offload-arch=%ARCH% -isystem "%ROCM%\include" -c "%PROBE%" -o "%TEMP%\strata-hip-cmath-probe.obj"
>> "%PBAT%" echo exit /b %%ERRORLEVEL%%
cmd /c call "%PBAT%" >nul 2>nul && (set "VSDIR=%~1"&set "VCARGS=%VCARGS%")
goto :eof
