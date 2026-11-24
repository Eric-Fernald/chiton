@echo off
setlocal
rem chiton.cmd - shim that locates the real chiton binary and hands off to it.
rem
rem On Windows the published plugin's .mcp.json command,
rem ${CLAUDE_PLUGIN_ROOT}/bin/chiton, resolves to this file, so it also starts
rem the MCP server. Every message goes to stderr: stdout is the protocol stream.
rem
rem A published plugin ships no binaries. It carries release\tag, the release it
rem was built from, and release\checksums.txt, that release's hashes. On first
rem use this script downloads chiton.exe into
rem %USERPROFILE%\.chiton\versions\<tag>\ and verifies it against those hashes.
rem
rem Resolution order:
rem   1. %CHITON_BIN%
rem   2. published plugin: %USERPROFILE%\.chiton\versions\<tag>\chiton.exe,
rem      downloading it first if it is missing
rem   3. chiton.exe on PATH (this shim is named chiton.cmd, so no recursion)
rem   4. %CLAUDE_PLUGIN_DATA%\bin\chiton.exe
rem   5. %USERPROFILE%\.chiton\bin\chiton.exe
rem
rem Delayed expansion is deliberately NOT enabled: it would mangle any candidate
rem path containing "!", and nothing here needs it ("if not defined" is
rem evaluated at run time, unlike "%VAR%").

set "CHITON_EXE="

rem "if exist <path>\" is true only for directories, so this rejects a
rem CHITON_BIN that points at a folder instead of falling over later.
if defined CHITON_BIN if exist "%CHITON_BIN%" if not exist "%CHITON_BIN%\" set "CHITON_EXE=%CHITON_BIN%"
if defined CHITON_EXE goto run

rem A source checkout has no release\tag and skips straight to PATH.
set "CHITON_TAG="
if exist "%~dp0..\release\tag" set /p CHITON_TAG=<"%~dp0..\release\tag"
if not defined CHITON_TAG goto search

set "CHITON_VDIR=%USERPROFILE%\.chiton\versions\%CHITON_TAG%"
if exist "%CHITON_VDIR%\chiton.exe" goto pinned
echo chiton: fetching the %CHITON_TAG% binary for this machine (first run only) 1>&2
powershell -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%~dp0..\scripts\install-chiton.ps1" -Version "%CHITON_TAG%" -Dir "%CHITON_VDIR%" -Checksums "%~dp0..\release\checksums.txt" -Quiet <NUL 1>&2
if errorlevel 1 echo chiton: download failed; looking for another chiton instead 1>&2
:pinned
if exist "%CHITON_VDIR%\chiton.exe" set "CHITON_EXE=%CHITON_VDIR%\chiton.exe"
if defined CHITON_EXE goto run

:search
for /f "delims=" %%I in ('where chiton.exe 2^>nul') do (
	if not defined CHITON_EXE set "CHITON_EXE=%%I"
)

if not defined CHITON_EXE if defined CLAUDE_PLUGIN_DATA (
	if exist "%CLAUDE_PLUGIN_DATA%\bin\chiton.exe" set "CHITON_EXE=%CLAUDE_PLUGIN_DATA%\bin\chiton.exe"
)

if not defined CHITON_EXE (
	if exist "%USERPROFILE%\.chiton\bin\chiton.exe" set "CHITON_EXE=%USERPROFILE%\.chiton\bin\chiton.exe"
)

if not defined CHITON_EXE (
	echo chiton: binary not found. 1>&2
	echo. 1>&2
	echo Install it with any of: 1>&2
	echo   powershell -NoProfile -ExecutionPolicy Bypass -File "%%CLAUDE_PLUGIN_ROOT%%\scripts\install-chiton.ps1" 1>&2
	echo   go install github.com/Eric-Fernald/chiton/cmd/chiton@latest 1>&2
	echo   make build 1>&2
	echo. 1>&2
	echo The -ExecutionPolicy override is required: Windows client SKUs default 1>&2
	echo to Restricted, which refuses to run any .ps1. 1>&2
	echo. 1>&2
	echo Or point %%CHITON_BIN%% at an existing binary. 1>&2
	exit /b 127
)

:run
"%CHITON_EXE%" %*
exit /b %errorlevel%
