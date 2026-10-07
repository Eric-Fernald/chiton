@echo off
setlocal
rem chiton.cmd - shim that locates the real chiton binary and hands off to it.
rem
rem Resolution order:
rem   1. %CHITON_BIN%
rem   2. chiton.exe beside this shim (bundled in plugin releases)
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

if not defined CHITON_EXE if exist "%~dp0chiton.exe" set "CHITON_EXE=%~dp0chiton.exe"

if not defined CHITON_EXE (
	for /f "delims=" %%I in ('where chiton.exe 2^>nul') do (
		if not defined CHITON_EXE set "CHITON_EXE=%%I"
	)
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
	echo Plugin releases bundle chiton.exe beside this shim. Without it, install 1>&2
	echo chiton with any of: 1>&2
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

"%CHITON_EXE%" %*
exit /b %errorlevel%
