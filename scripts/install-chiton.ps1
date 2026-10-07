<#
.SYNOPSIS
    Fetch the prebuilt chiton binary for this machine and install it locally.

.DESCRIPTION
    Downloads exactly one release asset, the Windows binary, verifies its
    SHA-256 against a checksums.txt, and stops on any mismatch. The script never
    executes anything it downloads.

    The plugin's bin\chiton.cmd launcher runs this on first use with -Checksums
    pointing at the checksums.txt shipped inside the plugin, so the binary is
    checked against hashes that arrived with the plugin rather than from the same
    place as the binary. Run by hand, it fetches the release's own checksums.txt.

    Assets come from the release download URLs on github.com; no API calls, no
    credentials, no rate limit.

.PARAMETER Version
    Release tag to install. Defaults to $env:CHITON_VERSION, then "latest".

.PARAMETER Dir
    Install directory. Defaults to $env:CHITON_INSTALL_DIR, then the plugin data
    directory, then ~\.chiton\bin.

.PARAMETER Checksums
    Verify against this checksums.txt instead of the release's own.

.PARAMETER Quiet
    Print errors only.

.NOTES
    Windows PowerShell 5.1 compatible. Run it with an explicit policy override,
    because Windows client SKUs default to Restricted:

        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install-chiton.ps1
#>
[CmdletBinding()]
param(
    [string]$Version,
    [string]$Dir,
    [string]$Checksums,
    [switch]$Quiet
)

$ErrorActionPreference = 'Stop'
# The progress bar slows Invoke-WebRequest down by an order of magnitude on 5.1.
$ProgressPreference = 'SilentlyContinue'
$Repo = 'Eric-Fernald/chiton'

if (-not $Version) { $Version = $env:CHITON_VERSION }
if (-not $Version) { $Version = 'latest' }

if (-not $Dir) { $Dir = $env:CHITON_INSTALL_DIR }
if (-not $Dir) {
    if ($env:CLAUDE_PLUGIN_DATA) {
        $Dir = Join-Path $env:CLAUDE_PLUGIN_DATA 'bin'
    } else {
        $Dir = Join-Path $HOME '.chiton\bin'
    }
}

function Die([string]$Message) {
    [Console]::Error.WriteLine("install-chiton: $Message")
    exit 1
}

function Say([string]$Message) {
    if (-not $Quiet) { Write-Host "install-chiton: $Message" }
}

# ---- host detection -------------------------------------------------------

# There is no windows/arm64 build; Windows on Arm runs the amd64 one under
# emulation.
switch ($env:PROCESSOR_ARCHITECTURE) {
    'AMD64' { $goarch = 'amd64' }
    'ARM64' { $goarch = 'amd64' }
    default { Die "unsupported architecture: $env:PROCESSOR_ARCHITECTURE" }
}

if ($Checksums -and -not (Test-Path -LiteralPath $Checksums -PathType Leaf)) {
    Die "checksums file not found: $Checksums"
}

# TLS 1.2 is not the default on stock Windows PowerShell 5.1.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

# ---- resolve the tag ------------------------------------------------------

if ($Version -eq 'latest') {
    # github.com/<repo>/releases/latest redirects to .../releases/tag/<tag>.
    # Reading the redirect avoids the rate-limited REST API.
    $tag = $null
    try {
        $req = [Net.HttpWebRequest]::Create("https://github.com/$Repo/releases/latest")
        $req.Method = 'HEAD'
        $req.AllowAutoRedirect = $false
        $req.UserAgent = 'install-chiton'
        $resp = $req.GetResponse()
        $location = $resp.Headers['Location']
        $resp.Close()
        if ($location -match '/releases/tag/([^/?#]+)$') { $tag = $Matches[1] }
    } catch { }
    if (-not $tag) {
        Die @"
could not find a published release of $Repo. Install with
  go install github.com/Eric-Fernald/chiton/cmd/chiton@latest
or build from source ('make build').
"@
    }
} else {
    $tag = $Version
}

$asset = "chiton-$tag-windows-$goarch.exe"
$base = "https://github.com/$Repo/releases/download/$tag"

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("chiton-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

try {
    # ---- download ---------------------------------------------------------

    Say "downloading $asset"
    $assetPath = Join-Path $tmp $asset
    try {
        Invoke-WebRequest -Uri "$base/$asset" -OutFile $assetPath -UseBasicParsing -UserAgent 'install-chiton'
    } catch {
        Die "download failed: $base/$asset ($($_.Exception.Message))"
    }

    if (-not $Checksums) {
        $Checksums = Join-Path $tmp 'checksums.txt'
        try {
            Invoke-WebRequest -Uri "$base/checksums.txt" -OutFile $Checksums -UseBasicParsing -UserAgent 'install-chiton'
        } catch {
            Die "release $tag publishes no checksums.txt - refusing to install an unverified binary"
        }
    }

    # ---- verify -----------------------------------------------------------
    #
    # checksums.txt is `sha256sum` output: "<hash>  <name>", or "<hash> *<name>"
    # in binary mode. Match the name exactly.

    $expected = $null
    foreach ($line in Get-Content -LiteralPath $Checksums) {
        $fields = $line.Trim() -split '\s+'
        if ($fields.Count -ge 2 -and $fields[-1].TrimStart('*') -eq $asset) {
            $expected = $fields[0]
            break
        }
    }
    if (-not $expected) { Die "checksums.txt has no entry for $asset - refusing to install" }

    $actual = (Get-FileHash -LiteralPath $assetPath -Algorithm SHA256).Hash
    if ($actual -ne $expected.ToUpper()) {
        Die "checksum mismatch for $asset (expected $expected, got $actual) - refusing to install"
    }
    Say 'sha256 verified'

    # ---- install ----------------------------------------------------------
    #
    # Copy beside the target and rename over it, so a concurrent launcher never
    # sees a half-written binary. If another launcher won the race and its copy
    # is already running, the rename fails; that copy is fine to keep when it
    # carries the same hash.

    New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    $target = Join-Path $Dir 'chiton.exe'
    $staged = "$target.tmp.$PID"
    Copy-Item -LiteralPath $assetPath -Destination $staged -Force
    try {
        Move-Item -LiteralPath $staged -Destination $target -Force
    } catch {
        Remove-Item -LiteralPath $staged -Force -ErrorAction SilentlyContinue
        $have = $null
        if (Test-Path -LiteralPath $target) { $have = (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash }
        if ($have -ne $actual) { Die "could not install to ${target}: $($_.Exception.Message)" }
    }

    # Deliberately not running the binary to report its version: this script
    # must not execute what it downloaded.
    Say "installed $tag -> $target"
} finally {
    Remove-Item -LiteralPath $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

if ($Quiet) { exit 0 }
$onPath = ($env:PATH -split ';') -contains $Dir
if (-not $onPath) {
    Write-Host @"

$Dir is not on your PATH. To run chiton by name, add it for future sessions:

  [Environment]::SetEnvironmentVariable('PATH', "$Dir;" + [Environment]::GetEnvironmentVariable('PATH','User'), 'User')
"@
}
