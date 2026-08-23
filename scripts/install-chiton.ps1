<#
.SYNOPSIS
    Fetch a prebuilt chiton binary and install it locally.

.DESCRIPTION
    Installs into $env:CLAUDE_PLUGIN_DATA\bin when running as part of the Claude
    Code plugin, otherwise ~\.chiton\bin. Downloads the release asset for the
    host architecture, verifies its SHA-256 against the release's checksums.txt,
    and stops on any mismatch. The script never executes anything it downloads.

    The repository is private, so release assets are NOT reachable through the
    github.com/<owner>/<repo>/releases/download/ browser URL — that endpoint is
    session-authenticated and returns 404 for a token. Assets come from the REST
    API instead, either through the GitHub CLI or through the per-asset API URL.

.PARAMETER Version
    Release tag to install. Defaults to $env:CHITON_VERSION, then "latest".

.PARAMETER Dir
    Install directory. Defaults to $env:CHITON_INSTALL_DIR, then the plugin data
    directory, then ~\.chiton\bin.

.NOTES
    Windows PowerShell 5.1 compatible. Authenticate with `gh auth login`, or set
    $env:GITHUB_TOKEN to a token with repo:read.

    Run it with an explicit policy override, because Windows client SKUs default
    to Restricted:

        powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install-chiton.ps1
#>
[CmdletBinding()]
param(
    [string]$Version,
    [string]$Dir
)

$ErrorActionPreference = 'Stop'
$Repo = 'Eric-Fernald/chiton'
$Api = "https://api.github.com/repos/$Repo"

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
    Write-Error "install-chiton: $Message"
    exit 1
}

function Have([string]$Name) {
    $null -ne (Get-Command $Name -ErrorAction SilentlyContinue)
}

function Show-NoAccess {
    Write-Host @"
install-chiton: could not read releases for $Repo.

The repository is private and may also have no published releases yet. Do one of:
  * gh auth login                                 (then re-run — simplest route)
  * `$env:GITHUB_TOKEN = '<repo:read PAT>'         (then re-run)
  * build from source: git clone the repo and run 'make build'
"@
    exit 1
}

# ---- host detection -------------------------------------------------------

$arch = $env:PROCESSOR_ARCHITECTURE
switch ($arch) {
    'AMD64' { $goarch = 'amd64' }
    'ARM64' { Die 'no prebuilt windows/arm64 release; build from source (see CONTRIBUTING.md)' }
    default { Die "unsupported architecture: $arch" }
}

# TLS 1.2 is not the default on stock Windows PowerShell 5.1.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$apiHeaders = @{ 'User-Agent' = 'install-chiton'; 'Accept' = 'application/vnd.github+json' }
if ($env:GITHUB_TOKEN) { $apiHeaders['Authorization'] = "Bearer $env:GITHUB_TOKEN" }

# Download one release asset by its API url.
#
# The API redirects to storage on another host. Windows PowerShell re-sends
# custom headers across a redirect, which would hand the token to that host, so
# the redirect is stopped and re-issued without any headers.
function Get-Asset([string]$Url, [string]$OutFile) {
    $headers = $apiHeaders.Clone()
    $headers['Accept'] = 'application/octet-stream'
    try {
        Invoke-WebRequest -Uri $Url -Headers $headers -OutFile $OutFile `
            -UseBasicParsing -MaximumRedirection 0 -ErrorAction Stop
        return
    } catch {
        $resp = $_.Exception.Response
        if (-not $resp) { throw }

        $code = 0
        try { $code = [int]$resp.StatusCode } catch { }
        if ($code -lt 300 -or $code -ge 400) { throw }

        $location = $null
        try { $location = $resp.Headers['Location'] } catch { }
        if (-not $location -and $resp.Headers.Location) { $location = $resp.Headers.Location.AbsoluteUri }
        if (-not $location) { throw }

        # Signed storage URL: carries its own credentials, needs none of ours.
        Invoke-WebRequest -Uri $location -OutFile $OutFile -UseBasicParsing -ErrorAction Stop
    }
}

$tmp = Join-Path ([IO.Path]::GetTempPath()) ("chiton-" + [Guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp -Force | Out-Null

try {
    # ---- resolve the release ----------------------------------------------

    $useGh = $false
    if (Have 'gh') {
        & gh auth status *> $null
        if ($LASTEXITCODE -eq 0) { $useGh = $true }
    }

    if ($useGh) {
        if ($Version -eq 'latest') {
            $tag = (& gh release view --repo $Repo --json tagName --jq .tagName 2>$null)
            if ($LASTEXITCODE -ne 0 -or -not $tag) { Show-NoAccess }
        } else {
            $tag = $Version
        }
    } else {
        $url = "$Api/releases/latest"
        if ($Version -ne 'latest') { $url = "$Api/releases/tags/$Version" }
        try {
            $release = Invoke-RestMethod -Uri $url -Headers $apiHeaders -UseBasicParsing
        } catch {
            Show-NoAccess
        }
        $tag = $release.tag_name
        if (-not $tag) { Die 'could not read a release tag from the GitHub API response' }
    }

    $asset = "chiton-$tag-windows-$goarch.exe"
    $assetPath = Join-Path $tmp $asset
    $sumsPath = Join-Path $tmp 'checksums.txt'

    Write-Host "install-chiton: installing $asset from $tag"

    # ---- download ---------------------------------------------------------

    if ($useGh) {
        & gh release download $tag --repo $Repo --dir $tmp --pattern $asset --pattern 'checksums.txt' --clobber
        if ($LASTEXITCODE -ne 0) { Die "gh could not download $asset and checksums.txt from $tag" }
    } else {
        $assetEntry = $release.assets | Where-Object { $_.name -eq $asset } | Select-Object -First 1
        $sumsEntry = $release.assets | Where-Object { $_.name -eq 'checksums.txt' } | Select-Object -First 1
        if (-not $assetEntry) { Die "release $tag has no asset named $asset" }
        if (-not $sumsEntry) { Die "release $tag publishes no checksums.txt - refusing to install an unverified binary" }

        try { Get-Asset $assetEntry.url $assetPath } catch { Die "download failed for ${asset}: $($_.Exception.Message)" }
        try { Get-Asset $sumsEntry.url $sumsPath } catch { Die "download failed for checksums.txt: $($_.Exception.Message)" }
    }

    if (-not (Test-Path $assetPath)) { Die "download did not produce $asset" }
    if (-not (Test-Path $sumsPath)) { Die "no checksums.txt for $tag - refusing to install an unverified binary" }

    # ---- verify -----------------------------------------------------------

    $line = Select-String -Path $sumsPath -Pattern ([regex]::Escape($asset)) | Select-Object -First 1
    if (-not $line) { Die "checksums.txt has no entry for $asset - refusing to install" }
    $expected = ($line.Line -split '\s+')[0]

    $actual = (Get-FileHash -Path $assetPath -Algorithm SHA256).Hash
    if ($actual -ne $expected.ToUpper()) {
        Die "checksum mismatch for $asset (expected $expected, got $actual) - refusing to install"
    }
    Write-Host 'install-chiton: sha256 verified'

    # ---- install ----------------------------------------------------------

    New-Item -ItemType Directory -Path $Dir -Force | Out-Null
    $target = Join-Path $Dir 'chiton.exe'
    Copy-Item -Path $assetPath -Destination $target -Force

    # Deliberately not running the binary to report its version: this script
    # must not execute what it downloaded.
    Write-Host "install-chiton: installed $tag -> $target"
} finally {
    Remove-Item -Path $tmp -Recurse -Force -ErrorAction SilentlyContinue
}

$onPath = ($env:PATH -split ';') -contains $Dir
if (-not $onPath) {
    Write-Host @"

$Dir is not on your PATH. The chiton MCP server launches the binary by name, so
add it for future sessions:

  [Environment]::SetEnvironmentVariable('PATH', "$Dir;" + [Environment]::GetEnvironmentVariable('PATH','User'), 'User')

Then restart Claude Code (or run /reload-plugins) so the MCP server picks it up.
"@
}
