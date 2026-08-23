#!/bin/sh
# install-chiton.sh — fetch a prebuilt chiton binary and install it locally.
#
# Installs into $CLAUDE_PLUGIN_DATA/bin when running as part of the Claude Code
# plugin, otherwise ~/.chiton/bin. Downloads the release asset for the host
# OS/arch, verifies its SHA-256 against the release's checksums.txt, and stops
# on any mismatch. The script never executes anything it downloads.
#
# The repository is private, so release assets are NOT reachable through the
# github.com/<owner>/<repo>/releases/download/ browser URL — that endpoint is
# session-authenticated and returns 404 for a token. Assets come from the REST
# API instead, either through the GitHub CLI or through the per-asset API URL.
#
# Environment:
#   CHITON_VERSION      release tag to install (default: latest)
#   CHITON_INSTALL_DIR  override the install directory
#   GITHUB_TOKEN        token with repo:read (or have `gh` authenticated)
#
# Usage: sh scripts/install-chiton.sh [--version vX.Y.Z] [--dir /path/to/bin]

set -eu

REPO="Eric-Fernald/chiton"
API="https://api.github.com/repos/$REPO"
VERSION="${CHITON_VERSION:-latest}"
INSTALL_DIR="${CHITON_INSTALL_DIR:-}"

usage() {
	cat <<'EOF'
install-chiton.sh — fetch a prebuilt chiton binary and install it locally.

  --version vX.Y.Z   release tag to install (default: latest)
  --dir <path>       install directory (default: $CLAUDE_PLUGIN_DATA/bin,
                     else ~/.chiton/bin)

The repository is private: authenticate with `gh auth login`, or export
GITHUB_TOKEN with repo:read. Without either, build from source instead.
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
	--version)
		VERSION="$2"
		shift 2
		;;
	--dir)
		INSTALL_DIR="$2"
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "install-chiton: unknown argument: $1" >&2
		exit 2
		;;
	esac
done

if [ -z "$INSTALL_DIR" ]; then
	if [ -n "${CLAUDE_PLUGIN_DATA:-}" ]; then
		INSTALL_DIR="$CLAUDE_PLUGIN_DATA/bin"
	else
		INSTALL_DIR="$HOME/.chiton/bin"
	fi
fi

die() {
	echo "install-chiton: $*" >&2
	exit 1
}

have() {
	command -v "$1" >/dev/null 2>&1
}

# ---- host detection -------------------------------------------------------

os_raw="$(uname -s)"
case "$os_raw" in
Linux) OS=linux ;;
Darwin) OS=darwin ;;
MINGW* | MSYS* | CYGWIN*) OS=windows ;;
*) die "unsupported OS: $os_raw — build from source instead (see CONTRIBUTING.md)" ;;
esac

arch_raw="$(uname -m)"
case "$arch_raw" in
x86_64 | amd64) ARCH=amd64 ;;
arm64 | aarch64) ARCH=arm64 ;;
*) die "unsupported architecture: $arch_raw — build from source instead (see CONTRIBUTING.md)" ;;
esac

EXT=""
[ "$OS" = windows ] && EXT=".exe"

if [ "$OS" = windows ] && [ "$ARCH" = arm64 ]; then
	die "no prebuilt windows/arm64 release; build from source (see CONTRIBUTING.md)"
fi

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT INT TERM

# ---- fetch the release ----------------------------------------------------
#
# Two routes, tried in order. Both work against a private repository; the
# browser download URL deliberately is not used, because it cannot.
#
#   1. gh — resolves the tag, handles auth, and downloads assets by name.
#   2. curl + jq — reads the release JSON, then GETs each asset's API URL with
#      Accept: application/octet-stream. curl drops the Authorization header on
#      the cross-host redirect to storage, so the token never leaves GitHub.

auth_header=""
[ -n "${GITHUB_TOKEN:-}" ] && auth_header="Authorization: Bearer $GITHUB_TOKEN"

api_get() {
	# api_get <url> <output-path> [accept]
	accept="${3:-application/vnd.github+json}"
	if [ -n "$auth_header" ]; then
		curl -fsSL -H "$auth_header" -H "Accept: $accept" -o "$2" "$1"
	else
		curl -fsSL -H "Accept: $accept" -o "$2" "$1"
	fi
}

no_access() {
	cat >&2 <<EOF
install-chiton: could not read releases for $REPO.

The repository is private and may also have no published releases yet. Do one of:
  * gh auth login                       (then re-run — this is the simplest route)
  * export GITHUB_TOKEN=<repo:read PAT> (then re-run)
  * build from source: git clone the repo and run 'make build'
EOF
	exit 1
}

if have gh && gh auth status >/dev/null 2>&1; then
	route=gh
	if [ "$VERSION" = latest ]; then
		tag="$(gh release view --repo "$REPO" --json tagName --jq .tagName 2>/dev/null)" || no_access
	else
		tag="$VERSION"
	fi
	[ -n "$tag" ] || no_access
else
	have curl || die "need either the GitHub CLI (gh) or curl to download a release"
	route=api
	url="$API/releases/latest"
	[ "$VERSION" != latest ] && url="$API/releases/tags/$VERSION"
	api_get "$url" "$tmpdir/release.json" 2>/dev/null || no_access
	tag="$(sed -n 's/.*"tag_name"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p' "$tmpdir/release.json" | head -1)"
	[ -n "$tag" ] || die "could not read a release tag from the GitHub API response"
fi

asset="chiton-$tag-$OS-$ARCH$EXT"
echo "install-chiton: installing $asset from $tag"

# ---- download -------------------------------------------------------------

if [ "$route" = gh ]; then
	gh release download "$tag" --repo "$REPO" --dir "$tmpdir" \
		--pattern "$asset" --pattern checksums.txt --clobber ||
		die "gh could not download $asset and checksums.txt from $tag"
else
	have jq || cat >&2 <<'EOF'
install-chiton: jq is required for the curl route, because a private
repository's assets can only be fetched by their API id, which lives in the
release JSON. Install jq, or authenticate the GitHub CLI (gh auth login) and
re-run — the gh route needs no JSON parsing.
EOF
	have jq || exit 1

	asset_url="$(jq -r --arg n "$asset" '.assets[] | select(.name == $n) | .url' "$tmpdir/release.json")"
	sums_url="$(jq -r '.assets[] | select(.name == "checksums.txt") | .url' "$tmpdir/release.json")"
	[ -n "$asset_url" ] && [ "$asset_url" != null ] || die "release $tag has no asset named $asset"
	[ -n "$sums_url" ] && [ "$sums_url" != null ] || die "release $tag publishes no checksums.txt — refusing to install an unverified binary"

	api_get "$asset_url" "$tmpdir/$asset" application/octet-stream || die "download failed: $asset"
	api_get "$sums_url" "$tmpdir/checksums.txt" application/octet-stream || die "download failed: checksums.txt"
fi

[ -f "$tmpdir/$asset" ] || die "download did not produce $asset"
[ -f "$tmpdir/checksums.txt" ] || die "no checksums.txt for $tag — refusing to install an unverified binary"

# ---- verify ---------------------------------------------------------------
#
# checksums.txt is `sha256sum` output: "<hash>  <name>", or "<hash> *<name>" in
# binary mode. Match either, anchored so linux-amd64 cannot match linux-arm64.

expected="$(grep " \{1,2\}\*\{0,1\}$asset\$" "$tmpdir/checksums.txt" | awk '{print $1}' | head -1)"
[ -n "$expected" ] || die "checksums.txt has no entry for $asset — refusing to install"

if have sha256sum; then
	actual="$(sha256sum "$tmpdir/$asset" | awk '{print $1}')"
elif have shasum; then
	actual="$(shasum -a 256 "$tmpdir/$asset" | awk '{print $1}')"
else
	die "no sha256sum or shasum available to verify the download — refusing to install"
fi

[ "$actual" = "$expected" ] || die "checksum mismatch for $asset (expected $expected, got $actual) — refusing to install"
echo "install-chiton: sha256 verified"

# ---- install --------------------------------------------------------------

mkdir -p "$INSTALL_DIR"
target="$INSTALL_DIR/chiton$EXT"
cp "$tmpdir/$asset" "$target"
chmod +x "$target"

# Deliberately not running the binary to report its version: this script must
# not execute what it downloaded.
echo "install-chiton: installed $tag -> $target"

case ":${PATH}:" in
*":$INSTALL_DIR:"*) ;;
*)
	cat <<EOF

$INSTALL_DIR is not on your PATH. The chiton MCP server launches the binary by
name, so add it to your shell profile:

  export PATH="$INSTALL_DIR:\$PATH"

Then restart Claude Code (or run /reload-plugins) so the MCP server picks it up.
EOF
	;;
esac
