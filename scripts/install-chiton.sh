#!/bin/sh
# install-chiton.sh — fetch the prebuilt chiton binary for this machine and
# install it locally.
#
# Downloads exactly one release asset, the one for the host OS/arch, verifies
# its SHA-256 against a checksums.txt, and stops on any mismatch. The script
# never executes anything it downloads.
#
# The plugin's bin/chiton launcher runs this on first use with --checksums
# pointing at the checksums.txt shipped inside the plugin, so the binary is
# checked against hashes that arrived with the plugin rather than from the same
# place as the binary. Run by hand, it fetches the release's own checksums.txt.
#
# Assets come from the release download URLs on github.com; no API calls, no
# credentials, no rate limit. Needs curl or wget, and sha256sum or shasum.
#
# Environment:
#   CHITON_VERSION      release tag to install (default: latest)
#   CHITON_INSTALL_DIR  override the install directory
#
# Usage: sh scripts/install-chiton.sh [--version vX.Y.Z] [--dir /path/to/bin]
#                                     [--checksums <file>] [--quiet]

set -eu

REPO="Eric-Fernald/chiton"
VERSION="${CHITON_VERSION:-latest}"
INSTALL_DIR="${CHITON_INSTALL_DIR:-}"
CHECKSUMS=""
QUIET=""

usage() {
	cat <<'EOF'
install-chiton.sh — fetch the prebuilt chiton binary for this machine.

  --version vX.Y.Z    release tag to install (default: latest)
  --dir <path>        install directory (default: $CLAUDE_PLUGIN_DATA/bin,
                      else ~/.chiton/bin)
  --checksums <file>  verify against this checksums.txt instead of the
                      release's own
  --quiet             print errors only
EOF
}

while [ $# -gt 0 ]; do
	case "$1" in
	--version | --dir | --checksums)
		[ $# -ge 2 ] || {
			echo "install-chiton: $1 needs a value" >&2
			exit 2
		}
		case "$1" in
		--version) VERSION="$2" ;;
		--dir) INSTALL_DIR="$2" ;;
		--checksums) CHECKSUMS="$2" ;;
		esac
		shift 2
		;;
	--quiet)
		QUIET=1
		shift
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

say() {
	[ -n "$QUIET" ] || echo "install-chiton: $*"
}

have() {
	command -v "$1" >/dev/null 2>&1
}

# fetch <url> <output-path>
if have curl; then
	fetch() { curl -fsSL --retry 2 -o "$2" "$1"; }
elif have wget; then
	fetch() { wget -q -O "$2" "$1"; }
else
	die "need curl or wget to download a release"
fi

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

# There is no windows/arm64 build; Windows on Arm runs the amd64 one under
# emulation.
[ "$OS" = windows ] && ARCH=amd64

if [ -n "$CHECKSUMS" ] && [ ! -f "$CHECKSUMS" ]; then
	die "checksums file not found: $CHECKSUMS"
fi

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT INT TERM

# ---- resolve the tag ------------------------------------------------------

if [ "$VERSION" = latest ]; then
	# github.com/<repo>/releases/latest redirects to .../releases/tag/<tag>.
	# Reading the redirect avoids the rate-limited REST API.
	if have curl; then
		effective="$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest" 2>/dev/null)" || effective=""
	else
		effective="$(wget -q -S --max-redirect=0 -O /dev/null "https://github.com/$REPO/releases/latest" 2>&1 | sed -n 's/^ *[Ll]ocation: *//p' | tr -d '\r' | tail -1)" || effective=""
	fi
	case "$effective" in
	*/releases/tag/*) tag="${effective##*/releases/tag/}" ;;
	*)
		die "could not find a published release of $REPO. Install with
  go install github.com/Eric-Fernald/chiton/cmd/chiton@latest
or build from source ('make build')."
		;;
	esac
else
	tag="$VERSION"
fi

asset="chiton-$tag-$OS-$ARCH$EXT"
base="https://github.com/$REPO/releases/download/$tag"
say "downloading $asset"

# ---- download -------------------------------------------------------------

fetch "$base/$asset" "$tmpdir/$asset" || die "download failed: $base/$asset"

if [ -z "$CHECKSUMS" ]; then
	fetch "$base/checksums.txt" "$tmpdir/checksums.txt" ||
		die "release $tag publishes no checksums.txt — refusing to install an unverified binary"
	CHECKSUMS="$tmpdir/checksums.txt"
fi

# ---- verify ---------------------------------------------------------------
#
# checksums.txt is `sha256sum` output: "<hash>  <name>", or "<hash> *<name>" in
# binary mode. Match the name exactly so linux-amd64 cannot match linux-arm64.

expected="$(awk -v a="$asset" '{ n = $NF; sub(/^\*/, "", n) } n == a { print $1; exit }' "$CHECKSUMS")"
[ -n "$expected" ] || die "checksums.txt has no entry for $asset — refusing to install"

if have sha256sum; then
	actual="$(sha256sum "$tmpdir/$asset" | awk '{print $1}')"
elif have shasum; then
	actual="$(shasum -a 256 "$tmpdir/$asset" | awk '{print $1}')"
else
	die "no sha256sum or shasum available to verify the download — refusing to install"
fi

[ "$actual" = "$expected" ] || die "checksum mismatch for $asset (expected $expected, got $actual) — refusing to install"
say "sha256 verified"

# ---- install --------------------------------------------------------------
#
# Copy beside the target and rename over it, so a concurrent launcher never
# sees a half-written binary.

mkdir -p "$INSTALL_DIR"
target="$INSTALL_DIR/chiton$EXT"
cp "$tmpdir/$asset" "$target.tmp.$$"
chmod +x "$target.tmp.$$"
mv -f "$target.tmp.$$" "$target"

# Deliberately not running the binary to report its version: this script must
# not execute what it downloaded.
say "installed $tag -> $target"

[ -z "$QUIET" ] || exit 0
case ":${PATH}:" in
*":$INSTALL_DIR:"*) ;;
*)
	cat <<EOF

$INSTALL_DIR is not on your PATH. To run chiton by name, add it to your shell
profile:

  export PATH="$INSTALL_DIR:\$PATH"
EOF
	;;
esac
