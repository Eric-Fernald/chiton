#!/bin/sh
# build-plugin-dist.sh — assemble the published Claude Code plugin, with its
# bundled binaries, into dist/plugin/.
#
# Run after `make release VERSION=<tag>`, which leaves the five cross-compiled
# binaries in dist/. The release workflow commits the result as the single
# commit of the plugin-dist branch, which .claude-plugin/marketplace.json on main
# points at.
#
# The tree differs from main in three ways: it carries the binaries, its
# .mcp.json launches the bundled launcher instead of `chiton` from PATH
# (packaging/plugin/.mcp.json), and it leaves out everything that is not
# plugin content.
#
# Usage: sh scripts/build-plugin-dist.sh <tag>

set -eu

tag="${1:?usage: build-plugin-dist.sh <tag>}"
case "$tag" in
v*) ;;
*)
	echo "build-plugin-dist: tag must start with 'v' (got '$tag')" >&2
	exit 2
	;;
esac
version="${tag#v}"

root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
dist="$root/dist"
out="$dist/plugin"

bin_for() {
	# bin_for <os> <arch> <ext> — the release artifact `make release` wrote.
	f="$dist/chiton-$tag-$1-$2$3"
	[ -f "$f" ] || {
		echo "build-plugin-dist: missing $f; run 'make release VERSION=$tag' first" >&2
		exit 1
	}
	printf '%s\n' "$f"
}

rm -rf "$out"
mkdir -p "$out/.claude-plugin" "$out/bin" "$out/scripts" "$out/assets"

# plugin.json with the version stamped from the tag. Claude Code only offers an
# update when this changes.
sed "s/\"version\": \"[^\"]*\"/\"version\": \"$version\"/" \
	"$root/.claude-plugin/plugin.json" >"$out/.claude-plugin/plugin.json"
grep -q "\"version\": \"$version\"" "$out/.claude-plugin/plugin.json" || {
	echo "build-plugin-dist: could not stamp version $version into plugin.json" >&2
	exit 1
}

cp "$root/packaging/plugin/.mcp.json" "$out/.mcp.json"
cp -R "$root/skills" "$out/skills"
cp "$root/bin/chiton" "$root/bin/chiton.cmd" "$out/bin/"
cp "$root/scripts/install-chiton.sh" "$root/scripts/install-chiton.ps1" "$out/scripts/"
cp "$root/assets/icon.png" "$out/assets/"
cp "$root/packaging/plugin/README.md" "$root/LICENSE" "$out/"

# Windows resolves `${CLAUDE_PLUGIN_ROOT}/bin/chiton` to bin/chiton.exe, so the
# native binary sits beside the POSIX launcher rather than under libexec/.
cp "$(bin_for windows amd64 .exe)" "$out/bin/chiton.exe"
for platform in linux/amd64 linux/arm64 darwin/amd64 darwin/arm64; do
	os=${platform%/*}
	arch=${platform#*/}
	mkdir -p "$out/libexec/$os-$arch"
	cp "$(bin_for "$os" "$arch" "")" "$out/libexec/$os-$arch/chiton"
done

chmod 0755 "$out/bin/chiton" "$out/scripts/install-chiton.sh" "$out"/libexec/*/chiton

echo "build-plugin-dist: assembled $out for $tag"
