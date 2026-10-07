#!/bin/sh
# build-plugin-dist.sh — assemble the published Claude Code plugin into
# dist/plugin/.
#
# Run after `make release VERSION=<tag>`, which leaves the cross-compiled
# binaries and dist/checksums.txt in dist/. The release workflow commits the
# result to the plugin-dist branch, which .claude-plugin/marketplace.json on
# main points at.
#
# The plugin ships no binaries. It pins the release instead: release/tag names
# it and release/checksums.txt carries its hashes. On first use, bin/chiton (or
# bin/chiton.cmd on Windows) downloads the single binary the machine needs from
# that release and verifies it against those hashes, so an install never
# downloads binaries for other platforms.
#
# The tree differs from main in three ways: it carries release/, its .mcp.json
# launches the bin/ launcher instead of `chiton` from PATH
# (packaging/plugin/.mcp.json), and it leaves out everything that is not plugin
# content.
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
sums="$dist/checksums.txt"

[ -f "$sums" ] || {
	echo "build-plugin-dist: missing $sums; run 'make release VERSION=$tag' first" >&2
	exit 1
}
# The launchers download these exact names; refuse to pin a release that lacks one.
for asset in \
	"chiton-$tag-linux-amd64" \
	"chiton-$tag-linux-arm64" \
	"chiton-$tag-darwin-amd64" \
	"chiton-$tag-darwin-arm64" \
	"chiton-$tag-windows-amd64.exe"; do
	awk -v a="$asset" '$NF == a { found = 1 } END { exit !found }' "$sums" || {
		echo "build-plugin-dist: $sums has no entry for $asset" >&2
		exit 1
	}
done

rm -rf "$out"
mkdir -p "$out/.claude-plugin" "$out/bin" "$out/scripts" "$out/assets" "$out/release"

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

# No trailing newline: chiton.cmd reads this with `set /p`, which would keep a CR.
printf '%s' "$tag" >"$out/release/tag"
cp "$sums" "$out/release/checksums.txt"

chmod 0755 "$out/bin/chiton" "$out/scripts/install-chiton.sh"

echo "build-plugin-dist: assembled $out for $tag"
