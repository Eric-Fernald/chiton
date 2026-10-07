# chiton — Claude Code plugin

This branch is generated. Each chiton release adds one commit, assembled from
the release tag of <https://github.com/Eric-Fernald/chiton>; do not edit it by
hand.

chiton is an on-device knowledge graph for your codebase. The plugin registers
its MCP server and the `/chiton:graph`, `/chiton:subagent-tasks` and
`/chiton:setup` skills. It carries no binary. On first use its launcher
downloads the chiton binary for this machine (Linux and macOS on amd64 or
arm64, Windows on amd64 or arm64, about 25 MB) from the release named in
`release/tag`, verifies it against `release/checksums.txt`, and caches it in
`~/.chiton/versions/<tag>/`. Nothing else needs installing.

The first start waits for that download. If it outlasts Claude Code's MCP
startup timeout, reconnect with `/mcp` or `/reload-plugins` once it finishes.
Uninstalling the plugin leaves `~/.chiton/versions/` behind; delete it to
reclaim the space.

```sh
claude plugin marketplace add Eric-Fernald/chiton
claude plugin install chiton@chiton-marketplace
```

Then run `/chiton:setup` in a workspace to build its graph.

- Documentation: <https://github.com/Eric-Fernald/chiton/blob/main/docs/INSTALL.md>
- Privacy policy: <https://github.com/Eric-Fernald/chiton/blob/main/docs/PRIVACY.md>
- Issues: <https://github.com/Eric-Fernald/chiton/issues>

Licensed under the PolyForm Strict License 1.0.0 with additional permissions:
use it for any purpose and install it by any documented route, but do not modify
or redistribute it. See [LICENSE](LICENSE).
