# chiton — Claude Code plugin

This branch is generated. Each chiton release adds one commit, assembled from
the release tag of <https://github.com/Eric-Fernald/chiton>; do not edit it by
hand.

chiton is an on-device knowledge graph for your codebase. The plugin registers
its MCP server and the `/chiton:graph`, `/chiton:subagent-tasks` and
`/chiton:setup` skills, and bundles the chiton binary for Linux and macOS
(amd64, arm64) and Windows (amd64), so nothing else needs installing.

```sh
claude plugin marketplace add Eric-Fernald/chiton
claude plugin install chiton@chiton-marketplace
```

Then run `/chiton:setup` in a workspace to build its graph.

- Documentation: <https://github.com/Eric-Fernald/chiton/blob/main/docs/INSTALL.md>
- Privacy policy: <https://github.com/Eric-Fernald/chiton/blob/main/docs/PRIVACY.md>
- Issues: <https://github.com/Eric-Fernald/chiton/issues>

MIT licensed; see [LICENSE](LICENSE).
