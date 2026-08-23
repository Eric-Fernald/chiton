## What this changes

<!-- One paragraph. What behaviour is different after this merges? -->

## Why

<!-- The problem being solved. Link an issue if there is one: Fixes #123 -->

## How it was verified

<!--
Paste the commands you actually ran, not the ones you meant to run.
  make check                # gofmt-check + vet + test — the pre-merge gate
  make check-msgraph        # only if you touched the Microsoft Graph path
  go test -race ./...       # for anything concurrent (watchers, jobs, stores)
-->

```
```

## Checklist

- [ ] `make check` passes locally.
- [ ] New behaviour has a test, or the PR explains why it cannot.
- [ ] Touched the Teams/Outlook code behind `//go:build msgraph`? `make check-msgraph` passes. Release artifacts are built without that tag, so the stub path must still compile and give a useful error.
- [ ] Touched `bin/chiton` or `scripts/install-chiton.sh`? They are still mode `100755` in the index (`git ls-files -s`); the plugin's PATH shim breaks without the executable bit.
- [ ] Docs updated for any new command, MCP tool, rule, config key, or environment variable — `README.md`, `docs/`, and, when the assistant-facing surface changed, `skills/*/SKILL.md` plus the `RulesPrompt` block in `internal/install/manifests`. (`CLAUDE.md` is gitignored and cannot be reviewed in a diff; edit it locally if you like, but the shipped equivalents are the ones that matter.)
- [ ] No secrets, tokens, customer data, or transcript content in the diff, tests, or fixtures.
- [ ] No changes to `LICENSE`.

## Notes for the reviewer

<!-- Anything non-obvious: a trade-off you made, a case you deliberately left out. -->
