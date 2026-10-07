# Security policy

chiton is licensed under the PolyForm Strict License 1.0.0 with additional
permissions (see [LICENSE](LICENSE)). This file covers one thing: **how to report a vulnerability**. The threat model,
encryption-at-rest design, PII handling, and full data-handling posture live in
[docs/SECURITY.md](docs/SECURITY.md). Read that first if you want to know what
chiton defends against and what it explicitly does not.

## Supported versions

There are no tagged releases yet and no maintained release branches. Only the
current tip of `main` is supported.

| Version | Supported |
|---|---|
| `main` (latest commit) | Yes |
| Any older commit or branch | No. Reproduce against `main` before reporting |

## Reporting a vulnerability

**Report privately. Do not open a public issue**, and do not describe the
problem in a pull request, commit message, or discussion thread. Those are
visible to everyone with repository access.

- **Preferred:** open a private GitHub Security Advisory on this repository
  (*Security → Advisories → Report a vulnerability*). This creates a private
  thread visible only to you and the maintainer.
- **Alternative:** contact the maintainer directly if you cannot reach the
  Advisories tab.

## What to include

The more of this you can supply, the faster a fix lands:

- **Version and commit.** The output of `chiton version`, or the commit SHA you
  built from, plus whether the binary was built with `-tags msgraph` or
  `-tags whisper_cgo`.
- **Environment.** OS and architecture, and how chiton was invoked (CLI,
  `chiton serve --stdio`, `--http`, a webhook receiver, an assistant plugin).
- **Description.** What the issue is and what an attacker gains from it.
- **Reproduction.** Minimal steps, a proof of concept, or a failing test. If
  the repro needs real data, describe the shape of the data rather than
  attaching secrets, source code, or PII.
- **Impact assessment.** Your view of severity, if you have one.

Before filing, skim the *Out of scope* section of
[docs/SECURITY.md](docs/SECURITY.md) so you do not spend effort on a risk that
is already documented and accepted.

## What to expect

This is a single-maintainer project, so there is no formal SLA. The aim is to
acknowledge a report within a few business days, confirm or dispute it after
triage, and agree a fix and disclosure timeline with you from there. If you have
not heard back in a reasonable window, follow up on the same private thread.

Please give the maintainer a chance to ship a fix before discussing the issue
anywhere public.
