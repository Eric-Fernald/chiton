# chiton — build, test, and release.
#
# Pure-Go by default (no CGo): `go build ./...` works on Windows/macOS/Linux
# without a C toolchain.
#
# Two optional build tags exist, and neither is part of the default targets:
#
#   whisper_cgo  in-process whisper.cpp transcription (needs a C toolchain).
#                Without it, `chiton meeting transcribe` shells out to a
#                whisper CLI on PATH.
#   msgraph      Microsoft Graph support: `chiton teams *` and
#                `chiton email outlook`. Costs ~117 MB of binary — the Graph
#                SDK links request builders for every Graph resource — so the
#                default build and every release artifact omit it. Use
#                `make build-msgraph` when you need those commands.

BINARY      := chiton
PKG         := ./cmd/chiton
DIST        := dist

# Version metadata injected via -ldflags into cmd/chiton's build vars.
VERSION     ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo dev)
COMMIT      ?= $(shell git rev-parse --short HEAD 2>/dev/null || echo none)
DATE        ?= $(shell date -u +%Y-%m-%dT%H:%M:%SZ)
LDFLAGS     := -s -w \
	-X main.version=$(VERSION) \
	-X main.commit=$(COMMIT) \
	-X main.date=$(DATE)

# Cross-compile matrix for `make release`.
PLATFORMS   := linux/amd64 linux/arm64 darwin/amd64 darwin/arm64 windows/amd64

.DEFAULT_GOAL := build

.PHONY: build
build: ## Build the chiton binary for the host platform
	go build -ldflags "$(LDFLAGS)" -o $(BINARY) $(PKG)

.PHONY: build-msgraph
build-msgraph: ## Build with Microsoft Graph support (Teams + Outlook); ~117 MB larger
	go build -tags msgraph -ldflags "$(LDFLAGS)" -o $(BINARY) $(PKG)

.PHONY: install
install: ## Install chiton onto the Go bin path
	go install -ldflags "$(LDFLAGS)" $(PKG)

.PHONY: test
test: ## Run the full test suite
	go test ./...

.PHONY: race
race: ## Run tests with the race detector
	go test -race ./...

.PHONY: vet
vet: ## Run go vet
	go vet ./...

.PHONY: lint
lint: ## Run golangci-lint if installed
	@command -v golangci-lint >/dev/null 2>&1 \
		&& golangci-lint run \
		|| echo "golangci-lint not installed; skipping (https://golangci-lint.run)"

.PHONY: fmt
fmt: ## Rewrite sources with gofmt
	gofmt -w .

.PHONY: fmt-check
fmt-check: ## Fail if anything is not gofmt-clean
	@out=$$(gofmt -l .); \
	if [ -n "$$out" ]; then echo "not gofmt-clean:"; echo "$$out"; exit 1; fi

.PHONY: check
check: fmt-check vet test ## gofmt + vet + test (the pre-merge gate)

.PHONY: check-msgraph
check-msgraph: ## Keep the tag-gated Microsoft Graph path compiling and passing
	go vet -tags msgraph ./...
	go test -tags msgraph ./internal/ingest/email/... ./internal/ingest/teams/...

# Release artifacts are deliberately built WITHOUT -tags msgraph: it would add
# ~117 MB per platform for two API calls. Users who need Teams/Outlook build
# with `make build-msgraph`.
.PHONY: release
release: ## Cross-compile release binaries + checksums into dist/
	@# Start clean: a leftover artifact from a previous version would be hashed
	@# into checksums.txt and swept into the next release by the workflow's glob.
	@rm -rf $(DIST)
	@mkdir -p $(DIST)
	@for platform in $(PLATFORMS); do \
		os=$${platform%/*}; arch=$${platform#*/}; \
		ext=""; [ "$$os" = "windows" ] && ext=".exe"; \
		out=$(DIST)/$(BINARY)-$(VERSION)-$$os-$$arch$$ext; \
		echo "build $$out"; \
		CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch \
			go build -ldflags "$(LDFLAGS)" -o $$out $(PKG) || exit 1; \
	done
	@$(MAKE) --no-print-directory checksums

# scripts/install-chiton.{sh,ps1} refuse to install a binary that is not listed
# here, so checksums.txt must ship with every release.
.PHONY: checksums
checksums: ## Write dist/checksums.txt over the built artifacts
	@cd $(DIST) && rm -f checksums.txt && \
	if command -v sha256sum >/dev/null 2>&1; then \
		sha256sum $(BINARY)-* > checksums.txt; \
	else \
		shasum -a 256 $(BINARY)-* > checksums.txt; \
	fi && \
	echo "wrote $(DIST)/checksums.txt"

# The published plugin bundles the release binaries; the release workflow
# commits dist/plugin/ as the plugin-dist branch. VERSION must be a v-tag.
.PHONY: plugin-dist
plugin-dist: release ## Assemble the published plugin, binaries included, into dist/plugin/
	sh scripts/build-plugin-dist.sh $(VERSION)

# Validates a fresh clone rather than the working tree, which may hold
# untracked files (a local CLAUDE.md, built binaries) that the published
# plugin never contains.
.PHONY: plugin-validate
plugin-validate: ## Run `claude plugin validate --strict` on a clean clone of HEAD
	@tmp=$$(mktemp -d) && trap 'rm -rf "$$tmp"' EXIT && \
	git clone -q --no-hardlinks . "$$tmp/chiton" && \
	claude plugin validate "$$tmp/chiton" --strict

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf $(BINARY) $(BINARY).exe $(DIST) chiton-out

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
