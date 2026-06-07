# chiton — build, test, and release.
#
# Pure-Go by default (no CGo): `go build ./...` works on Windows/macOS/Linux
# without a C toolchain. The optional in-process whisper.cpp backend is behind
# the `whisper_cgo` build tag and is NOT part of any target here.

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

.PHONY: check
check: vet test ## Vet + test (the pre-merge gate)

.PHONY: release
release: ## Cross-compile release binaries into dist/
	@mkdir -p $(DIST)
	@for platform in $(PLATFORMS); do \
		os=$${platform%/*}; arch=$${platform#*/}; \
		ext=""; [ "$$os" = "windows" ] && ext=".exe"; \
		out=$(DIST)/$(BINARY)-$(VERSION)-$$os-$$arch$$ext; \
		echo "build $$out"; \
		CGO_ENABLED=0 GOOS=$$os GOARCH=$$arch \
			go build -ldflags "$(LDFLAGS)" -o $$out $(PKG) || exit 1; \
	done

.PHONY: clean
clean: ## Remove build artifacts
	rm -rf $(BINARY) $(BINARY).exe $(DIST) chiton-out

.PHONY: help
help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'
