// Package hooks installs and runs chiton's git pre-commit and pre-push hooks.
//
// The hooks.yaml file is committed to the repo so the whole team agrees on
// what severity triggers a block. The Go-side hook runner reads that file,
// invokes the analyzer, and exits non-zero when any finding crosses the
// configured threshold.
package hooks

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// Stage names match git's hook names (pre-commit, pre-push) so the on-disk
// keys in hooks.yaml are immediately recognisable.
type Stage string

const (
	StagePreCommit Stage = "pre-commit"
	StagePrePush   Stage = "pre-push"
	// Sync stages keep the graph current after history changes (Phase 3). They
	// run `chiton sync git` rather than the gating `chiton hooks run`.
	StagePostCommit   Stage = "post-commit"
	StagePostCheckout Stage = "post-checkout"
)

// Stages lists the gating hook stages chiton supports (pre-commit, pre-push).
// These are the stages `chiton hooks run` understands.
func Stages() []Stage { return []Stage{StagePreCommit, StagePrePush} }

// SyncStages lists the live-sync hook stages (post-commit, post-checkout) that
// re-index changed files into the graph after history moves.
func SyncStages() []Stage { return []Stage{StagePostCommit, StagePostCheckout} }

// AllStages lists every hook stage chiton can install.
func AllStages() []Stage {
	return []Stage{StagePreCommit, StagePrePush, StagePostCommit, StagePostCheckout}
}

// IsSyncStage reports whether s is a live-sync stage (vs a gating stage).
func IsSyncStage(s Stage) bool {
	return s == StagePostCommit || s == StagePostCheckout
}

// StageConfig is one stage's policy.
type StageConfig struct {
	BlockOn        []string `yaml:"block-on"`
	LLM            bool     `yaml:"llm"`
	DiffBaseline   string   `yaml:"diff-baseline"`
	TimeoutSeconds int      `yaml:"timeout-seconds"`
}

// Config is the parsed shape of .chiton/hooks.yaml.
type Config struct {
	PreCommit StageConfig `yaml:"pre-commit"`
	PrePush   StageConfig `yaml:"pre-push"`
}

// Default returns the same defaults chiton init writes into a fresh repo.
func Default() Config {
	return Config{
		PreCommit: StageConfig{
			BlockOn:        []string{"critical", "high"},
			LLM:            false,
			DiffBaseline:   "HEAD~1",
			TimeoutSeconds: 30,
		},
		PrePush: StageConfig{
			BlockOn:        []string{"critical"},
			LLM:            true,
			DiffBaseline:   "HEAD~1",
			TimeoutSeconds: 60,
		},
	}
}

// Load reads hooks.yaml from path. Missing fields fall back to Default().
func Load(path string) (Config, error) {
	defaults := Default()
	b, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return defaults, nil
	}
	if err != nil {
		return defaults, fmt.Errorf("read %s: %w", path, err)
	}
	var cfg Config
	if err := yaml.Unmarshal(b, &cfg); err != nil {
		return defaults, fmt.Errorf("parse %s: %w", path, err)
	}
	if len(cfg.PreCommit.BlockOn) == 0 {
		cfg.PreCommit.BlockOn = defaults.PreCommit.BlockOn
	}
	if cfg.PreCommit.TimeoutSeconds == 0 {
		cfg.PreCommit.TimeoutSeconds = defaults.PreCommit.TimeoutSeconds
	}
	if cfg.PreCommit.DiffBaseline == "" {
		cfg.PreCommit.DiffBaseline = defaults.PreCommit.DiffBaseline
	}
	if len(cfg.PrePush.BlockOn) == 0 {
		cfg.PrePush.BlockOn = defaults.PrePush.BlockOn
	}
	if cfg.PrePush.TimeoutSeconds == 0 {
		cfg.PrePush.TimeoutSeconds = defaults.PrePush.TimeoutSeconds
	}
	if cfg.PrePush.DiffBaseline == "" {
		cfg.PrePush.DiffBaseline = defaults.PrePush.DiffBaseline
	}
	return cfg, nil
}

// For returns the StageConfig for the given stage, falling back to the
// defaults' stage if name is unknown.
func (c Config) For(stage Stage) StageConfig {
	switch stage {
	case StagePreCommit:
		return c.PreCommit
	case StagePrePush:
		return c.PrePush
	}
	return StageConfig{}
}

// MinBlockingSeverity returns the lowest-ranked severity in BlockOn — that's
// the threshold the runner uses to decide which findings count.
// Returns "" when BlockOn is empty (no severity blocks).
func (s StageConfig) MinBlockingSeverity() analyze.Severity {
	if len(s.BlockOn) == 0 {
		return ""
	}
	min := analyze.Severity("")
	minRank := 1 << 30
	for _, raw := range s.BlockOn {
		sev := analyze.Severity(raw)
		rank := analyze.SeverityRank(sev)
		if rank < 0 {
			continue
		}
		if rank < minRank {
			minRank = rank
			min = sev
		}
	}
	return min
}
