// Package hooks installs chiton's git hooks and runs the gating ones, which block when a finding crosses
// the severity threshold in the committed .chiton/hooks.yaml.
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
	// Sync stages keep the graph current after history changes. They
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
	cfg.PreCommit.fillDefaults(defaults.PreCommit)
	cfg.PrePush.fillDefaults(defaults.PrePush)
	return cfg, nil
}

// fillDefaults replaces unset BlockOn, TimeoutSeconds, and DiffBaseline with d's values. It leaves LLM
// unchanged because an omitted bool is indistinguishable from false.
func (s *StageConfig) fillDefaults(d StageConfig) {
	if len(s.BlockOn) == 0 {
		s.BlockOn = d.BlockOn
	}
	if s.TimeoutSeconds == 0 {
		s.TimeoutSeconds = d.TimeoutSeconds
	}
	if s.DiffBaseline == "" {
		s.DiffBaseline = d.DiffBaseline
	}
}

// For returns the StageConfig for the given stage, or the zero StageConfig if the stage is unknown.
func (c Config) For(stage Stage) StageConfig {
	switch stage {
	case StagePreCommit:
		return c.PreCommit
	case StagePrePush:
		return c.PrePush
	}
	return StageConfig{}
}

// MinBlockingSeverity returns the lowest-ranked known severity in BlockOn, the runner's blocking threshold,
// or "" when BlockOn has none.
func (s StageConfig) MinBlockingSeverity() analyze.Severity {
	var lowest analyze.Severity
	lowestRank := -1
	for _, raw := range s.BlockOn {
		sev := analyze.Severity(raw)
		rank := analyze.SeverityRank(sev)
		if rank >= 0 && (lowestRank < 0 || rank < lowestRank) {
			lowest, lowestRank = sev, rank
		}
	}
	return lowest
}
