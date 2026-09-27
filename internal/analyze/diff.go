package analyze

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Baseline is the finding set at a commit (.chiton/baselines/<commit>.json); `--diff` compares
// against the latest one so a commit is judged only on the findings it adds.
type Baseline struct {
	Commit      string        `json:"commit"`
	GeneratedAt time.Time     `json:"generated_at"`
	Findings    []FindingView `json:"findings"`
}

// FindingChange pairs a finding's before/after views when it persists across
// two scans but its message or severity changed.
type FindingChange struct {
	ID     string      `json:"id"`
	Before FindingView `json:"before"`
	After  FindingView `json:"after"`
}

// DiffResult is the new/resolved/changed breakdown between two finding sets.
type DiffResult struct {
	Baseline string          `json:"baseline_commit,omitempty"`
	Target   string          `json:"target_commit,omitempty"`
	New      []FindingView   `json:"new"`
	Resolved []FindingView   `json:"resolved"`
	Changed  []FindingChange `json:"changed"`
}

// Diff compares a baseline finding set to the current one by stable finding ID; Changed means
// the same ID with a different message or severity.
func Diff(baseline, current []FindingView) DiffResult {
	base := indexFindings(baseline)
	cur := indexFindings(current)

	var res DiffResult
	for id, c := range cur {
		b, ok := base[id]
		if !ok {
			res.New = append(res.New, c)
			continue
		}
		if b.Message != c.Message || b.Severity != c.Severity {
			res.Changed = append(res.Changed, FindingChange{ID: id, Before: b, After: c})
		}
	}
	for id, b := range base {
		if _, ok := cur[id]; !ok {
			res.Resolved = append(res.Resolved, b)
		}
	}
	sortFindings(res.New)
	sortFindings(res.Resolved)
	slices.SortFunc(res.Changed, func(a, b FindingChange) int { return strings.Compare(a.ID, b.ID) })
	return res
}

// Empty reports whether the diff found no new, resolved, or changed findings.
func (d DiffResult) Empty() bool {
	return len(d.New) == 0 && len(d.Resolved) == 0 && len(d.Changed) == 0
}

func indexFindings(fs []FindingView) map[string]FindingView {
	m := make(map[string]FindingView, len(fs))
	for _, f := range fs {
		m[f.ID] = f
	}
	return m
}

// BaselineDir returns the baselines directory under a .chiton directory.
func BaselineDir(chitonDir string) string { return filepath.Join(chitonDir, "baselines") }

// SaveBaseline writes findings as the baseline for commit and returns the path.
func SaveBaseline(dir, commit string, findings []FindingView, now time.Time) (string, error) {
	if commit == "" {
		commit = "working"
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return "", err
	}
	b := Baseline{Commit: commit, GeneratedAt: now, Findings: findings}
	data, err := json.MarshalIndent(b, "", "  ")
	if err != nil {
		return "", err
	}
	path := filepath.Join(dir, baselineFile(commit))
	if err := fsutil.WriteFileAtomic(path, data, 0o644); err != nil {
		return "", err
	}
	return path, nil
}

// LoadBaseline reads the baseline for commit. commit "latest" (or "") returns
// the most recently generated baseline. Returns (nil, nil) when none exists.
func LoadBaseline(dir, commit string) (*Baseline, error) {
	if commit == "" || commit == "latest" {
		return LatestBaseline(dir)
	}
	b, err := readBaseline(filepath.Join(dir, baselineFile(commit)))
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	return b, err
}

// LatestBaseline returns the baseline with the newest GeneratedAt, or
// (nil, nil) when the directory has none.
func LatestBaseline(dir string) (*Baseline, error) {
	entries, err := os.ReadDir(dir)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var latest *Baseline
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), ".json") {
			continue
		}
		b, err := readBaseline(filepath.Join(dir, e.Name()))
		if err != nil {
			continue // skip unreadable/corrupt baselines
		}
		// Newest GeneratedAt wins; ties break by commit so the result never depends on dir order.
		if latest == nil || b.GeneratedAt.After(latest.GeneratedAt) ||
			(b.GeneratedAt.Equal(latest.GeneratedAt) && b.Commit > latest.Commit) {
			latest = b
		}
	}
	return latest, nil
}

func readBaseline(path string) (*Baseline, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var b Baseline
	if err := json.Unmarshal(data, &b); err != nil {
		return nil, fmt.Errorf("parse baseline %s: %w", path, err)
	}
	return &b, nil
}

// baselineFile maps a commit ref to a safe filename. Git SHAs are already
// filesystem-safe; any other ref (e.g. "working") is sanitised.
func baselineFile(commit string) string {
	safe := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			return r
		default:
			return '-'
		}
	}, commit)
	return safe + ".json"
}

// GitCommit resolves ref to a full commit SHA in repoDir, or "" when git is unavailable or the
// ref doesn't resolve, so callers can fall back to a "working" baseline.
func GitCommit(repoDir, ref string) string {
	out, err := exec.Command("git", "-C", repoDir, "rev-parse", ref).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// CurrentFindings is a convenience that lists confirmed (non-AMBIGUOUS)
// findings from the store for diffing.
func CurrentFindings(ctx context.Context, gs store.GraphStore) ([]FindingView, error) {
	return List(ctx, gs, ListOptions{})
}
