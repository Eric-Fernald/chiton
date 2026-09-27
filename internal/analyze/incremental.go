package analyze

import (
	"context"
	"fmt"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// AnalyzeFile is the single-file counterpart of Analyze for live sync: it runs only FileRules
// (GraphScope rules need the whole corpus), and an unsupported or unreadable path is a no-op.
func (a *Analyzer) AnalyzeFile(ctx context.Context, path string) (*Result, error) {
	start := a.now()
	// A read or parse failure (e.g. removed mid-flight) leaves existing findings untouched.
	ft, _, ok := loadFileTarget(ctx, path)
	if !ok {
		return &Result{}, nil
	}
	return a.analyzeTarget(ctx, ft, start)
}

// AnalyzeExtracted is AnalyzeFile for a caller that already read path (content) and extracted it
// (nodes and edges as ExtractSource returned them), so the file isn't read and parsed twice.
func (a *Analyzer) AnalyzeExtracted(ctx context.Context, path string, content []byte, nodes []model.Node, edges []model.Edge) (*Result, error) {
	start := a.now()
	return a.analyzeTarget(ctx, newFileTarget(path, content, nodes, edges), start)
}

// analyzeTarget runs the FileRules applicable to ft and replaces ft's findings from those rules.
func (a *Analyzer) analyzeTarget(ctx context.Context, ft FileTarget, start time.Time) (*Result, error) {
	ext := FileExt(ft.Path)
	rs := a.fileRuleSet()
	var findings []Finding
	var usedRules []Rule
	ruleScope := map[string]bool{}
	for i, fr := range rs.rules {
		if !ruleApplies(rs.langs[i], ext) {
			continue
		}
		usedRules = append(usedRules, fr)
		ruleScope[rs.ids[i]] = true
		findings = append(findings, fr.MatchFile(ctx, ft)...)
	}

	ids, err := a.persist(ctx, usedRules, findings)
	if err != nil {
		return nil, err
	}
	scanned := map[string]bool{}
	if ft.URI != "" {
		scanned[ft.URI] = true
	}
	// Scope the prune to the rules just run: this file's GraphScope findings must survive the edit.
	if err := a.pruneResolved(ctx, scanned, ids, ruleScope); err != nil {
		return nil, err
	}

	return &Result{
		FilesScanned: 1,
		RulesRun:     len(usedRules),
		Findings:     findings,
		Duration:     a.now().Sub(start),
	}, nil
}

// fileRuleSet is the registry's FileRules in ID order, with each rule's Meta ID and languages.
type fileRuleSet struct {
	rules []FileRule
	ids   []string
	langs [][]string
}

// fileRuleSet builds the FileRule set on first use so AnalyzeFile needn't re-sort and re-query the registry.
func (a *Analyzer) fileRuleSet() *fileRuleSet {
	a.fileRulesOnce.Do(func() {
		for _, r := range a.registry.All() {
			if fr, ok := r.(FileRule); ok {
				m := fr.Meta()
				a.fileRules.rules = append(a.fileRules.rules, fr)
				a.fileRules.ids = append(a.fileRules.ids, m.ID)
				a.fileRules.langs = append(a.fileRules.langs, m.Languages)
			}
		}
	})
	return &a.fileRules
}

// PruneFileURI removes every finding (deterministic and LLM) whose target_uri is uri, for a deleted
// file. Findings' own source_uri is "finding://<id>", so DeleteBySourceURI can't reach them.
func (a *Analyzer) PruneFileURI(ctx context.Context, uri string) (int, error) {
	if uri == "" {
		return 0, nil
	}
	stale, err := store.NodesMatching(ctx, a.store, store.NodeFilter{
		Kinds: []model.NodeKind{model.KindFinding},
		Attrs: map[string]string{"target_uri": uri},
	})
	if err != nil {
		return 0, fmt.Errorf("load nodes for prune: %w", err)
	}
	if err := deleteFindingNodes(ctx, a.store, stale); err != nil {
		return 0, err
	}
	return len(stale), nil
}
