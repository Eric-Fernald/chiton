package analyze

import (
	"context"
	"fmt"
	"time"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// AnalyzeFile is the single-file version of Analyze for live sync. It runs only FileRules, because
// GraphScope rules need the whole corpus. An unsupported or unreadable path is a no-op.
func (a *Analyzer) AnalyzeFile(ctx context.Context, path string) (*Result, error) {
	start := a.now()
	// A read or parse failure, such as a file deleted during the read, leaves existing findings untouched.
	ft, _, ok := loadFileTarget(ctx, path)
	if !ok {
		return &Result{}, nil
	}
	return a.analyzeTarget(ctx, ft, start, nil)
}

// AnalyzeExtracted is AnalyzeFile for a caller that already read path (content) and extracted it
// (nodes and edges as ExtractSource returned them), so the file isn't read and parsed twice.
func (a *Analyzer) AnalyzeExtracted(ctx context.Context, path string, content []byte, nodes []model.Node, edges []model.Edge) (*Result, error) {
	return a.ReanalyzeExtracted(ctx, path, content, nodes, edges, nil)
}

// PriorFindings holds a file's deterministic findings as ReadPriorFindings saw them.
type PriorFindings struct{ nodes []model.Node }

// ReadPriorFindings returns uri's deterministic findings via AFFECTS edges into its file node and CONTAINS'd
// symbols, a file-sized read. Call it before replacing uri's nodes, since deleting them drops those edges.
func (a *Analyzer) ReadPriorFindings(ctx context.Context, uri string) (*PriorFindings, error) {
	kinds := []model.EdgeKind{model.EdgeContains, model.EdgeAffects}
	sub, err := a.store.Neighbors(ctx, model.NewNodeID(uri, uri), 2, kinds)
	if err != nil {
		return nil, fmt.Errorf("read prior findings for %s: %w", uri, err)
	}
	p := &PriorFindings{}
	for _, n := range sub.Nodes {
		if n.Kind == model.KindFinding && model.AttrString(n.Attrs, "rule_provenance") == string(ProvenanceDeterministic) {
			p.nodes = append(p.nodes, n)
		}
	}
	return p, nil
}

// IDs returns the prior findings' node IDs; a nil p has none.
func (p *PriorFindings) IDs() []string {
	if p == nil {
		return nil
	}
	ids := make([]string, len(p.nodes))
	for i, n := range p.nodes {
		ids[i] = n.ID
	}
	return ids
}

// ReanalyzeExtracted is AnalyzeExtracted for live sync: it prunes from prior, read by ReadPriorFindings
// before the caller replaced path's nodes, instead of querying the store. A nil prior queries.
func (a *Analyzer) ReanalyzeExtracted(ctx context.Context, path string, content []byte, nodes []model.Node, edges []model.Edge, prior *PriorFindings) (*Result, error) {
	start := a.now()
	return a.analyzeTarget(ctx, newFileTarget(path, content, nodes, edges), start, prior)
}

// analyzeTarget runs the FileRules applicable to ft and replaces ft's findings from those rules.
func (a *Analyzer) analyzeTarget(ctx context.Context, ft FileTarget, start time.Time, prior *PriorFindings) (*Result, error) {
	res := newFileResult()
	res.prior = prior
	a.matchFile(ctx, ft, res)
	if err := a.Record(ctx, res); err != nil {
		return nil, err
	}
	res.Duration = a.now().Sub(start)
	return res, nil
}

// newFileResult starts a Result for FileRule-only passes. Its prune covers only the rules that ran, so a
// file's GraphScope findings survive a single-file re-analyze.
func newFileResult() *Result {
	return &Result{scanned: map[string]bool{}, ruleScope: map[string]bool{}}
}

// matchFile runs the FileRules applicable to ft and adds ft, the rules, and their findings to res.
func (a *Analyzer) matchFile(ctx context.Context, ft FileTarget, res *Result) {
	ext := FileExt(ft.Path)
	rs := a.fileRuleSet()
	for i, fr := range rs.rules {
		if !ruleApplies(rs.langs[i], ext) {
			continue
		}
		if !res.ruleScope[rs.ids[i]] {
			res.ruleScope[rs.ids[i]] = true
			res.rules = append(res.rules, fr)
		}
		res.Findings = append(res.Findings, fr.MatchFile(ctx, ft)...)
	}
	res.RulesRun = len(res.rules)
	res.FilesScanned++
	if ft.URI != "" {
		res.scanned[ft.URI] = true
	}
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
