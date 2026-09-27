package analyze

import (
	"context"
	"fmt"
	"sort"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// FindingView is the lean shape returned by list/explain operations.
type FindingView struct {
	ID             string            `json:"id"`
	RuleID         string            `json:"rule_id"`
	Category       string            `json:"category"`
	Severity       string            `json:"severity"`
	Message        string            `json:"message"`
	TargetURI      string            `json:"target_uri,omitempty"`
	Span           *model.SourceSpan `json:"span,omitempty"`
	Fix            string            `json:"fix_suggestion,omitempty"`
	Provenance     string            `json:"provenance,omitempty"`      // EXTRACTED | INFERRED | AMBIGUOUS
	RuleProvenance string            `json:"rule_provenance,omitempty"` // DETERMINISTIC | LLM
	Confidence     float32           `json:"confidence,omitempty"`      // LLM findings only
}

// ListOptions filters findings returned by List.
type ListOptions struct {
	Category    string   // empty = all
	MinSeverity Severity // empty = all
	Scope       string   // substring match against target_uri; empty = all
	Limit       int      // 0 = no cap
	// IncludeAmbiguous returns below-threshold AMBIGUOUS LLM findings, hidden by default so they
	// never masquerade as confirmed findings.
	IncludeAmbiguous bool
}

// List returns findings in the store, filtered by opts.
func List(ctx context.Context, gs store.GraphStore, opts ListOptions) ([]FindingView, error) {
	findings, err := store.NodesMatching(ctx, gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindFinding}})
	if err != nil {
		return nil, fmt.Errorf("load nodes: %w", err)
	}
	scopeLower := strings.ToLower(opts.Scope)
	minRank := -1
	if opts.MinSeverity != "" {
		minRank = SeverityRank(opts.MinSeverity)
	}

	var out []FindingView
	for _, n := range findings {
		if !opts.IncludeAmbiguous && n.Provenance == model.ProvAmbiguous {
			continue
		}
		fv := nodeToFindingView(n)
		if opts.Category != "" && !strings.EqualFold(fv.Category, opts.Category) {
			continue
		}
		if minRank >= 0 && SeverityRank(Severity(fv.Severity)) < minRank {
			continue
		}
		if opts.Scope != "" && !strings.Contains(strings.ToLower(fv.TargetURI), scopeLower) {
			continue
		}
		out = append(out, fv)
	}
	sortFindings(out)
	if opts.Limit > 0 && len(out) > opts.Limit {
		out = out[:opts.Limit]
	}
	return out, nil
}

// Explanation is a finding plus its rule metadata and any related findings on the same target.
type Explanation struct {
	Finding FindingView   `json:"finding"`
	Rule    *RuleView     `json:"rule,omitempty"`
	Related []FindingView `json:"related,omitempty"`
}

// RuleView is the lean shape for a rule node.
type RuleView struct {
	ID       string `json:"id"`
	Category string `json:"category"`
	Severity string `json:"severity"`
	Summary  string `json:"summary,omitempty"`
	Fix      string `json:"fix_hint,omitempty"`
}

// Explain finds the finding by ID (or label substring) and returns its
// neighborhood as an Explanation.
func Explain(ctx context.Context, gs store.GraphStore, idOrLabel string) (*Explanation, error) {
	node, err := gs.GetNode(ctx, idOrLabel)
	if err != nil {
		return nil, fmt.Errorf("get node: %w", err)
	}
	if node == nil {
		matches, err := gs.SearchNodes(ctx, idOrLabel, 20)
		if err != nil {
			return nil, fmt.Errorf("search: %w", err)
		}
		for _, m := range matches {
			if m.Kind == model.KindFinding {
				node = &m
				break
			}
		}
	}
	if node == nil || node.Kind != model.KindFinding {
		return nil, fmt.Errorf("finding %q not found", idOrLabel)
	}

	exp := &Explanation{Finding: nodeToFindingView(*node)}

	// 1-hop neighbourhood gives us the rule and any neighbours on the target.
	sub, err := gs.Neighbors(ctx, node.ID, 1, nil)
	if err != nil {
		return nil, fmt.Errorf("neighbors: %w", err)
	}
	for _, n := range sub.Nodes {
		if n.Kind == model.KindRule && n.Attrs["rule_id"] == exp.Finding.RuleID {
			rv := nodeToRuleView(n)
			exp.Rule = &rv
		}
	}

	// "Related" = other findings on the same target URI.
	if exp.Finding.TargetURI != "" {
		siblings, err := store.NodesMatching(ctx, gs, store.NodeFilter{
			Kinds: []model.NodeKind{model.KindFinding},
			Attrs: map[string]string{"target_uri": exp.Finding.TargetURI},
		})
		if err != nil {
			return nil, fmt.Errorf("load nodes: %w", err)
		}
		for _, n := range siblings {
			if n.ID == node.ID {
				continue
			}
			// Hide AMBIGUOUS siblings, matching List's default so they never
			// masquerade as confirmed findings in the related list.
			if n.Provenance == model.ProvAmbiguous {
				continue
			}
			exp.Related = append(exp.Related, nodeToFindingView(n))
		}
		sortFindings(exp.Related)
	}
	return exp, nil
}

func nodeToFindingView(n model.Node) FindingView {
	var conf float32
	switch v := n.Attrs["confidence_score"].(type) {
	case float64:
		conf = float32(v)
	case float32:
		conf = v
	}
	return FindingView{
		ID:             n.ID,
		RuleID:         model.AttrString(n.Attrs, "rule_id"),
		Category:       model.AttrString(n.Attrs, "category"),
		Severity:       model.AttrString(n.Attrs, "severity"),
		Message:        model.AttrString(n.Attrs, "message"),
		TargetURI:      model.AttrString(n.Attrs, "target_uri"),
		Span:           n.Span,
		Fix:            model.AttrString(n.Attrs, "fix_suggestion"),
		Provenance:     string(n.Provenance),
		RuleProvenance: model.AttrString(n.Attrs, "rule_provenance"),
		Confidence:     conf,
	}
}

func nodeToRuleView(n model.Node) RuleView {
	return RuleView{
		ID:       model.AttrString(n.Attrs, "rule_id"),
		Category: model.AttrString(n.Attrs, "category"),
		Severity: model.AttrString(n.Attrs, "severity"),
		Summary:  model.AttrString(n.Attrs, "summary"),
		Fix:      model.AttrString(n.Attrs, "fix_hint"),
	}
}

// sortFindings orders findings by severity descending, then rule, then target.
func sortFindings(fs []FindingView) {
	sort.Slice(fs, func(i, j int) bool {
		ri, rj := SeverityRank(Severity(fs[i].Severity)), SeverityRank(Severity(fs[j].Severity))
		if ri != rj {
			return ri > rj
		}
		if fs[i].RuleID != fs[j].RuleID {
			return fs[i].RuleID < fs[j].RuleID
		}
		return fs[i].TargetURI < fs[j].TargetURI
	})
}
