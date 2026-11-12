package verify

import (
	"context"
	"fmt"
	"maps"
	"os"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// decisionClaims builds one task per node decision d governs, checked against that node's current source. A
// governed node whose source can't be read reliably is left out.
func (s *Service) decisionClaims(ctx context.Context, d model.Node) ([]claim, error) {
	sub, err := s.store.Neighbors(ctx, d.ID, 1, []model.EdgeKind{model.EdgeDecidedIn})
	if err != nil {
		return nil, fmt.Errorf("load links of decision %s: %w", d.ID, err)
	}
	byID := make(map[string]model.Node, len(sub.Nodes))
	for _, n := range sub.Nodes {
		byID[n.ID] = n
	}
	text := "Decision: " + model.AttrString(d.Attrs, "summary")
	if rationale := model.AttrString(d.Attrs, "rationale"); rationale != "" {
		text += "\nRationale: " + rationale
	}
	var claims []claim
	for _, e := range sub.Edges {
		sym, ok := byID[e.Dst]
		// A decision's DECIDED_IN edges also reach the meeting it was made in.
		if e.Src != d.ID || !ok || sym.Kind == model.KindMeeting {
			continue
		}
		source, ok := s.sourceText(ctx, sym)
		if !ok {
			continue
		}
		evidence := []Evidence{{NodeID: sym.ID, Label: sym.Label, SourceURI: sym.SourceURI, Text: source}}
		claims = append(claims, claim{node: d, symbol: sym, link: e, spec: TaskSpec{
			ID:           d.ID + ":" + sym.ID,
			Kind:         KindDecisionDrift,
			Label:        d.Label + " -> " + sym.Label,
			Claim:        text,
			Evidence:     evidence,
			Fingerprint:  fingerprint(KindDecisionDrift, text, evidence),
			Schema:       decisionSchema,
			Instructions: decisionInstructions,
		}})
	}
	return claims, nil
}

// sourceText returns what n says now: its stored text, or for a code symbol the lines of its span read from
// disk. ok is false when the file is outside the project root, unreadable, or changed since it was indexed.
func (s *Service) sourceText(ctx context.Context, n model.Node) (text string, ok bool) {
	if text := strings.TrimSpace(model.AttrString(n.Attrs, "text")); text != "" {
		return clip(text), true
	}
	path, isFile := model.PathFromFileURI(n.SourceURI)
	if !isFile || n.Span == nil || n.Span.StartLine <= 0 {
		return "", false
	}
	// Task specs carry file contents, so nothing outside the project root (symlinks resolved) is read.
	abs, err := fsutil.ResolveWithin(s.root, path)
	if err != nil {
		return "", false
	}
	data, err := os.ReadFile(abs)
	if err != nil || !s.indexedAs(ctx, n.SourceURI, data) {
		return "", false
	}
	lines := strings.Split(strings.ReplaceAll(string(data), "\r\n", "\n"), "\n")
	start, end := n.Span.StartLine, min(max(n.Span.EndLine, n.Span.StartLine), len(lines))
	if start > end {
		return "", false
	}
	text = strings.Join(lines[start-1:end], "\n")
	return clip(text), strings.TrimSpace(text) != ""
}

// indexedAs reports whether data is the content the graph's spans for uri were taken from. After an edit the
// graph has not caught up with, the lines at a span may belong to something else. A store with no file node or
// no recorded hash can't tell, and passes.
func (s *Service) indexedAs(ctx context.Context, uri string, data []byte) bool {
	file, err := s.store.GetNode(ctx, model.NewNodeID(uri, uri))
	if err != nil {
		return false
	}
	return file == nil || file.ContentSHA256 == "" || file.ContentSHA256 == model.ContentSHA(data)
}

// applyToDecision writes a verdict onto the decision's link to one symbol and returns what it did. Drift keeps
// the link, since the decision is still about that symbol, and adds a CONTRADICTS edge from symbol to decision
// that a later supported verdict removes. Mislinked removes the link.
func (s *Service) applyToDecision(ctx context.Context, c claim, r Result, decisive bool, record func(map[string]any, string)) (string, error) {
	contradicts := model.NewEdgeID(c.symbol.ID, c.node.ID, model.EdgeContradicts)
	clearDrift := func() error {
		if _, err := store.DeleteEdges(ctx, s.store, []string{contradicts}); err != nil {
			return fmt.Errorf("clear recorded drift: %w", err)
		}
		return nil
	}
	if decisive && r.Verdict == Mislinked {
		if err := s.links.RejectLink(ctx, c.node.ID, c.symbol.ID); err != nil {
			return "", err
		}
		return "link removed", clearDrift()
	}

	link := c.link
	attrs := maps.Clone(link.Attrs)
	if attrs == nil {
		attrs = map[string]any{}
	}
	record(attrs, c.spec.Fingerprint)
	setOrDelete(attrs, attrNote, r.Note)
	link.Attrs = attrs
	edges, effect := []model.Edge{link}, "marked verified"
	if decisive && r.Verdict == Drifted {
		prov, conf := model.ResolveProvenance(r.Confidence, s.threshold)
		edges = append(edges, model.Edge{
			ID: contradicts, Src: c.symbol.ID, Dst: c.node.ID, Kind: model.EdgeContradicts,
			Provenance: prov, Confidence: conf, CreatedAt: s.now(),
			Attrs: map[string]any{attrNote: r.Note, attrQuote: r.Quote},
		})
		effect = "drift recorded"
	}
	if decisive && r.Verdict == Supported {
		if err := clearDrift(); err != nil {
			return "", err
		}
	}
	if err := s.store.UpsertEdges(ctx, edges); err != nil {
		return "", fmt.Errorf("update decision link: %w", err)
	}
	return effect, nil
}
