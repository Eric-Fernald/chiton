package verify

import (
	"cmp"
	"context"
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// conceptClaim builds the task for concept n from the text of every source an extraction linked to it. ok is
// false when no such source has text, which leaves nothing to quote: an orphaned concept, or one only an image
// describes. A node that merely reads like the concept (a similarity edge) never claimed it and is not evidence.
func (s *Service) conceptClaim(ctx context.Context, n model.Node) (c claim, ok bool, err error) {
	sub, err := s.store.Neighbors(ctx, n.ID, 1, subagent.SemanticKinds)
	if err != nil {
		return claim{}, false, fmt.Errorf("load sources of concept %s: %w", n.ID, err)
	}
	linked := map[string]bool{}
	for _, e := range sub.Edges {
		if subagent.IsExtractionLink(e) {
			linked[e.Src], linked[e.Dst] = true, true
		}
	}
	var evidence []Evidence
	truncated := false
	for _, src := range sub.Nodes {
		text := strings.TrimSpace(model.AttrString(src.Attrs, "text"))
		if src.ID == n.ID || !linked[src.ID] || src.Kind == model.KindConcept || text == "" {
			continue
		}
		text, clipped := clip(text)
		truncated = truncated || clipped
		evidence = append(evidence, Evidence{NodeID: src.ID, Label: src.Label, SourceURI: src.SourceURI, Text: text})
	}
	if len(evidence) == 0 {
		return claim{}, false, nil
	}
	// A stable order and cut keep the fingerprint the same from one call to the next.
	slices.SortFunc(evidence, func(a, b Evidence) int { return cmp.Compare(a.NodeID, b.NodeID) })
	if len(evidence) > maxEvidence {
		evidence, truncated = evidence[:maxEvidence], true
	}
	text := conceptClaimText(n.Label, model.AttrString(n.Attrs, "description"))
	return claim{node: n, spec: TaskSpec{
		ID:           n.ID,
		Kind:         KindConceptSupport,
		Label:        n.Label,
		Claim:        text,
		Evidence:     evidence,
		Fingerprint:  fingerprint(KindConceptSupport, text, evidence),
		Truncated:    truncated,
		Schema:       conceptSchema,
		Instructions: conceptInstructions,
	}}, true, nil
}

func conceptClaimText(label, description string) string {
	if description == "" {
		return label
	}
	return label + ": " + description
}

// applyToConcept writes a verdict onto the concept and returns what it did. One unsupported verdict demotes the
// concept to AMBIGUOUS, which a later supported verdict undoes. A second one, given for different evidence,
// deletes it. model.AttrVerifyDemoted carries the demotion between verdicts, because the verdict attrs are
// overwritten by every later verdict, including ones too unsure to act on.
func (s *Service) applyToConcept(ctx context.Context, c claim, r Result, decisive bool, record func(map[string]any, string)) (string, error) {
	n := c.node
	attrs := maps.Clone(n.Attrs)
	if attrs == nil {
		attrs = map[string]any{}
	}
	demoted := attrs[model.AttrVerifyDemoted] == true
	fp, effect := c.spec.Fingerprint, "marked verified"
	if decisive {
		switch r.Verdict {
		case Unsupported:
			if demoted {
				if _, err := store.DeleteNodes(ctx, s.store, []model.Node{n}); err != nil {
					return "", fmt.Errorf("delete unsupported concept: %w", err)
				}
				return "deleted: unsupported for the second time", nil
			}
			attrs[model.AttrVerifyDemoted] = true
			n.Provenance, effect = model.ProvAmbiguous, "demoted to AMBIGUOUS"
		case Revised:
			attrs["description"] = r.Description
			// The stored fingerprint must cover the new description, or the rewrite would requeue the concept.
			fp, effect = fingerprint(KindConceptSupport, conceptClaimText(n.Label, r.Description), c.spec.Evidence), "description rewritten"
		}
		if r.Verdict != Unsupported && demoted {
			delete(attrs, model.AttrVerifyDemoted)
			n.Provenance, _ = model.ResolveProvenance(n.Confidence, s.threshold)
		}
	}
	record(attrs, fp)
	n.Attrs = attrs
	if err := s.store.UpsertNodes(ctx, []model.Node{n}); err != nil {
		return "", fmt.Errorf("update concept: %w", err)
	}
	return effect, nil
}
