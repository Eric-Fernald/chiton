// Package subagent emits semantic-extraction task specs (Pending) for a host assistant and ingests its
// schema-pinned JSON results (Submit). chiton calls no LLM and has no prompt for ingested text to hijack.
package subagent

import (
	"context"
	"errors"
	"fmt"
	"net"
	"strings"
	"time"

	"github.com/Eric-Fernald/chiton/internal/embed"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Task kinds.
const (
	KindDocConcepts      = "doc_concepts"
	KindImageDescription = "image_description"
)

// extractionStatusDone marks a source node whose semantic extraction is
// complete, so Pending will not re-emit it.
const extractionStatusDone = "done"

// Service emits extraction task specs and ingests their results. With an embedder, Submit embeds new
// concepts inline so they are immediately retrievable. Otherwise the next `chiton embed` does.
type Service struct {
	store     store.GraphStore
	embedder  embed.Embedder
	threshold float32
	now       func() time.Time
}

// New builds a Service. Pass embedder=nil to defer concept embedding to
// `chiton embed`. threshold<=0 uses model.DefaultConfidenceThreshold.
func New(s store.GraphStore, embedder embed.Embedder, threshold float32) *Service {
	if threshold <= 0 {
		threshold = model.DefaultConfidenceThreshold
	}
	return &Service{store: s, embedder: embedder, threshold: threshold, now: time.Now}
}

// TaskSpec is one unit of work for the host assistant: the doc section's text or the image's path, plus
// the Schema and Instructions for the JSON to return.
type TaskSpec struct {
	ID           string `json:"id"`    // source node ID (also the submit task_id)
	Kind         string `json:"kind"`  // doc_concepts | image_description
	Label        string `json:"label"` // human-readable source label
	SourceURI    string `json:"source_uri"`
	Input        string `json:"input"`  // section text, or image file path
	Schema       string `json:"schema"` // expected JSON shape for submit_extraction
	Instructions string `json:"instructions"`
}

// Pending returns task specs for doc_section and image nodes whose attrs["extraction_status"] is not "done".
// kind filters to one task kind (""=both); limit<=0 means no cap.
func (s *Service) Pending(ctx context.Context, kind string, limit int) ([]TaskSpec, error) {
	nodes, err := store.NodesMatching(ctx, s.store, store.NodeFilter{Kinds: []model.NodeKind{model.KindDocSection, model.KindImage}})
	if err != nil {
		return nil, fmt.Errorf("load nodes: %w", err)
	}
	var specs []TaskSpec
	for _, n := range nodes {
		if model.AttrString(n.Attrs, "extraction_status") == extractionStatusDone {
			continue
		}
		var spec TaskSpec
		switch n.Kind {
		case model.KindDocSection:
			if kind != "" && kind != KindDocConcepts {
				continue
			}
			text := model.AttrString(n.Attrs, "text")
			if strings.TrimSpace(text) == "" {
				continue // nothing to extract from an empty section
			}
			spec = docTask(n, text)
		case model.KindImage:
			if kind != "" && kind != KindImageDescription {
				continue
			}
			spec = imageTask(n)
		default:
			continue
		}
		specs = append(specs, spec)
		if limit > 0 && len(specs) >= limit {
			break
		}
	}
	return specs, nil
}

// ConceptResult is one extracted concept.
type ConceptResult struct {
	Label       string  `json:"label"`
	Description string  `json:"description"`
	Confidence  float32 `json:"confidence"`
}

// EdgeResult links From (a concept label from the same submission, or "" for the source node) to the
// concept labelled To. Kind is a semantic edge kind, and an empty or unknown Kind defaults to DESCRIBES.
type EdgeResult struct {
	From       string  `json:"from"`
	To         string  `json:"to"`
	Kind       string  `json:"kind"`
	Confidence float32 `json:"confidence"`
}

// SubmitResult is the host assistant's structured answer for one task.
type SubmitResult struct {
	TaskID   string          `json:"task_id"`
	Concepts []ConceptResult `json:"concepts"`
	Edges    []EdgeResult    `json:"edges"`
}

// SubmitReport summarises what Submit wrote.
type SubmitReport struct {
	TaskID        string `json:"task_id"`
	ConceptsAdded int    `json:"concepts_added"`
	EdgesAdded    int    `json:"edges_added"`
	Ambiguous     int    `json:"ambiguous"`
}

// Submit writes concept nodes and semantic edges (INFERRED, or AMBIGUOUS below threshold), marks the source
// done, and embeds the new concepts if the Service has an embedder. Stable IDs make re-submits a no-op.
func (s *Service) Submit(ctx context.Context, r SubmitResult) (SubmitReport, error) {
	if r.TaskID == "" {
		return SubmitReport{}, fmt.Errorf("submit: task_id is required")
	}
	// Reject a read-only store with an actionable message before any work, such as the embed round-trip below.
	if store.IsReadOnly(s.store) {
		return SubmitReport{}, fmt.Errorf("submit: store is read-only — restart 'chiton serve --writable' or use 'chiton extract submit'")
	}
	src, err := s.store.GetNode(ctx, r.TaskID)
	if err != nil {
		return SubmitReport{}, fmt.Errorf("load source node: %w", err)
	}
	if src == nil {
		return SubmitReport{}, fmt.Errorf("submit: no node with id %q (call pending_extractions first)", r.TaskID)
	}

	now := s.now()
	rep := SubmitReport{TaskID: r.TaskID}

	// Build concept nodes, keyed by label for edge resolution.
	conceptID := map[string]string{}
	var nodes []model.Node
	for _, c := range r.Concepts {
		label := strings.TrimSpace(c.Label)
		if label == "" {
			continue
		}
		prov, conf := model.ResolveProvenance(c.Confidence, s.threshold)
		if prov == model.ProvAmbiguous {
			rep.Ambiguous++
		}
		uri := "concept://" + model.Slug(label, "concept")
		id := model.NewNodeID(uri, label)
		conceptID[label] = id
		n := model.Node{
			ID:        id,
			Kind:      model.KindConcept,
			Label:     label,
			SourceURI: uri,
			Attrs: map[string]any{
				"description":    strings.TrimSpace(c.Description),
				"extracted_from": r.TaskID,
			},
			Provenance: prov,
			Confidence: conf,
			CreatedAt:  now,
			UpdatedAt:  now,
		}
		nodes = append(nodes, n)
		rep.ConceptsAdded++
	}

	// Embedding is best-effort, since the next `chiton embed` covers any concept left without a vector.
	var toEmbed []store.Embedding
	if s.embedder != nil {
		toEmbed = s.embedConcepts(ctx, nodes)
	}

	// Mark the source node done (preserve its other attrs).
	if src.Attrs == nil {
		src.Attrs = map[string]any{}
	}
	src.Attrs["extraction_status"] = extractionStatusDone
	nodes = append(nodes, *src)

	if err := s.store.UpsertNodes(ctx, nodes); err != nil {
		return SubmitReport{}, fmt.Errorf("upsert concept nodes: %w", err)
	}

	// Build edges. Default From is the source node; default kind is DESCRIBES.
	var edges []model.Edge
	for _, e := range r.Edges {
		toID, ok := conceptID[strings.TrimSpace(e.To)]
		if !ok {
			continue // edge points at a concept we didn't create; skip
		}
		fromID := r.TaskID
		if f := strings.TrimSpace(e.From); f != "" {
			if cid, ok := conceptID[f]; ok {
				fromID = cid
			}
		}
		kind := edgeKind(e.Kind)
		prov, conf := model.ResolveProvenance(e.Confidence, s.threshold)
		edges = append(edges, model.Edge{
			ID:         model.NewEdgeID(fromID, toID, kind),
			Src:        fromID,
			Dst:        toID,
			Kind:       kind,
			Provenance: prov,
			Confidence: conf,
			CreatedAt:  now,
		})
		rep.EdgesAdded++
	}

	// If no edge survived (none declared, or all pointed at uncreated concepts), default to DESCRIBES from the
	// source to every concept so none is orphaned.
	if len(edges) == 0 {
		for _, c := range r.Concepts {
			toID, ok := conceptID[strings.TrimSpace(c.Label)]
			if !ok {
				continue
			}
			prov, conf := model.ResolveProvenance(c.Confidence, s.threshold)
			edges = append(edges, model.Edge{
				ID:         model.NewEdgeID(r.TaskID, toID, model.EdgeDescribes),
				Src:        r.TaskID,
				Dst:        toID,
				Kind:       model.EdgeDescribes,
				Provenance: prov,
				Confidence: conf,
				CreatedAt:  now,
			})
			rep.EdgesAdded++
		}
	}

	if len(edges) > 0 {
		if err := s.store.UpsertEdges(ctx, edges); err != nil {
			return SubmitReport{}, fmt.Errorf("upsert semantic edges: %w", err)
		}
	}

	if len(toEmbed) > 0 {
		if es, ok := s.store.(store.EmbeddingStore); ok {
			if err := es.UpsertEmbeddings(ctx, toEmbed); err != nil {
				return SubmitReport{}, fmt.Errorf("embed concepts: %w", err)
			}
		}
	}

	return rep, nil
}

// embedConcepts embeds concepts in one batch, retrying per concept on failure so only concepts that fail alone
// lose a vector. It skips the retries for an unreachable embedder, since each would hit the same timeout.
func (s *Service) embedConcepts(ctx context.Context, concepts []model.Node) []store.Embedding {
	embs, exact, err := embed.NodeEmbeddings(ctx, s.embedder, concepts)
	if err == nil && exact {
		return embs
	}
	if embedderUnreachable(err) {
		return nil
	}
	embs = nil
	for i := range concepts {
		one, exact, err := embed.NodeEmbeddings(ctx, s.embedder, concepts[i:i+1])
		if embedderUnreachable(err) {
			break
		}
		if err == nil && exact {
			embs = append(embs, one...)
		}
	}
	return embs
}

// embedderUnreachable reports a failure to reach the embedder at all (network error, timeout, cancellation),
// as opposed to one input it rejected.
func embedderUnreachable(err error) bool {
	var ne net.Error
	return errors.As(err, &ne) || errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded)
}

// edgeKind normalises a host-supplied edge kind to a known semantic kind,
// defaulting to DESCRIBES for empty/unknown values.
func edgeKind(raw string) model.EdgeKind {
	switch model.EdgeKind(strings.ToUpper(strings.TrimSpace(raw))) {
	case model.EdgeDescribes:
		return model.EdgeDescribes
	case model.EdgeElaborates:
		return model.EdgeElaborates
	case model.EdgeRationaleFor:
		return model.EdgeRationaleFor
	case model.EdgeSimilarTo:
		return model.EdgeSimilarTo
	case model.EdgeContradicts:
		return model.EdgeContradicts
	case model.EdgeDerivedFrom:
		return model.EdgeDerivedFrom
	case model.EdgeReferences:
		return model.EdgeReferences
	default:
		return model.EdgeDescribes
	}
}
