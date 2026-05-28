// Package subagent implements chiton's subagent-mode semantic extraction.
//
// chiton never calls an LLM itself in this mode. Instead it emits structured
// task specs (Pending) describing what needs extracting — doc-section concepts,
// image descriptions — and ingests the host assistant's structured results
// (Submit), writing concept nodes and semantic edges into the graph tagged
// INFERRED or AMBIGUOUS by confidence. The MCP tools and the `chiton extract`
// CLI are both thin wrappers over this one Service, so a future daemon mode
// (direct LLM calls) can reuse the exact same write path.
//
// Output shape is pinned by an explicit JSON schema in every task spec, so
// ingested content can never smuggle instructions into a chiton prompt — there
// is no chiton prompt. That is the prompt-injection mitigation the security
// section of the plan calls for.
package subagent

import (
	"context"
	"fmt"
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

// DefaultConfidenceThreshold splits INFERRED from AMBIGUOUS when none is set.
const DefaultConfidenceThreshold = 0.5

// Service emits extraction task specs and ingests their results.
// embedder is optional; when present, Submit embeds new concept nodes inline
// so they are immediately retrievable (otherwise the next `chiton embed` does).
type Service struct {
	store     store.GraphStore
	embedder  embed.Embedder
	threshold float32
	now       func() time.Time
}

// New builds a Service. Pass embedder=nil to defer concept embedding to
// `chiton embed`. threshold<=0 uses DefaultConfidenceThreshold.
func New(s store.GraphStore, embedder embed.Embedder, threshold float32) *Service {
	if threshold <= 0 {
		threshold = DefaultConfidenceThreshold
	}
	return &Service{store: s, embedder: embedder, threshold: threshold, now: time.Now}
}

// TaskSpec is one unit of work for the host assistant. Input carries the doc
// section's text or the image's file path; Schema/Instructions tell the
// assistant exactly what JSON to return.
type TaskSpec struct {
	ID           string `json:"id"`    // source node ID (also the submit task_id)
	Kind         string `json:"kind"`  // doc_concepts | image_description
	Label        string `json:"label"` // human-readable source label
	SourceURI    string `json:"source_uri"`
	Input        string `json:"input"`  // section text, or image file path
	Schema       string `json:"schema"` // expected JSON shape for submit_extraction
	Instructions string `json:"instructions"`
}

// Pending returns task specs for nodes that still need semantic extraction:
// doc_section and image nodes whose attrs["extraction_status"] != "done".
// kind filters to one task kind (""=both); limit<=0 means no cap.
func (s *Service) Pending(ctx context.Context, kind string, limit int) ([]TaskSpec, error) {
	nodes, err := s.store.AllNodes(ctx)
	if err != nil {
		return nil, fmt.Errorf("load nodes: %w", err)
	}
	var specs []TaskSpec
	for _, n := range nodes {
		if attrString(n, "extraction_status") == extractionStatusDone {
			continue
		}
		var spec TaskSpec
		switch n.Kind {
		case model.KindDocSection:
			if kind != "" && kind != KindDocConcepts {
				continue
			}
			text := attrString(n, "text")
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

// EdgeResult links the source node (by its task ID) or a concept to a concept.
// From may be "" (defaults to the source node) or a concept label from the same
// submission; To is a concept label. Kind is a semantic edge kind; empty
// defaults to DESCRIBES.
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

// Submit ingests one task's results: it creates/updates concept nodes and
// semantic edges (provenance INFERRED, or AMBIGUOUS below threshold), marks the
// source node done, and (if an embedder is wired) embeds the new concepts.
// It is idempotent: stable IDs mean re-submitting the same result is a no-op.
func (s *Service) Submit(ctx context.Context, r SubmitResult) (SubmitReport, error) {
	if r.TaskID == "" {
		return SubmitReport{}, fmt.Errorf("submit: task_id is required")
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
	var toEmbed []store.Embedding
	for _, c := range r.Concepts {
		label := strings.TrimSpace(c.Label)
		if label == "" {
			continue
		}
		prov, conf := resolveProvenance(c.Confidence, s.threshold)
		if prov == model.ProvAmbiguous {
			rep.Ambiguous++
		}
		uri := "concept://" + slug(label)
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

		if s.embedder != nil {
			if text := embed.TextFor(n); text != "" {
				vecs, err := s.embedder.Embed(ctx, []string{text})
				if err == nil && len(vecs) == 1 {
					toEmbed = append(toEmbed, store.Embedding{NodeID: id, Vec: vecs[0]})
				}
			}
		}
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
		prov, conf := resolveProvenance(e.Confidence, s.threshold)
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

	// If the result declared no edges, default to DESCRIBES from the source to
	// every extracted concept, so concepts are never orphaned in the graph.
	if len(r.Edges) == 0 {
		for _, c := range r.Concepts {
			toID, ok := conceptID[strings.TrimSpace(c.Label)]
			if !ok {
				continue
			}
			prov, conf := resolveProvenance(c.Confidence, s.threshold)
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

// resolveProvenance maps a host-supplied confidence to a provenance tag and a
// clamped confidence. At or above threshold → INFERRED; below → AMBIGUOUS.
func resolveProvenance(conf, threshold float32) (model.Provenance, float32) {
	if conf < 0 {
		conf = 0
	}
	if conf > 1 {
		conf = 1
	}
	if conf >= threshold {
		return model.ProvInferred, conf
	}
	return model.ProvAmbiguous, conf
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

func attrString(n model.Node, key string) string {
	if n.Attrs == nil {
		return ""
	}
	if v, ok := n.Attrs[key].(string); ok {
		return v
	}
	return ""
}

// slug lowercases and replaces runs of non-alphanumerics with single hyphens,
// for a readable concept:// URI. Empty input yields "concept".
func slug(s string) string {
	var b strings.Builder
	lastHyphen := false
	for _, r := range strings.ToLower(s) {
		switch {
		case (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9'):
			b.WriteRune(r)
			lastHyphen = false
		default:
			if !lastHyphen {
				b.WriteByte('-')
				lastHyphen = true
			}
		}
	}
	out := strings.Trim(b.String(), "-")
	if out == "" {
		return "concept"
	}
	return out
}
