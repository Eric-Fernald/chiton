// Package subagent emits semantic-extraction task specs (Pending) for a host assistant and ingests its
// schema-pinned JSON results (Submit). chiton calls no LLM and has no prompt for ingested text to hijack.
package subagent

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"net"
	"slices"
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
		spec, ok := taskFor(n, kind)
		if !ok {
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
	TaskID          string `json:"task_id"`
	ConceptsAdded   int    `json:"concepts_added"`
	ConceptsUpdated int    `json:"concepts_updated"`
	EdgesAdded      int    `json:"edges_added"`
	EdgesRemoved    int    `json:"edges_removed"`
	Ambiguous       int    `json:"ambiguous"`
}

// Attrs Submit and SweepOrphans keep on concepts. attrExtractedFrom is also stamped on the edges a task writes.
const (
	attrExtractedFrom = "extracted_from" // ID of the task (source node) that wrote it
	attrOrphanedAt    = "orphaned_at"    // RFC 3339 time SweepOrphans found no source linking to the concept
)

// Submit writes concept nodes and semantic edges (INFERRED, or AMBIGUOUS below threshold), marks the source
// done, and embeds the written concepts if the Service has an embedder. A result is the source's full current
// set: edges an earlier submit for this task wrote and this one omits are deleted. Stable IDs make an identical
// re-submit a no-op.
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
	// Bind task_id to a node Pending issues tasks for, so a submit can't mark or link arbitrary nodes.
	if _, ok := taskFor(*src, ""); !ok {
		return SubmitReport{}, fmt.Errorf("submit: node %q is a %s, not an extraction task (want a doc_section with text, or an image)", r.TaskID, src.Kind)
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
		uri := "concept://" + model.Slug(label, "concept")
		id := model.NewNodeID(uri, label)
		conceptID[label] = id
		n := model.Node{
			ID:        id,
			Kind:      model.KindConcept,
			Label:     label,
			SourceURI: uri,
			Attrs: map[string]any{
				"description":     strings.TrimSpace(c.Description),
				attrExtractedFrom: r.TaskID,
			},
			Provenance: prov,
			Confidence: conf,
			CreatedAt:  now,
			UpdatedAt:  now,
		}
		nodes = append(nodes, n)
	}
	// conceptID keeps the dropped IDs, so this task's edges still link to concepts other tasks created.
	if nodes, rep.ConceptsUpdated, err = s.writableConcepts(ctx, r.TaskID, nodes); err != nil {
		return SubmitReport{}, err
	}
	rep.ConceptsAdded = len(nodes) - rep.ConceptsUpdated
	for _, n := range nodes {
		if n.Provenance == model.ProvAmbiguous {
			rep.Ambiguous++
		}
	}

	// Embedding is best-effort, since the next `chiton embed` covers any concept left without a vector.
	var toEmbed []store.Embedding
	if s.embedder != nil {
		toEmbed = s.embedConcepts(ctx, nodes)
	}

	// The writes below are separate, so their order makes a failure safe to retry: new edges land before the
	// superseded ones go, and the source is marked done last, which keeps the task pending until all of it is in.
	if len(nodes) > 0 {
		if err := s.store.UpsertNodes(ctx, nodes); err != nil {
			return SubmitReport{}, fmt.Errorf("upsert concept nodes: %w", err)
		}
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
			Attrs:      map[string]any{attrExtractedFrom: r.TaskID},
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
				Attrs:      map[string]any{attrExtractedFrom: r.TaskID},
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
	stale, err := s.staleEdges(ctx, r.TaskID, edges)
	if err != nil {
		return SubmitReport{}, err
	}
	if rep.EdgesRemoved, err = store.DeleteEdges(ctx, s.store, stale); err != nil {
		return SubmitReport{}, fmt.Errorf("delete superseded edges: %w", err)
	}

	// Mark the source node done (preserve its other attrs).
	if src.Attrs == nil {
		src.Attrs = map[string]any{}
	}
	src.Attrs["extraction_status"] = extractionStatusDone
	if err := s.store.UpsertNodes(ctx, []model.Node{*src}); err != nil {
		return SubmitReport{}, fmt.Errorf("mark source done: %w", err)
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

// writableConcepts keeps the concepts taskID may write and counts those that already exist. Concept IDs are
// global (concept://<slug>), so a task may link to another task's concept but only rewrite one it created or one
// SweepOrphans left without a source.
func (s *Service) writableConcepts(ctx context.Context, taskID string, concepts []model.Node) ([]model.Node, int, error) {
	if len(concepts) == 0 {
		return concepts, 0, nil
	}
	ids := make([]string, len(concepts))
	for i, c := range concepts {
		ids[i] = c.ID
	}
	existing, err := store.GetNodes(ctx, s.store, ids)
	if err != nil {
		return nil, 0, fmt.Errorf("load existing concepts: %w", err)
	}
	have := make(map[string]model.Node, len(existing))
	for _, n := range existing {
		have[n.ID] = n
	}
	var keep []model.Node
	updated := 0
	for _, c := range concepts {
		old, exists := have[c.ID]
		if !exists {
			keep = append(keep, c)
			continue
		}
		orphaned := model.AttrString(old.Attrs, attrOrphanedAt) != ""
		owned := model.AttrString(old.Attrs, attrExtractedFrom) == taskID
		// Only a verification verdict lifts a demotion a verification verdict made.
		if old.Attrs[model.AttrVerifyDemoted] == true {
			c.Provenance = old.Provenance
		}
		if !orphaned && (!owned || sameConcept(old, c)) {
			continue
		}
		// Keep what other flows recorded on the concept, such as its last verification verdict.
		attrs := maps.Clone(old.Attrs)
		if attrs == nil {
			attrs = map[string]any{}
		}
		maps.Copy(attrs, c.Attrs)
		delete(attrs, attrOrphanedAt)
		c.Attrs = attrs
		c.CreatedAt = old.CreatedAt
		keep = append(keep, c)
		updated++
	}
	return keep, updated, nil
}

// sameConcept reports whether a and b agree on everything a submit sets besides the owner.
func sameConcept(a, b model.Node) bool {
	return a.Provenance == b.Provenance && a.Confidence == b.Confidence &&
		model.AttrString(a.Attrs, "description") == model.AttrString(b.Attrs, "description")
}

// staleEdges returns the IDs of edges an earlier submit for taskID wrote that keep omits: inferred edges from the
// source to a concept, and edges between concepts stamped with the task. An edge between concepts written before
// the stamp existed has no owner and stays.
func (s *Service) staleEdges(ctx context.Context, taskID string, keep []model.Edge) ([]string, error) {
	sub, err := s.store.Neighbors(ctx, taskID, 2, SemanticKinds)
	if err != nil {
		return nil, fmt.Errorf("load prior edges: %w", err)
	}
	kept := make(map[string]bool, len(keep))
	for _, e := range keep {
		kept[e.ID] = true
	}
	isConcept := make(map[string]bool, len(sub.Nodes))
	for _, n := range sub.Nodes {
		isConcept[n.ID] = n.Kind == model.KindConcept
	}
	var stale []string
	for _, e := range sub.Edges {
		if kept[e.ID] || !IsExtractionLink(e) {
			continue
		}
		fromSource := e.Src == taskID && isConcept[e.Dst]
		if fromSource || model.AttrString(e.Attrs, attrExtractedFrom) == taskID {
			stale = append(stale, e.ID)
		}
	}
	return stale, nil
}

// SweepReport summarises what SweepOrphans changed.
type SweepReport struct {
	Demoted int `json:"demoted"`
}

// IsExtractionLink reports whether e is an edge a submit wrote. `chiton embed --link-similar` also writes INFERRED
// SIMILAR_TO edges, between any two embedded nodes. Those carry no task stamp and say only that two nodes read
// alike, so they are not links from a source to a concept.
func IsExtractionLink(e model.Edge) bool {
	if e.Provenance == model.ProvExtracted || !slices.Contains(SemanticKinds, e.Kind) {
		return false
	}
	return e.Kind != model.EdgeSimilarTo || model.AttrString(e.Attrs, attrExtractedFrom) != ""
}

// SweepOrphans demotes to AMBIGUOUS every concept no source backs, and stamps it with attrs["orphaned_at"] so the
// next task to extract it can take it over. A source backs the concepts it links to and, because a submit may
// relate one of its concepts to another without linking the second, every concept reachable from those through
// links between concepts. Already-stamped concepts are skipped, so a repeat sweep changes nothing.
func (s *Service) SweepOrphans(ctx context.Context) (SweepReport, error) {
	if store.IsReadOnly(s.store) {
		return SweepReport{}, fmt.Errorf("sweep: store is read-only — stop 'chiton serve' or restart it with --writable")
	}
	concepts, err := store.NodesMatching(ctx, s.store, store.NodeFilter{Kinds: []model.NodeKind{model.KindConcept}})
	if err != nil {
		return SweepReport{}, fmt.Errorf("load concepts: %w", err)
	}
	isConcept := make(map[string]bool, len(concepts))
	for _, c := range concepts {
		isConcept[c.ID] = true
	}
	// Reading each concept's own edges costs the size of the semantic layer, not of the whole graph.
	sourced := map[string]bool{}
	peers := map[string][]string{}
	var frontier []string
	for _, c := range concepts {
		sub, err := s.store.Neighbors(ctx, c.ID, 1, SemanticKinds)
		if err != nil {
			return SweepReport{}, fmt.Errorf("load links of concept %s: %w", c.ID, err)
		}
		for _, e := range sub.Edges {
			if !IsExtractionLink(e) {
				continue
			}
			other := e.Src
			if other == c.ID {
				other = e.Dst
			}
			switch {
			case isConcept[other]:
				peers[c.ID] = append(peers[c.ID], other)
			case !sourced[c.ID]:
				sourced[c.ID] = true
				frontier = append(frontier, c.ID)
			}
		}
	}
	for len(frontier) > 0 {
		id := frontier[0]
		frontier = frontier[1:]
		for _, p := range peers[id] {
			if !sourced[p] {
				sourced[p] = true
				frontier = append(frontier, p)
			}
		}
	}
	stamp := s.now().UTC().Format(time.RFC3339)
	var demoted []model.Node
	for _, c := range concepts {
		if sourced[c.ID] || model.AttrString(c.Attrs, attrOrphanedAt) != "" {
			continue
		}
		attrs := maps.Clone(c.Attrs)
		if attrs == nil {
			attrs = map[string]any{}
		}
		attrs[attrOrphanedAt] = stamp
		c.Attrs, c.Provenance = attrs, model.ProvAmbiguous
		demoted = append(demoted, c)
	}
	if len(demoted) > 0 {
		if err := s.store.UpsertNodes(ctx, demoted); err != nil {
			return SweepReport{}, fmt.Errorf("demote orphaned concepts: %w", err)
		}
	}
	return SweepReport{Demoted: len(demoted)}, nil
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

// SemanticKinds are the edge kinds a submit may write.
var SemanticKinds = []model.EdgeKind{
	model.EdgeDescribes, model.EdgeElaborates, model.EdgeRationaleFor, model.EdgeSimilarTo,
	model.EdgeContradicts, model.EdgeDerivedFrom, model.EdgeReferences,
}

// edgeKind normalises a host-supplied edge kind to a known semantic kind,
// defaulting to DESCRIBES for empty/unknown values.
func edgeKind(raw string) model.EdgeKind {
	if k := model.EdgeKind(strings.ToUpper(strings.TrimSpace(raw))); slices.Contains(SemanticKinds, k) {
		return k
	}
	return model.EdgeDescribes
}
