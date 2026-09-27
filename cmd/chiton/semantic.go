package main

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/config"
	"github.com/Eric-Fernald/chiton/internal/embed"
	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/retrieve"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	"github.com/Eric-Fernald/chiton/internal/vector"
)

const embedBatchSize = 64

// embedCmd embeds the graph's embeddable nodes (symbols, doc sections,
// concepts) and persists the vectors into the store's encrypted keyspace.
func embedCmd() *cobra.Command {
	var provider, modelName string
	var all, linkSimilar bool
	var similarThreshold float64

	cmd := &cobra.Command{
		Use:   "embed",
		Short: "Embed symbol/doc/concept nodes for semantic search",
		Long: `Compute embeddings for every embeddable node (symbols with docstrings,
doc sections, concepts) and store them alongside the graph. Only nodes without
an embedding are processed unless --all is given.

The default provider is local Ollama (no data leaves the machine). Use
--provider hash for an offline, dependency-free embedder (lower quality; good
for CI and demos). Restart 'chiton serve' afterwards to load the new vectors.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, gs, err := loadStore(false)
			if err != nil {
				return err
			}
			defer gs.Close()
			out := cmd.OutOrStdout()

			es, ok := store.GraphStore(gs).(store.EmbeddingStore)
			if !ok {
				return fmt.Errorf("the configured store does not support embeddings")
			}

			emb, err := newEmbedder(cfg, provider, modelName)
			if err != nil {
				return err
			}
			// Report the provider actually in use, honoring the --provider override
			// rather than the (possibly superseded) config value.
			effectiveProvider := cfg.Embed.Provider
			if provider != "" {
				effectiveProvider = provider
			}
			fmt.Fprintf(out, "embed: provider=%s model=%s\n", effectiveProvider, emb.Model())

			ctx := cmd.Context()
			nodes, err := gs.AllNodes(ctx)
			if err != nil {
				return fmt.Errorf("load nodes: %w", err)
			}

			existing := map[string]bool{}
			if !all {
				have, err := store.EmbeddingIDs(ctx, es)
				if err != nil {
					return fmt.Errorf("load existing embeddings: %w", err)
				}
				for _, id := range have {
					existing[id] = true
				}
			}

			// work holds the embeddable nodes (non-empty TextFor) still lacking a vector. Building texts per
			// batch keeps only one batch of doc-section text, which can be large, live at a time.
			var work []model.Node
			for _, n := range nodes {
				if (all || !existing[n.ID]) && embed.TextFor(n) != "" {
					work = append(work, n)
				}
			}
			if len(work) == 0 {
				fmt.Fprintln(out, "embed: nothing to do (all embeddable nodes already have vectors; use --all to re-embed)")
				return nil
			}
			fmt.Fprintf(out, "embed: %d node(s) to embed\n", len(work))

			written := 0
			for start := 0; start < len(work); start += embedBatchSize {
				// Stop between batches on Ctrl-C (root context cancelled) so
				// the deferred store close runs cleanly.
				if err := ctx.Err(); err != nil {
					return err
				}
				embs, _, err := embed.NodeEmbeddings(ctx, emb, work[start:min(start+embedBatchSize, len(work))])
				if err != nil {
					return fmt.Errorf("embed batch: %w", err)
				}
				if err := es.UpsertEmbeddings(ctx, embs); err != nil {
					return fmt.Errorf("store embeddings: %w", err)
				}
				written += len(embs)
				fmt.Fprintf(out, "embed: %d/%d\n", written, len(work))
			}

			fmt.Fprintf(out, "embed: done — %d embeddings written (dim %d)\n", written, emb.Dim())

			if linkSimilar {
				n, err := linkSimilarNodes(ctx, gs, es, float32(similarThreshold))
				if err != nil {
					return fmt.Errorf("link-similar: %w", err)
				}
				fmt.Fprintf(out, "embed: link-similar — %d SIMILAR_TO edge(s) added (threshold %.2f)\n", n, similarThreshold)
			}

			fmt.Fprintln(out, "embed: restart 'chiton serve' to enable semantic_search")
			return nil
		},
	}

	cmd.Flags().StringVar(&provider, "provider", "", "Embedding provider override (ollama|hash)")
	cmd.Flags().StringVar(&modelName, "model", "", "Embedding model override")
	cmd.Flags().BoolVar(&all, "all", false, "Re-embed every embeddable node, not just those missing a vector")
	cmd.Flags().BoolVar(&linkSimilar, "link-similar", false, "Also add SIMILAR_TO edges between highly-similar nodes")
	cmd.Flags().Float64Var(&similarThreshold, "similar-threshold", 0.85, "Cosine threshold for --link-similar edges")
	return cmd
}

// linkSimilarNodes builds a flat index over all embeddings and adds a
// SIMILAR_TO edge (INFERRED, confidence=score) for each pair above threshold.
func linkSimilarNodes(ctx context.Context, gs store.GraphStore, es store.EmbeddingStore, threshold float32) (int, error) {
	embs, err := es.AllEmbeddings(ctx)
	if err != nil {
		return 0, err
	}
	idx := consumeIntoFlatIndex(slices.Clone(embs)) // clone so embs keeps raw vectors for the loop below

	seen := map[[2]string]bool{}
	var edges []model.Edge
	for _, e := range embs {
		for _, r := range idx.Search(e.Vec, 6) {
			if r.ID == e.NodeID || r.Score < threshold {
				continue
			}
			// Undirected: store one edge per unordered pair.
			a, b := e.NodeID, r.ID
			if a > b {
				a, b = b, a
			}
			key := [2]string{a, b}
			if seen[key] {
				continue
			}
			seen[key] = true
			edges = append(edges, model.Edge{
				ID:         model.NewEdgeID(a, b, model.EdgeSimilarTo),
				Src:        a,
				Dst:        b,
				Kind:       model.EdgeSimilarTo,
				Provenance: model.ProvInferred,
				Confidence: r.Score,
			})
		}
	}
	if len(edges) == 0 {
		return 0, nil
	}
	if err := gs.UpsertEdges(ctx, edges); err != nil {
		return 0, err
	}
	return len(edges), nil
}

// consumeIntoFlatIndex indexes embs and clears each embs[i].Vec, since the index keeps a normalized copy.
// That way raw and normalized vectors are never both fully live. Pass a clone to keep the raw vectors.
func consumeIntoFlatIndex(embs []store.Embedding) *vector.Flat {
	idx := vector.NewFlatSized(len(embs))
	item := make([]vector.Item, 1)
	for i := range embs {
		item[0] = vector.Item{ID: embs[i].NodeID, Vec: embs[i].Vec}
		idx.Add(item)
		embs[i].Vec = nil
	}
	return idx
}

// searchCmd runs hybrid retrieval from the command line.
func searchCmd() *cobra.Command {
	var k, hops, maxTokens int
	var sourcePrefix string
	var asJSON, includeAmbiguous bool

	cmd := &cobra.Command{
		Use:   "search <query>",
		Short: "Hybrid semantic + graph search over the embedded graph",
		Args:  cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, gs, err := loadStore(true)
			if err != nil {
				return err
			}
			defer gs.Close()
			out := cmd.OutOrStdout()

			es, ok := store.GraphStore(gs).(store.EmbeddingStore)
			if !ok {
				return fmt.Errorf("the configured store does not support embeddings")
			}
			embs, err := es.AllEmbeddings(cmd.Context())
			if err != nil {
				return fmt.Errorf("load embeddings: %w", err)
			}
			if len(embs) == 0 {
				fmt.Fprintln(out, "search: no embeddings indexed — run 'chiton embed' first")
				return nil
			}
			emb, err := newEmbedder(cfg, "", "")
			if err != nil {
				return err
			}
			clusterResults, _ := cluster.LoadSidecar(cluster.SidecarPath(cfg.GraphPath))
			eng := retrieve.New(gs, consumeIntoFlatIndex(embs), emb, clusterResults)

			query := strings.Join(args, " ")
			res, err := eng.Query(cmd.Context(), query, retrieve.Options{
				SeedK:            k,
				Hops:             hops,
				MaxTokens:        maxTokens,
				SourcePrefix:     sourcePrefix,
				IncludeAmbiguous: includeAmbiguous,
			})
			if err != nil {
				return err
			}

			if asJSON {
				return printJSON(out, res)
			}

			fmt.Fprintf(out, "%s\n\n", res.Summary)
			for _, s := range res.Nodes {
				loc := s.Node.SourceURI
				if s.Node.Span != nil {
					loc = fmt.Sprintf("%s:%d", loc, s.Node.Span.StartLine)
				}
				fmt.Fprintf(out, "  %.3f  [%-11s hop=%d %-9s] %-30s %s\n",
					s.Score, s.Node.Kind, s.Hop, s.Node.Provenance, s.Node.Label, loc)
			}
			return nil
		},
	}

	cmd.Flags().IntVar(&k, "k", 8, "Number of vector seeds")
	cmd.Flags().IntVar(&hops, "hops", 1, "Graph expansion depth around each seed")
	cmd.Flags().IntVar(&maxTokens, "max-tokens", 4000, "Approximate token budget for results")
	cmd.Flags().StringVar(&sourcePrefix, "source-prefix", "", "Only return nodes whose source_uri contains this substring")
	cmd.Flags().BoolVar(&includeAmbiguous, "include-ambiguous", false, "Include low-confidence AMBIGUOUS nodes")
	cmd.Flags().BoolVar(&asJSON, "json", false, "Output raw JSON")
	return cmd
}

// extractCmd is the parent for the subagent-mode extraction workflow.
func extractCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "extract",
		Short: "Subagent-mode semantic extraction (emit task specs, ingest results)",
		Long: `chiton emits structured task specs for doc sections and images that need
semantic extraction; you (or your assistant) run them and feed the structured
results back in. chiton never calls an LLM itself in this mode.

  chiton extract pending --out tasks.json   # emit task specs
  # ... run each task, producing results.json matching each spec's schema ...
  chiton extract submit results.json        # ingest concepts + edges`,
	}
	cmd.AddCommand(extractPendingCmd(), extractSubmitCmd())
	return cmd
}

func extractPendingCmd() *cobra.Command {
	var kind, outFile string
	var limit int

	cmd := &cobra.Command{
		Use:   "pending",
		Short: "Print pending extraction task specs as JSON",
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, gs, err := loadStore(true)
			if err != nil {
				return err
			}
			defer gs.Close()

			svc := subagent.New(gs, nil, cfg.Semantic.ConfidenceThreshold)
			specs, err := svc.Pending(cmd.Context(), kind, limit)
			if err != nil {
				return err
			}
			return emitSpecs(cmd.OutOrStdout(), specs, outFile,
				fmt.Sprintf("extract: %d pending task(s)", len(specs)))
		},
	}
	cmd.Flags().StringVar(&kind, "kind", "", "Filter to one kind: doc_concepts | image_description")
	cmd.Flags().IntVar(&limit, "limit", 0, "Max specs (0 = all)")
	cmd.Flags().StringVar(&outFile, "out", "", "Write specs to this file instead of stdout")
	return cmd
}

func extractSubmitCmd() *cobra.Command {
	var provider, modelName string
	var noEmbed bool

	cmd := &cobra.Command{
		Use:   "submit <results.json>",
		Short: "Ingest extraction results (a SubmitResult object or array)",
		Args:  cobra.ExactArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			cfg, err := config.Load()
			if err != nil {
				return err
			}

			data, err := os.ReadFile(args[0])
			if err != nil {
				return fmt.Errorf("read %s: %w", args[0], err)
			}
			results, err := parseOneOrMany[subagent.SubmitResult](data, "a SubmitResult")
			if err != nil {
				return err
			}

			gs, err := openGraphStore(cfg.GraphPath, false)
			if err != nil {
				return err
			}
			defer gs.Close()

			// Embed new concepts inline unless --no-embed; ignore embedder
			// construction errors so submit still works offline.
			var emb embed.Embedder
			if !noEmbed {
				emb, _ = newEmbedder(cfg, provider, modelName)
			}
			svc := subagent.New(gs, emb, cfg.Semantic.ConfidenceThreshold)

			out := cmd.OutOrStdout()
			var concepts, edges, ambiguous int
			for _, r := range results {
				rep, err := svc.Submit(cmd.Context(), r)
				if err != nil {
					return err
				}
				concepts += rep.ConceptsAdded
				edges += rep.EdgesAdded
				ambiguous += rep.Ambiguous
				fmt.Fprintf(out, "submit: %s — %d concept(s), %d edge(s), %d ambiguous\n",
					rep.TaskID, rep.ConceptsAdded, rep.EdgesAdded, rep.Ambiguous)
			}
			fmt.Fprintf(out, "submit: done — %d task(s), %d concept(s), %d edge(s), %d ambiguous\n",
				len(results), concepts, edges, ambiguous)
			if emb != nil {
				fmt.Fprintln(out, "submit: restart 'chiton serve' to index new concepts for semantic_search")
			}
			return nil
		},
	}
	cmd.Flags().StringVar(&provider, "provider", "", "Embedding provider for new concepts (ollama|hash)")
	cmd.Flags().StringVar(&modelName, "model", "", "Embedding model override")
	cmd.Flags().BoolVar(&noEmbed, "no-embed", false, "Do not embed new concepts (defer to a later 'chiton embed')")
	return cmd
}
