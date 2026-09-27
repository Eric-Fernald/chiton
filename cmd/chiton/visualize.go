package main

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/report"
)

// visualizeCmd renders the graph as interactive HTML or Mermaid. Unlike `chiton report` it works on a
// bare graph; a prior `chiton cluster` only adds community colours and betweenness sizing.
func visualizeCmd() *cobra.Command {
	var format, outFile, scope, title string
	var maxNodes int
	var toStdout bool

	cmd := &cobra.Command{
		Use:     "visualize",
		Aliases: []string{"viz"},
		Short:   "Export the graph as interactive HTML or a Mermaid diagram",
		Long: `Render the graph for humans: a self-contained interactive HTML page
(vis-network; pan/zoom/search, coloured by community, sized by degree) or a
Mermaid flowchart for embedding in Markdown docs.

A prior 'chiton cluster' is optional — it adds community colouring and
betweenness-based sizing. Use --scope to restrict to a subtree and --max-nodes
to cap dense graphs (Mermaid is additionally capped internally).`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			format = strings.ToLower(format)
			if format != "html" && format != "mermaid" && format != "mmd" {
				return fmt.Errorf("unknown --format %q (want: html, mermaid)", format)
			}

			cfg, gs, err := loadStore(true)
			if err != nil {
				return err
			}
			defer gs.Close()

			nodes, edges, err := loadGraph(cmd.Context(), gs)
			if err != nil {
				return err
			}
			if len(nodes) == 0 {
				return fmt.Errorf("graph is empty — run 'chiton build' first")
			}

			nodes, edges = scopeGraph(nodes, edges, scope)
			if len(nodes) == 0 {
				return fmt.Errorf("no nodes match --scope %q", scope)
			}

			// Cluster overlay is optional; a missing sidecar just means no colours.
			overlay, err := cluster.LoadSidecar(cluster.SidecarPath(cfg.GraphPath))
			if err != nil {
				fmt.Fprintf(cmd.ErrOrStderr(), "visualize: warning: could not load cluster sidecar: %v\n", err)
			}

			opts := report.VisualOptions{MaxNodes: maxNodes, Title: title}

			// Resolve the output sink and a default filename per format.
			var w io.Writer
			if toStdout {
				w = cmd.OutOrStdout()
			} else {
				if outFile == "" {
					if format == "html" {
						outFile = "graph.html"
					} else {
						outFile = "graph.mmd"
					}
				}
				f, err := os.Create(outFile)
				if err != nil {
					return fmt.Errorf("create %s: %w", outFile, err)
				}
				defer f.Close()
				w = f
			}

			switch format {
			case "html":
				err = report.GenerateHTML(w, nodes, edges, overlay, opts)
			default: // mermaid / mmd
				err = report.GenerateMermaid(w, nodes, edges, overlay, opts)
			}
			if err != nil {
				return fmt.Errorf("generate %s: %w", format, err)
			}

			if !toStdout {
				fmt.Fprintf(cmd.OutOrStdout(), "visualize: wrote %s (%d nodes, %d edges in scope)\n", outFile, len(nodes), len(edges))
			}
			return nil
		},
	}

	cmd.Flags().StringVar(&format, "format", "html", "Output format: html | mermaid")
	cmd.Flags().StringVar(&outFile, "out", "", "Output file (default: graph.html / graph.mmd)")
	cmd.Flags().BoolVar(&toStdout, "stdout", false, "Write to stdout instead of a file")
	cmd.Flags().StringVar(&scope, "scope", "", "Only include nodes whose source_uri contains this substring")
	cmd.Flags().IntVar(&maxNodes, "max-nodes", report.DefaultVisual.MaxNodes, "Cap rendered nodes (highest-degree kept); 0 = no cap")
	cmd.Flags().StringVar(&title, "title", report.DefaultVisual.Title, "Title shown in the export")
	return cmd
}

// scopeGraph keeps nodes whose source_uri contains scope (case-insensitive) and edges whose endpoints
// are both kept nodes or keepIDs. An empty scope returns the inputs unchanged.
func scopeGraph(nodes []model.Node, edges []model.Edge, scope string, keepIDs ...string) ([]model.Node, []model.Edge) {
	if scope == "" {
		return nodes, edges
	}
	lower := strings.ToLower(scope)
	kept := make(map[string]bool, len(nodes)+len(keepIDs))
	for _, id := range keepIDs {
		kept[id] = true
	}
	var fn []model.Node
	for _, n := range nodes {
		if strings.Contains(strings.ToLower(n.SourceURI), lower) {
			fn = append(fn, n)
			kept[n.ID] = true
		}
	}
	var fe []model.Edge
	for _, e := range edges {
		if kept[e.Src] && kept[e.Dst] {
			fe = append(fe, e)
		}
	}
	return fn, fe
}
