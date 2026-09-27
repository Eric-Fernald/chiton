package main

import (
	"fmt"
	"os"
	"path/filepath"

	"github.com/spf13/cobra"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/report"
)

// dashboardCmd groups the static dashboard commands. The dashboard is a read-only
// artifact generated from the graph + findings, deliberately not a server.
func dashboardCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "dashboard",
		Short: "Generate a read-only static web dashboard from the graph + findings",
		Long: `Generate a self-contained, read-only web dashboard: an interactive graph
view (vis-network) plus a findings explorer that filters by severity, category,
community, file, and provenance. The output is static files — there is no
server. Open the generated index.html directly in a browser.`,
	}
	cmd.AddCommand(dashboardBuildCmd())
	return cmd
}

func dashboardBuildCmd() *cobra.Command {
	var outDir, scope, title string
	var maxNodes int
	var includeAmbiguous bool

	cmd := &cobra.Command{
		Use:   "build",
		Short: "Build the static dashboard into a directory",
		Long: `Render the graph and analyze findings into a static dashboard directory
(default: ./chiton-dashboard). The graph view is enriched with community
colours and betweenness sizing when a prior 'chiton cluster' has run; findings
come from the most recent 'chiton analyze'. Use --scope to restrict to a
subtree and --max-nodes to cap dense graphs.`,
		RunE: func(cmd *cobra.Command, _ []string) error {
			cfg, gs, err := loadStore(true)
			if err != nil {
				return err
			}
			defer gs.Close()

			ctx := cmd.Context()
			nodes, edges, err := loadGraph(ctx, gs)
			if err != nil {
				return err
			}
			if len(nodes) == 0 {
				return fmt.Errorf("graph is empty — run 'chiton build' first")
			}

			// Cluster overlay is optional; a missing sidecar just means no
			// community colours and community "none" on every finding.
			overlay, err := cluster.LoadSidecar(cluster.SidecarPath(cfg.GraphPath))
			if err != nil {
				fmt.Fprintf(cmd.ErrOrStderr(), "dashboard: warning: could not load cluster sidecar: %v\n", err)
			}

			// All findings (the explorer filters client-side), but AMBIGUOUS ones only on request
			// so low-confidence LLM findings don't masquerade as confirmed.
			views, err := analyze.List(ctx, gs, analyze.ListOptions{
				Scope:            scope,
				IncludeAmbiguous: includeAmbiguous,
			})
			if err != nil {
				return fmt.Errorf("load findings: %w", err)
			}

			scopedNodes, scopedEdges := scopeGraph(nodes, edges, scope)
			if len(scopedNodes) == 0 {
				return fmt.Errorf("no nodes match --scope %q", scope)
			}

			findings := buildDashboardFindings(views, nodes, overlay)

			data := report.DashboardData{
				Title:    title,
				Nodes:    scopedNodes,
				Edges:    scopedEdges,
				Overlay:  overlay,
				Findings: findings,
				MaxNodes: maxNodes,
			}

			if err := os.MkdirAll(outDir, 0o755); err != nil {
				return fmt.Errorf("create %s: %w", outDir, err)
			}
			indexPath := filepath.Join(outDir, "index.html")
			f, err := os.Create(indexPath)
			if err != nil {
				return fmt.Errorf("create %s: %w", indexPath, err)
			}
			defer f.Close()

			if err := report.GenerateDashboard(f, data); err != nil {
				return fmt.Errorf("generate dashboard: %w", err)
			}

			fmt.Fprintf(cmd.OutOrStdout(),
				"dashboard: wrote %s (%d nodes, %d edges in scope, %d findings)\n",
				indexPath, len(scopedNodes), len(scopedEdges), len(findings))
			return nil
		},
	}

	cmd.Flags().StringVar(&outDir, "out", "chiton-dashboard", "Output directory for the static dashboard")
	cmd.Flags().StringVar(&scope, "scope", "", "Only include nodes/findings whose source_uri contains this substring")
	cmd.Flags().IntVar(&maxNodes, "max-nodes", report.DefaultVisual.MaxNodes, "Cap rendered graph nodes (highest-degree kept); 0 = no cap")
	cmd.Flags().StringVar(&title, "title", "chiton dashboard", "Title shown in the dashboard")
	cmd.Flags().BoolVar(&includeAmbiguous, "include-ambiguous", false, "Include low-confidence AMBIGUOUS LLM findings")
	return cmd
}

// buildDashboardFindings projects findings into the report shape, taking each one's community from its
// target_node_id, else from a node sharing its target_uri (file node preferred), else -1.
func buildDashboardFindings(views []analyze.FindingView, allNodes []model.Node, overlay *cluster.Results) []report.DashboardFinding {
	nodeComm := func(string) int { return -1 }
	if overlay != nil {
		nodeComm = func(id string) int {
			if r, ok := overlay.Nodes[id]; ok {
				return r.CommunityID
			}
			return -1
		}
	}

	// findingTarget: finding node id -> affected node id (target_node_id attr).
	// uriComm: source_uri -> representative community (file node wins).
	findingTarget := make(map[string]string)
	uriComm := make(map[string]int)
	for _, n := range allNodes {
		switch n.Kind {
		case model.KindFinding:
			if tn := model.AttrString(n.Attrs, "target_node_id"); tn != "" {
				findingTarget[n.ID] = tn
			}
		default:
			if overlay == nil {
				continue
			}
			c := nodeComm(n.ID)
			if c < 0 {
				continue
			}
			if _, seen := uriComm[n.SourceURI]; !seen || n.Kind == model.KindFile {
				uriComm[n.SourceURI] = c
			}
		}
	}

	out := make([]report.DashboardFinding, 0, len(views))
	for _, v := range views {
		targetNode := findingTarget[v.ID]
		comm := -1
		if targetNode != "" {
			comm = nodeComm(targetNode)
		}
		if comm < 0 {
			if c, ok := uriComm[v.TargetURI]; ok {
				comm = c
			}
		}
		line := 0
		if v.Span != nil {
			line = v.Span.StartLine
		}
		out = append(out, report.DashboardFinding{
			ID:         v.ID,
			RuleID:     v.RuleID,
			Category:   v.Category,
			Severity:   v.Severity,
			Message:    v.Message,
			TargetURI:  v.TargetURI,
			Line:       line,
			Fix:        v.Fix,
			Provenance: v.Provenance,
			Community:  comm,
			NodeID:     targetNode,
		})
	}
	return out
}
