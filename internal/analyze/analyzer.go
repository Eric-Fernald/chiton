package analyze

import (
	"context"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// Analyzer runs registered rules against a directory tree and persists the
// resulting findings (and rule metadata) as graph nodes/edges.
type Analyzer struct {
	store    store.GraphStore
	registry *Registry
	now      func() time.Time
}

// New returns an Analyzer wired to the given store and rule registry.
// Pass nil for registry to use the package-default registry.
func New(s store.GraphStore, reg *Registry) *Analyzer {
	if reg == nil {
		reg = DefaultRegistry()
	}
	return &Analyzer{
		store:    s,
		registry: reg,
		now:      time.Now,
	}
}

// Result summarises one analyze run.
type Result struct {
	FilesScanned int
	RulesRun     int
	Findings     []Finding
	Duration     time.Duration
}

// Analyze walks root, extracts each supported source file, runs FileScope
// rules over each file, then runs GraphScope rules over the full extracted
// set. All findings (and the rules they violated) are upserted into the store.
func (a *Analyzer) Analyze(ctx context.Context, root string) (*Result, error) {
	start := a.now()
	rules := a.registry.All()

	var fileRules []FileRule
	var graphRules []GraphRule
	for _, r := range rules {
		switch v := r.(type) {
		case FileRule:
			fileRules = append(fileRules, v)
		case GraphRule:
			graphRules = append(graphRules, v)
		}
	}

	var (
		allNodes    []model.Node
		allEdges    []model.Edge
		findings    []Finding
		filesScanned int
	)

	walkErr := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skip common noise directories
			base := d.Name()
			if base == ".git" || base == "node_modules" || base == "vendor" || base == ".chiton" || base == "chiton-out" {
				return filepath.SkipDir
			}
			return nil
		}
		ex := extast.ForFile(path)
		if ex == nil {
			return nil
		}
		content, err := os.ReadFile(path)
		if err != nil {
			return nil // skip unreadable
		}
		nodes, edges, err := ex.ExtractFile(ctx, path)
		if err != nil {
			return nil
		}
		filesScanned++

		fileNode := findFileNode(nodes)
		ft := FileTarget{
			Path:     path,
			URI:      fileNode.SourceURI,
			Content:  content,
			FileNode: fileNode,
			Nodes:    nonFileNodes(nodes),
			Edges:    edges,
		}

		ext := fileExt(path)
		for _, fr := range fileRules {
			if !ruleApplies(fr.Meta().Languages, ext) {
				continue
			}
			findings = append(findings, fr.MatchFile(ctx, ft)...)
		}

		allNodes = append(allNodes, nodes...)
		allEdges = append(allEdges, edges...)
		return nil
	})
	if walkErr != nil {
		return nil, fmt.Errorf("walk: %w", walkErr)
	}

	gt := GraphTarget{Nodes: allNodes, Edges: allEdges}
	for _, gr := range graphRules {
		findings = append(findings, gr.MatchGraph(ctx, gt)...)
	}

	if err := a.persist(ctx, rules, findings); err != nil {
		return nil, err
	}

	return &Result{
		FilesScanned: filesScanned,
		RulesRun:     len(rules),
		Findings:     findings,
		Duration:     a.now().Sub(start),
	}, nil
}

// persist upserts rule nodes (so VIOLATES edges resolve), finding nodes,
// and VIOLATES/AFFECTS edges.
func (a *Analyzer) persist(ctx context.Context, rules []Rule, findings []Finding) error {
	now := a.now()

	nodes := make([]model.Node, 0, len(rules)+len(findings))
	for _, r := range rules {
		nodes = append(nodes, RuleAsNode(r.Meta(), now))
	}
	for _, f := range findings {
		nodes = append(nodes, f.AsNode(now))
	}
	if err := a.store.UpsertNodes(ctx, nodes); err != nil {
		return fmt.Errorf("upsert nodes: %w", err)
	}

	edges := make([]model.Edge, 0, len(findings)*2)
	for _, f := range findings {
		edges = append(edges, f.ViolatesEdge(now))
		if ae := f.AffectsEdge(now); ae.ID != "" {
			edges = append(edges, ae)
		}
	}
	if err := a.store.UpsertEdges(ctx, edges); err != nil {
		return fmt.Errorf("upsert edges: %w", err)
	}
	return nil
}

func findFileNode(nodes []model.Node) model.Node {
	for _, n := range nodes {
		if n.Kind == model.KindFile {
			return n
		}
	}
	return model.Node{}
}

func nonFileNodes(nodes []model.Node) []model.Node {
	out := make([]model.Node, 0, len(nodes))
	for _, n := range nodes {
		if n.Kind != model.KindFile {
			out = append(out, n)
		}
	}
	return out
}

// fileExt returns the lowercase extension without the leading dot.
func fileExt(path string) string {
	e := strings.ToLower(filepath.Ext(path))
	if strings.HasPrefix(e, ".") {
		return e[1:]
	}
	return e
}

// ruleApplies returns true if the rule's language filter accepts the file.
// nil/empty filter accepts everything.
func ruleApplies(langs []string, ext string) bool {
	if len(langs) == 0 {
		return true
	}
	for _, l := range langs {
		if strings.EqualFold(l, ext) {
			return true
		}
	}
	return false
}
