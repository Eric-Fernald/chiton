package analyze

import (
	"context"
	"fmt"
	"os"

	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// AnalyzeFile runs the deterministic FileRules over a single file and persists
// the resulting findings, pruning any prior deterministic findings for that
// file that no longer fire. It is the incremental counterpart of Analyze used
// by the live-sync engine (Phase 3) when one file changes.
//
// GraphScope rules (architecture-level: circular-import, god-module,
// dead-symbol) are intentionally NOT run here — they need the whole corpus and
// are refreshed by a full `chiton analyze`. AnalyzeFile only re-evaluates the
// file-local rules whose inputs actually changed, which is the cheap, common
// case on a single-file edit.
//
// A path with no extractor (or that fails to read/extract) yields an empty
// Result and no error, so callers can drive it from a file watcher without
// special-casing unsupported files.
func (a *Analyzer) AnalyzeFile(ctx context.Context, path string) (*Result, error) {
	start := a.now()
	ex := extast.ForFile(path)
	if ex == nil {
		return &Result{}, nil
	}
	content, err := os.ReadFile(path)
	if err != nil {
		return &Result{}, nil // unreadable (e.g. removed mid-flight) — nothing to do
	}
	nodes, edges, err := ex.ExtractFile(ctx, path)
	if err != nil {
		return &Result{}, nil // parse error — leave existing findings untouched
	}

	fileNode := findFileNode(nodes)
	fileRules := a.fileRules()

	ext := fileExt(path)
	var findings []Finding
	ft := FileTarget{
		Path:     path,
		URI:      fileNode.SourceURI,
		Content:  content,
		FileNode: fileNode,
		Nodes:    nonFileNodes(nodes),
		Edges:    edges,
	}
	var usedRules []Rule
	for _, fr := range fileRules {
		if !ruleApplies(fr.Meta().Languages, ext) {
			continue
		}
		usedRules = append(usedRules, fr)
		findings = append(findings, fr.MatchFile(ctx, ft)...)
	}

	if err := a.persist(ctx, usedRules, findings); err != nil {
		return nil, err
	}
	scanned := map[string]bool{}
	if fileNode.SourceURI != "" {
		scanned[fileNode.SourceURI] = true
	}
	// Only prune findings from the FileRules we actually ran. GraphScope rules
	// (architecture/dead-symbol) are deliberately skipped here, so their findings
	// — though DETERMINISTIC and targeting this same file — must survive a single-
	// file edit rather than be wiped until the next full `chiton analyze`.
	ruleScope := make(map[string]bool, len(usedRules))
	for _, r := range usedRules {
		ruleScope[r.Meta().ID] = true
	}
	if err := a.pruneResolved(ctx, scanned, findings, ruleScope); err != nil {
		return nil, err
	}

	return &Result{
		FilesScanned: 1,
		RulesRun:     len(usedRules),
		Findings:     findings,
		Duration:     a.now().Sub(start),
	}, nil
}

// PruneFileURI removes every finding attached to the given file source_uri —
// BOTH deterministic and LLM findings. The live-sync engine calls this when a
// file is deleted so no findings linger after the code they flagged is gone.
// (DeleteBySourceURI cannot do this: a finding node's source_uri is
// "finding://<id>", not the file URI, so findings are only reachable via their
// target_uri attribute.) Unlike the deterministic-only pruneResolved used on an
// edit re-scan, a delete resolves the file's LLM findings too, since live-sync
// never re-runs the LLM rules to supersede them.
func (a *Analyzer) PruneFileURI(ctx context.Context, uri string) (int, error) {
	if uri == "" {
		return 0, nil
	}
	all, err := a.store.AllNodes(ctx)
	if err != nil {
		return 0, fmt.Errorf("load nodes for prune: %w", err)
	}
	var stale []model.Node
	for _, n := range all {
		if n.Kind != model.KindFinding {
			continue
		}
		if turi, _ := n.Attrs["target_uri"].(string); turi == uri {
			stale = append(stale, n)
		}
	}
	if err := deleteFindingNodes(ctx, a.store, stale); err != nil {
		return 0, err
	}
	return len(stale), nil
}

// fileRules returns the registry's FileRules.
func (a *Analyzer) fileRules() []FileRule {
	var out []FileRule
	for _, r := range a.registry.All() {
		if fr, ok := r.(FileRule); ok {
			out = append(out, fr)
		}
	}
	return out
}
