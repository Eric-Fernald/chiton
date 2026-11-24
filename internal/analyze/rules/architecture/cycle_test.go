package architecture

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
)

// extractAll runs the real extractors over in-memory sources rooted at dir, keyed by slash path.
func extractAll(t *testing.T, dir string, srcs [][2]string) analyze.GraphTarget {
	t.Helper()
	var tg analyze.GraphTarget
	for _, s := range srcs {
		p := filepath.Join(dir, filepath.FromSlash(s[0]))
		nodes, edges, err := extast.ForFile(p).ExtractSource(context.Background(), p, []byte(s[1]))
		if err != nil {
			t.Fatalf("extract %s: %v", s[0], err)
		}
		tg.Nodes = append(tg.Nodes, nodes...)
		tg.Edges = append(tg.Edges, edges...)
	}
	return tg
}

// TestCircularImport_ResolvesRelativeImports checks cycles built from extracted relative imports, whose
// IMPORTS edges all land on external package nodes: two TypeScript packages that import each other through
// index files, and two Python modules in one package. A file that only imports into a cycle is not on it.
func TestCircularImport_ResolvesRelativeImports(t *testing.T) {
	tg := extractAll(t, t.TempDir(), [][2]string{
		{"web/a/index.ts", "import { b } from '../b'\nexport const a = 1\n"},
		{"web/b/index.ts", "import { a } from '../a'\nexport const b = 2\n"},
		{"web/c/index.ts", "import { a } from '../a'\n"},
		{"pkg/x.py", "from .y import g\n"},
		{"pkg/y.py", "from .x import f\n"},
		{"pkg/z.py", "import os\n"},
	})
	got := (CircularImport{}).MatchGraph(context.Background(), tg)
	if len(got) != 2 {
		t.Fatalf("want 2 cycles (web/a <-> web/b, pkg/x <-> pkg/y), got %d: %+v", len(got), got)
	}
	var targets []string
	for _, f := range got {
		targets = append(targets, f.TargetURI)
	}
	joined := strings.Join(targets, " ")
	if !strings.Contains(joined, "/web/a/index.ts") && !strings.Contains(joined, "/web/b/index.ts") {
		t.Errorf("no TypeScript cycle reported: %v", targets)
	}
	if !strings.Contains(joined, "/pkg/x.py") && !strings.Contains(joined, "/pkg/y.py") {
		t.Errorf("no Python cycle reported: %v", targets)
	}
	if strings.Contains(joined, "/web/c/") || strings.Contains(joined, "/pkg/z.py") {
		t.Errorf("a file off the cycle was reported: %v", targets)
	}
}
