package mcpsrv

import (
	"context"
	"encoding/json"
	"os"
	"path/filepath"
	"slices"
	"testing"

	extast "github.com/Eric-Fernald/chiton/internal/extract/ast"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// indexModule extracts a small Go module with the real extractor into gs:
//
//	app.Main → svc.Start → (sibling file) svc.helper → store.Open
//	app.Main → store.Open
//	app/other.go has its own New, so "New" is ambiguous.
func indexModule(t *testing.T, gs store.GraphStore) {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"go.mod":         "module example.com/m\n\ngo 1.25\n",
		"store/store.go": "package store\n\nfunc Open() {}\n\nfunc New() {}\n",
		"svc/svc.go":     "package svc\n\nfunc Start() { helper() }\n",
		"svc/helper.go":  "package svc\n\nimport \"example.com/m/store\"\n\nfunc helper() { store.Open() }\n",
		"app/app.go":     "package app\n\nimport (\n\t\"example.com/m/store\"\n\t\"example.com/m/svc\"\n)\n\nfunc Main() {\n\tsvc.Start()\n\tstore.Open()\n}\n",
		"app/other.go":   "package app\n\nfunc New() {}\n",
		// A test-only import: svc's tests depend on app, but svc itself does not.
		"svc/svc_test.go": "package svc\n\nimport \"example.com/m/app\"\n\nfunc testUsesApp() { app.Main() }\n",
	}
	ctx := context.Background()
	ex := &extast.GoExtractor{}
	for rel, body := range files {
		p := filepath.Join(root, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	for rel := range files {
		if filepath.Ext(rel) != ".go" {
			continue
		}
		nodes, edges, err := ex.ExtractFile(ctx, filepath.Join(root, filepath.FromSlash(rel)))
		if err != nil {
			t.Fatal(err)
		}
		if err := gs.UpsertNodes(ctx, nodes); err != nil {
			t.Fatal(err)
		}
		if err := gs.UpsertEdges(ctx, edges); err != nil {
			t.Fatal(err)
		}
	}
}

func neighborLabels(t *testing.T, out string) []string {
	t.Helper()
	var got struct {
		Seed  leanNode   `json:"seed"`
		Nodes []leanNode `json:"nodes"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	var labels []string
	for _, n := range got.Nodes {
		if n.Label != got.Seed.Label {
			labels = append(labels, n.Label)
		}
	}
	slices.Sort(labels)
	return labels
}

func TestNeighborhoodResolvesCallersAcrossFilesAndPackages(t *testing.T) {
	gs := openStore(t)
	indexModule(t, gs)
	h := neighborhoodHandler(gs, redactor{})

	// store.Open is called from another package (app, svc), through a pkg:// reference.
	out := call(t, h, map[string]any{"label": "store:Open", "direction": "in", "edge_kinds": []any{"CALLS"}, "depth": 1})
	if got, want := neighborLabels(t, out), []string{"Main", "helper"}; !slices.Equal(got, want) {
		t.Errorf("direct callers of store.Open = %v, want %v\n%s", got, want, out)
	}

	// Three hops: Start reaches Open through helper in a sibling file of svc, and a test function reaches it
	// through Main.
	out = call(t, h, map[string]any{"label": "store:Open", "direction": "in", "edge_kinds": []any{"CALLS"}, "depth": 3})
	if got, want := neighborLabels(t, out), []string{"Main", "Start", "helper", "testUsesApp"}; !slices.Equal(got, want) {
		t.Errorf("callers of store.Open within 3 calls = %v, want %v\n%s", got, want, out)
	}
	out = call(t, h, map[string]any{"label": "store:Open", "direction": "in", "edge_kinds": []any{"CALLS"}, "depth": 3, "exclude_tests": true})
	if got, want := neighborLabels(t, out), []string{"Main", "Start", "helper"}; !slices.Equal(got, want) {
		t.Errorf("callers of store.Open within 3 calls, tests excluded = %v, want %v\n%s", got, want, out)
	}

	// direction out from Main: what it calls, resolved to the definitions.
	out = call(t, h, map[string]any{"label": "Main", "direction": "out", "edge_kinds": []any{"CALLS"}, "depth": 1})
	if got, want := neighborLabels(t, out), []string{"Open", "Start"}; !slices.Equal(got, want) {
		t.Errorf("callees of Main = %v, want %v\n%s", got, want, out)
	}
}

func TestLookupReportsAmbiguousNames(t *testing.T) {
	gs := openStore(t)
	indexModule(t, gs)
	out := call(t, getNodeHandler(gs, redactor{}), map[string]any{"label": "New"})
	var got struct {
		Label     string `json:"label"`
		SourceURI string `json:"source_uri"`
		Note      string `json:"note"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if got.Label != "New" || got.Note == "" {
		t.Errorf("get_node(New) = %+v; want the label New and a note that another symbol shares it", got)
	}
	out = call(t, getNodeHandler(gs, redactor{}), map[string]any{"label": "app:New"})
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("unmarshal: %v\n%s", err, out)
	}
	if filepath.Base(got.SourceURI) != "other.go" {
		t.Errorf("get_node(app:New) resolved to %s, want app/other.go", got.SourceURI)
	}
}

func TestPackageDeps(t *testing.T) {
	gs := openStore(t)
	indexModule(t, gs)
	h := packageDepsHandler(gs)
	dirs := func(args map[string]any) []string {
		out := call(t, h, args)
		var got struct {
			Packages []struct {
				Dir string `json:"dir"`
			} `json:"packages"`
		}
		if err := json.Unmarshal([]byte(out), &got); err != nil {
			t.Fatalf("unmarshal: %v\n%s", err, out)
		}
		var ds []string
		for _, p := range got.Packages {
			ds = append(ds, p.Dir)
		}
		slices.Sort(ds)
		return ds
	}
	if got, want := dirs(map[string]any{"package": "app"}), []string{"store", "svc"}; !slices.Equal(got, want) {
		t.Errorf("imports of app = %v, want %v", got, want)
	}
	if got, want := dirs(map[string]any{"package": "store", "direction": "importers"}), []string{"app", "svc"}; !slices.Equal(got, want) {
		t.Errorf("importers of store = %v, want %v", got, want)
	}
	// svc imports only store; transitively, app reaches store twice but lists it once.
	if got, want := dirs(map[string]any{"package": "svc", "direction": "importers", "transitive": true}), []string{"app"}; !slices.Equal(got, want) {
		t.Errorf("transitive importers of svc = %v, want %v", got, want)
	}
	// svc_test.go imports app; that counts only with include_tests.
	if got := dirs(map[string]any{"package": "app", "direction": "importers"}); len(got) != 0 {
		t.Errorf("importers of app = %v, want none: only a test file imports it", got)
	}
	if got, want := dirs(map[string]any{"package": "app", "direction": "importers", "include_tests": true}), []string{"svc"}; !slices.Equal(got, want) {
		t.Errorf("importers of app with include_tests = %v, want %v", got, want)
	}
	if got, want := dirs(map[string]any{"package": "svc"}), []string{"store"}; !slices.Equal(got, want) {
		t.Errorf("imports of svc = %v, want %v (its test file's import of app excluded)", got, want)
	}
}
