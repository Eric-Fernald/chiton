package main

import (
	"os"
	"path/filepath"
	"slices"
	"testing"
)

// writeModule lays out a tiny module: package a calls b.Do and its own helper, package b calls nothing in the
// module, and a test file in a calls b.Do too (and must be ignored).
func writeModule(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	files := map[string]string{
		"go.mod": "module example.com/m\n\ngo 1.25\n",
		"a/a.go": `package a

import (
	"fmt"
	bee "example.com/m/b"
)

func Run() {
	helper()
	go func() { bee.Do() }()
	fmt.Println()
}

func helper() { bee.Do() }

type T struct{}

func (t *T) Method() {
	helper := func() {}
	helper() // a local, not the package-level helper
	bee.Do()
}
`,
		"a/a_test.go": `package a

import "example.com/m/b"

func callFromTest() { b.Do() }
`,
		"b/b.go":  "package b\n\nfunc Do() {}\n",
		"root.go": "package m\n\nimport \"example.com/m/b\"\n\nfunc Top() { b.Do() }\n",
	}
	for p, body := range files {
		full := filepath.Join(root, filepath.FromSlash(p))
		if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(full, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestAnalyze(t *testing.T) {
	f, err := Analyze(writeModule(t))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := f.Callers("b:Do"), []string{".:Top", "a:Run", "a:T.Method", "a:helper"}; !slices.Equal(got, want) {
		t.Errorf("callers of b:Do = %v, want %v (test files ignored, closures counted)", got, want)
	}
	if got, want := f.Callers("a:helper"), []string{"a:Run"}; !slices.Equal(got, want) {
		t.Errorf("callers of a:helper = %v, want %v (a shadowing local is not a call)", got, want)
	}
	if got, want := f.Callees("a:Run"), []string{"a:helper", "b:Do"}; !slices.Equal(got, want) {
		t.Errorf("callees of a:Run = %v, want %v (fmt is outside the module)", got, want)
	}
	if got, want := f.Importers("b"), []string{".", "a"}; !slices.Equal(got, want) {
		t.Errorf("importers of b = %v, want %v", got, want)
	}
	if got, want := f.ImportsOf("a"), []string{"b"}; !slices.Equal(got, want) {
		t.Errorf("imports of a = %v, want %v", got, want)
	}
}

func TestGrade(t *testing.T) {
	expected := []string{"internal/mcp:New", "cmd/chiton:Server.Run", "internal/graph"}
	cases := []struct {
		name  string
		reply string
		f1    float64
		exact bool
	}{
		{"exact with forgiven formatting", "Here you go:\n```json\n[\"./internal/mcp:New()\", \"cmd\\\\chiton:(*Server).Run\", \"github.com/x/y/internal/graph\"]\n```", 1, true},
		{"one missing", `["internal/mcp:New", "internal/graph"]`, 0.8, false},
		{"one extra", `["internal/mcp:New", "cmd/chiton:Server.Run", "internal/graph", "internal/x:Y"]`, 2 * 0.75 / 1.75, false},
		{"unparseable", "I could not find it.", 0, false},
	}
	for _, c := range cases {
		sc := Grade(c.reply, expected, "github.com/x/y")
		if diff := sc.F1 - c.f1; diff > 1e-9 || diff < -1e-9 || sc.Exact != c.exact {
			t.Errorf("%s: F1 %.3f exact %v, want %.3f %v (missing %v, extra %v)", c.name, sc.F1, sc.Exact, c.f1, c.exact, sc.Missing, sc.Extra)
		}
	}
	if sc := Grade("[]", nil, "m"); sc.F1 != 1 || !sc.Exact {
		t.Errorf("a correct empty answer should score 1, got %+v", sc)
	}
}

func TestGenerateIsDeterministic(t *testing.T) {
	f, err := Analyze(writeModule(t))
	if err != nil {
		t.Fatal(err)
	}
	a, b := Generate(f), Generate(f)
	if len(a) != len(b) {
		t.Fatalf("question count differs between runs: %d vs %d", len(a), len(b))
	}
	for i := range a {
		if a[i].ID != b[i].ID || a[i].Target != b[i].Target {
			t.Errorf("question %d differs between runs: %+v vs %+v", i, a[i], b[i])
		}
	}
}
