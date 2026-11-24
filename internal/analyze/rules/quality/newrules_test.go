package quality

import (
	"context"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func funcNode(id string, start, end int) model.Node {
	return model.Node{
		ID:    id,
		Kind:  model.KindSymbol,
		Label: id,
		Span:  &model.SourceSpan{StartLine: start, EndLine: end},
		Attrs: map[string]any{"lang_kind": "func"},
	}
}

func TestTooManyParams(t *testing.T) {
	src := `package x

func fine(a, b, c int) {}
func tooMany(a int, b int, c int, d int, e int, f int) {}
func goNamed(a, b, c, d, e, f int) {}

// a call site that happens to have many args must NOT fire
result := compute(1, 2, 3, 4, 5, 6, 7)
`
	got := (TooManyParams{}).MatchFile(context.Background(), target(t, "x.go", src, nil))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings (tooMany + goNamed), got %d (%+v)", len(got), got)
	}

	js := `function ok(a, b) { return a; }
function bad(a, b, c, d, e, f) { return a; }
const arrowDecl = (a, b, c, d, e, f, g) => a;
foo(1, 2, 3, 4, 5, 6, 7);
`
	gotJS := (TooManyParams{}).MatchFile(context.Background(), target(t, "x.js", js, nil))
	if len(gotJS) != 1 {
		t.Fatalf("expected 1 JS finding (named decl only), got %d (%+v)", len(gotJS), gotJS)
	}

	clean := `func a(x, y int) {}
def f(a, b, c):
    return a
`
	if g := (TooManyParams{}).MatchFile(context.Background(), target(t, "x.go", clean, nil)); len(g) != 0 {
		t.Fatalf("expected 0 findings on clean source, got %d (%+v)", len(g), g)
	}
}

func TestPythonSemicolons(t *testing.T) {
	src := `x = 1; y = 2
z = 3
a = "has ; inside string"
b = 4  # trailing comment ; not code
c = 5;
do(); other()
`
	got := (PythonSemicolons{}).MatchFile(context.Background(), target(t, "x.py", src, nil))
	// fires on line 1 (x=1; y=2) and line 6 (do(); other())
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}
}

func TestNestedTernary(t *testing.T) {
	src := `const a = cond ? x : y;
const b = c1 ? (c2 ? p : q) : r;
const d = obj?.field ?? fallback;
const label = "a ? b : c ? d : e";
const e = a ? b : c;
`
	got := (NestedTernary{}).MatchFile(context.Background(), target(t, "x.ts", src, nil))
	if len(got) != 1 {
		t.Fatalf("expected 1 nested-ternary finding, got %d (%+v)", len(got), got)
	}
	if got[0].Span.StartLine != 2 {
		t.Fatalf("expected finding on line 2, got line %d", got[0].Span.StartLine)
	}
}

func TestHighComplexity(t *testing.T) {
	// Build a function body with many branches.
	body := "func big() {\n"
	for i := 0; i < 20; i++ {
		body += "\tif a && b || c {\n\t}\n"
	}
	body += "}\n"
	node := funcNode("big", 1, 1+20*2+1)

	got := (HighComplexity{}).MatchFile(context.Background(), target(t, "x.go", body, []model.Node{node}))
	if len(got) != 1 {
		t.Fatalf("expected 1 high-complexity finding, got %d (%+v)", len(got), got)
	}

	simple := "func small() {\n\tif a {\n\t}\n\treturn 1\n}\n"
	snode := funcNode("small", 1, 5)
	if g := (HighComplexity{}).MatchFile(context.Background(), target(t, "x.go", simple, []model.Node{snode})); len(g) != 0 {
		t.Fatalf("expected 0 findings on simple function, got %d (%+v)", len(g), g)
	}

	// Comments containing keywords must not inflate complexity.
	commented := "func c() {\n\t// if for while case catch && ||\n\treturn 1\n}\n"
	cnode := funcNode("c", 1, 4)
	if g := (HighComplexity{}).MatchFile(context.Background(), target(t, "x.go", commented, []model.Node{cnode})); len(g) != 0 {
		t.Fatalf("expected 0 findings when keywords are in comments, got %d (%+v)", len(g), g)
	}
}

func TestTooManyReturns(t *testing.T) {
	body := "func many() {\n"
	for i := 0; i < 7; i++ {
		body += "\treturn x\n"
	}
	body += "}\n"
	node := funcNode("many", 1, 1+7+1)

	got := (TooManyReturns{}).MatchFile(context.Background(), target(t, "x.go", body, []model.Node{node}))
	if len(got) != 1 {
		t.Fatalf("expected 1 too-many-returns finding, got %d (%+v)", len(got), got)
	}

	few := "func ok() {\n\tif a {\n\t\treturn 1\n\t}\n\treturn 2\n}\n"
	fnode := funcNode("ok", 1, 6)
	if g := (TooManyReturns{}).MatchFile(context.Background(), target(t, "x.go", few, []model.Node{fnode})); len(g) != 0 {
		t.Fatalf("expected 0 findings with few returns, got %d (%+v)", len(g), g)
	}
}

func TestEmptyBlock(t *testing.T) {
	src := `package x

func f() {
	if cond {
	}
	for i := 0; i < 10; i++ {}
	if other { doThing() }
}

type S struct{}
var m = map[string]int{}
`
	got := (EmptyBlock{}).MatchFile(context.Background(), target(t, "x.go", src, nil))
	// fires on `if cond {` ... `}`? No: `if cond {` then `}` on next line is multi-line, our
	// rule is single-line. So only `for i := 0; i < 10; i++ {}` fires.
	if len(got) != 1 {
		t.Fatalf("expected 1 empty-block finding, got %d (%+v)", len(got), got)
	}
	if got[0].Span.StartLine != 6 {
		t.Fatalf("expected finding on line 6, got line %d", got[0].Span.StartLine)
	}

	js := `if (x) {}
while (y) {  }
function f() {}
const o = {};
if (a) { run(); }
`
	gotJS := (EmptyBlock{}).MatchFile(context.Background(), target(t, "x.js", js, nil))
	if len(gotJS) != 2 {
		t.Fatalf("expected 2 JS empty-block findings, got %d (%+v)", len(gotJS), gotJS)
	}
}
