package quality

import (
	"context"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// -----------------------------------------------------------------------------
// quality/high-complexity: Python docstrings and Rust lifetimes must not inflate
// the branch count; a genuinely complex function still fires.
// -----------------------------------------------------------------------------

func TestHighComplexity_PythonDocstringFP(t *testing.T) {
	// A simple function whose docstring is full of branch keywords in prose.
	// The triple-quoted docstring is masked, so none of those words count.
	body := "def simple(x):\n" +
		"    \"\"\"\n" +
		strings.Repeat("    if for while case except and if or while elif catch\n", 30) +
		"    \"\"\"\n" +
		"    return x\n"
	lines := strings.Count(body, "\n")
	node := funcNode("simple", 1, lines)
	if g := (HighComplexity{}).MatchFile(context.Background(), target(t, "x.py", body, []model.Node{node})); len(g) != 0 {
		t.Fatalf("expected 0 findings (docstring keywords masked), got %d (%+v)", len(g), g)
	}
}

func TestHighComplexity_RustLifetimeFP(t *testing.T) {
	// Rust lifetimes (`&'a`) used to open fake string state, swallowing the rest
	// of the body so branch keywords were mis-handled. A lifetime-heavy but
	// simple function must not fire.
	body := "fn render<'a>(buf: &'a mut Vec<u8>, label: &'a str) {\n" +
		"    let x: &'a str = label;\n" +
		"    let y: &'a str = label;\n" +
		"    if buf.is_empty() {\n" +
		"        return;\n" +
		"    }\n" +
		"}\n"
	lines := strings.Count(body, "\n")
	node := funcNode("render", 1, lines)
	if g := (HighComplexity{}).MatchFile(context.Background(), target(t, "x.rs", body, []model.Node{node})); len(g) != 0 {
		t.Fatalf("expected 0 findings (lifetime is not a string), got %d (%+v)", len(g), g)
	}
}

func TestHighComplexity_StillFiresTP(t *testing.T) {
	// A truly branch-heavy body still fires (true positive preserved).
	b := "func big() {\n"
	for i := 0; i < 20; i++ {
		b += "\tif a && b || c {\n\t}\n"
	}
	b += "}\n"
	node := funcNode("big", 1, strings.Count(b, "\n"))
	if g := (HighComplexity{}).MatchFile(context.Background(), target(t, "x.go", b, []model.Node{node})); len(g) != 1 {
		t.Fatalf("expected 1 high-complexity finding, got %d (%+v)", len(g), g)
	}
}

// -----------------------------------------------------------------------------
// quality/too-many-returns: returns inside a Python docstring must not count.
// -----------------------------------------------------------------------------

func TestTooManyReturns_DocstringFP(t *testing.T) {
	body := "def f(x):\n" +
		"    \"\"\"\n" +
		strings.Repeat("    return the value to the caller\n", 10) +
		"    \"\"\"\n" +
		"    return x\n"
	node := funcNode("f", 1, strings.Count(body, "\n"))
	if g := (TooManyReturns{}).MatchFile(context.Background(), target(t, "x.py", body, []model.Node{node})); len(g) != 0 {
		t.Fatalf("expected 0 findings (returns in docstring masked), got %d (%+v)", len(g), g)
	}
}

// -----------------------------------------------------------------------------
// quality/nested-ternary: independent sibling ternaries on one line must not
// fire; a genuinely nested ternary still does.
// -----------------------------------------------------------------------------

func TestNestedTernary_SiblingsFP(t *testing.T) {
	fp := `const a = x ? 1 : 2; const b = y ? 3 : 4;
foo(cond1 ? p : q, cond2 ? r : s);
`
	if g := (NestedTernary{}).MatchFile(context.Background(), target(t, "x.ts", fp, nil)); len(g) != 0 {
		t.Fatalf("expected 0 findings on sibling ternaries, got %d (%+v)", len(g), g)
	}
}

func TestNestedTernary_GenuineTP(t *testing.T) {
	tp := `const v = a ? b ? c : d : e;
`
	g := (NestedTernary{}).MatchFile(context.Background(), target(t, "x.ts", tp, nil))
	if len(g) != 1 {
		t.Fatalf("expected 1 nested-ternary finding, got %d (%+v)", len(g), g)
	}
}

// -----------------------------------------------------------------------------
// quality/too-many-params: Go receiver methods and angle-bracket generics must
// now fire (previously missed); Python self/cls is not counted.
// -----------------------------------------------------------------------------

func TestTooManyParams_GoReceiverTP(t *testing.T) {
	src := `package x
func (s *Server) Handle(a int, b int, c int, d int, e int, f int) {}
func (s *Server) Ok(a, b int) {}
`
	g := (TooManyParams{}).MatchFile(context.Background(), target(t, "x.go", src, nil))
	if len(g) != 1 {
		t.Fatalf("expected 1 finding (receiver method), got %d (%+v)", len(g), g)
	}
	if g[0].Span.StartLine != 2 {
		t.Fatalf("expected finding on line 2, got %d", g[0].Span.StartLine)
	}
}

func TestTooManyParams_GenericsTP(t *testing.T) {
	// TS angle-bracket generics and Rust <...> generics must be matched.
	ts := `function build<T, U>(a: T, b: U, c: T, d: U, e: T, f: U) { return a; }
function ok<T>(a: T, b: T) { return a; }
`
	g := (TooManyParams{}).MatchFile(context.Background(), target(t, "x.ts", ts, nil))
	if len(g) != 1 {
		t.Fatalf("expected 1 TS generic finding, got %d (%+v)", len(g), g)
	}

	rs := `fn transform<'a, T>(a: T, b: T, c: T, d: T, e: T, f: T) {}
fn small<T>(a: T) {}
`
	gr := (TooManyParams{}).MatchFile(context.Background(), target(t, "x.rs", rs, nil))
	if len(gr) != 1 {
		t.Fatalf("expected 1 Rust generic finding, got %d (%+v)", len(gr), gr)
	}
}

func TestTooManyParams_PythonSelfNotCounted(t *testing.T) {
	// `self` is the implicit receiver and must not push a 5-param method over
	// the limit; a method with 6 real params (besides self) still fires.
	src := `class C:
    def ok(self, a, b, c, d, e):
        return a
    def bad(self, a, b, c, d, e, f):
        return a
`
	g := (TooManyParams{}).MatchFile(context.Background(), target(t, "x.py", src, nil))
	if len(g) != 1 {
		t.Fatalf("expected 1 finding (self excluded), got %d (%+v)", len(g), g)
	}
	if g[0].Span.StartLine != 4 {
		t.Fatalf("expected finding on line 4, got %d", g[0].Span.StartLine)
	}
}

// -----------------------------------------------------------------------------
// Masking helper: directly exercise the literal/comment/lifetime masking.
// -----------------------------------------------------------------------------

func TestQMaskLiteralsAndComments(t *testing.T) {
	in := "a := \"if for while\"\n" +
		"// if for while\n" +
		"b := `case catch`\n" +
		"x := &'a y\n" // Rust lifetime: must NOT swallow the rest of the file
	out := string(qMaskLiteralsAndComments([]byte(in)))
	if strings.Contains(out, "if for while") {
		t.Fatalf("expected string/comment contents masked, got:\n%s", out)
	}
	if strings.Contains(out, "case catch") {
		t.Fatalf("expected backtick contents masked, got:\n%s", out)
	}
	// The `y` after the lifetime must survive — proving the lone single quote
	// did not open a string that swallowed the rest of the line.
	if !strings.Contains(out, "x := &'a y") {
		t.Fatalf("lifetime line should be untouched, got:\n%s", out)
	}
	// Newlines preserved so line numbers stay accurate.
	if strings.Count(out, "\n") != strings.Count(in, "\n") {
		t.Fatalf("newline count changed: in=%d out=%d", strings.Count(in, "\n"), strings.Count(out, "\n"))
	}
}
