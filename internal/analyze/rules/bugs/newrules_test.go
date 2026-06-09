package bugs

import (
	"context"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// ntarget builds a FileTarget for the new-rule tests. Named distinctly from the
// existing test helper `target` to avoid a redeclaration in this package.
func ntarget(t *testing.T, path, content string) analyze.FileTarget {
	t.Helper()
	uri := "file://test/" + path
	return analyze.FileTarget{
		Path:    path,
		URI:     uri,
		Content: []byte(content),
		FileNode: model.Node{
			ID:        model.NewNodeID(uri, path),
			Kind:      model.KindFile,
			Label:     path,
			SourceURI: uri,
		},
	}
}

// ---- bugs/loose-equality ----

func TestLooseEquality(t *testing.T) {
	pos := `function f(a, b) {
  if (a == b) return 1;
  if (a != b) return 2;
}
`
	got := (LooseEquality{}).MatchFile(context.Background(), ntarget(t, "x.js", pos))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}

	neg := `function g(a, b) {
  if (a === b) return 1;
  if (a !== b) return 2;
  const h = (x) => x + 1;
  a >= b;
  a <= b;
}
`
	if got := (LooseEquality{}).MatchFile(context.Background(), ntarget(t, "y.ts", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings for strict-equality file, got %d (%+v)", len(got), got)
	}
}

// ---- bugs/assignment-in-condition ----

func TestAssignmentInCondition(t *testing.T) {
	pos := `function f(x, y) {
  if (x = y) { return 1; }
  while (a = next()) { use(a); }
}
`
	got := (AssignmentInCondition{}).MatchFile(context.Background(), ntarget(t, "x.js", pos))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}

	neg := `function g(x, y) {
  if (x == y) { return 1; }
  if (x === y) { return 2; }
  while (x <= y) { x += 1; }
  if (x != y) {}
}
`
	if got := (AssignmentInCondition{}).MatchFile(context.Background(), ntarget(t, "y.java", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// ---- bugs/leftover-debugger ----

func TestLeftoverDebuggerJS(t *testing.T) {
	pos := `function f() {
  debugger;
  return 1;
}
`
	if got := (LeftoverDebugger{}).MatchFile(context.Background(), ntarget(t, "x.js", pos)); len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}

	neg := `const myDebugger = 1; // debugger; in comment
const x = obj.debugger;
`
	if got := (LeftoverDebugger{}).MatchFile(context.Background(), ntarget(t, "y.js", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

func TestLeftoverDebuggerPy(t *testing.T) {
	pos := `def f():
    import pdb; pdb.set_trace()
    breakpoint()
    return 1
`
	if got := (LeftoverDebugger{}).MatchFile(context.Background(), ntarget(t, "x.py", pos)); len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}

	neg := `def g():
    # breakpoint() here once
    return 1
`
	if got := (LeftoverDebugger{}).MatchFile(context.Background(), ntarget(t, "y.py", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// ---- bugs/self-assignment ----

func TestSelfAssignment(t *testing.T) {
	pos := `package x
func f() {
	a = a
	this.count = this.count
}
`
	got := (SelfAssignment{}).MatchFile(context.Background(), ntarget(t, "x.go", pos))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}

	neg := `package x
func g() {
	a = b
	a := a + 1
	if a == a { }
	x.y = y.x
}
`
	if got := (SelfAssignment{}).MatchFile(context.Background(), ntarget(t, "y.go", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// ---- bugs/unchecked-type-assertion ----

func TestUncheckedTypeAssertion(t *testing.T) {
	pos := `package x
func f(i interface{}) {
	v := i.(string)
	w = i.(*Foo)
	_ = v
	_ = w
}
`
	got := (UncheckedTypeAssertion{}).MatchFile(context.Background(), ntarget(t, "x.go", pos))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}

	neg := `package x
func g(i interface{}) {
	v, ok := i.(string)
	switch t := i.(type) {
	case int:
		_ = t
	}
	_ = v
	_ = ok
}
`
	if got := (UncheckedTypeAssertion{}).MatchFile(context.Background(), ntarget(t, "y.go", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// ---- bugs/python-is-literal ----

func TestPythonIsLiteral(t *testing.T) {
	pos := `def f(x, n):
    if x is "foo":
        return 1
    if n is 5:
        return 2
    if n is not 0:
        return 3
`
	got := (PythonIsLiteral{}).MatchFile(context.Background(), ntarget(t, "x.py", pos))
	if len(got) != 3 {
		t.Fatalf("expected 3 findings, got %d (%+v)", len(got), got)
	}

	neg := `def g(x, axis):
    if x is None:
        return 1
    if x is not None:
        return 2
    if x == "foo":
        return 3
    return axis
`
	if got := (PythonIsLiteral{}).MatchFile(context.Background(), ntarget(t, "y.py", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// ---- bugs/loose-equality: literal/comment masking (FP regression) ----

func TestLooseEquality_LiteralsNotFlagged(t *testing.T) {
	// FP inputs: == / != that live inside string, template, and regex literals
	// (plus comments) must NOT be flagged.
	neg := "const a = \"YQ==\";\n" + // base64 padding inside a double-quoted string
		"const sql = `SELECT * FROM t WHERE a == b`;\n" + // template literal
		"const sql2 = 'WHERE x != y';\n" + // single-quoted SQL fragment
		"const re = /a==b/;\n" + // regex literal containing ==
		"const re2 = /x!=y/g;\n" + // regex with flags
		"// note: prefer == over != here\n" + // line comment
		"/* a == b and a != b inside block */\n"
	if got := (LooseEquality{}).MatchFile(context.Background(), ntarget(t, "lit.ts", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings for literal/comment ==, got %d (%+v)", len(got), got)
	}

	// TP still fires: genuine code-level loose equality.
	pos := "const a = \"YQ==\";\n" + // string (ignored)
		"if (x == y) doThing();\n" // real loose-equality (flagged)
	if got := (LooseEquality{}).MatchFile(context.Background(), ntarget(t, "tp.ts", pos)); len(got) != 1 {
		t.Fatalf("expected 1 finding for real loose equality, got %d (%+v)", len(got), got)
	}
}

// ---- bugs/python-is-literal: string/docstring masking (FP regression) ----

func TestPythonIsLiteral_StringsNotFlagged(t *testing.T) {
	// FP inputs: the `is <literal>` pattern inside a normal string and inside a
	// single-line triple-quoted docstring must NOT be flagged.
	neg := "msg = \"the answer is 42 today\"\n" + // word `is` + number inside a string
		"\"\"\"This function checks whether x is 5 and returns it.\"\"\"\n" + // single-line docstring
		"note = 'the value is \"7\" here'\n"
	if got := (PythonIsLiteral{}).MatchFile(context.Background(), ntarget(t, "lit.py", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings for `is`-in-string, got %d (%+v)", len(got), got)
	}

	// Multi-line triple-quoted docstring containing the pattern is also ignored,
	// while a genuine `is 5` comparison after it still fires.
	mixed := "def f(n):\n" +
		"    \"\"\"Doc line one.\n" +
		"    behavior: if n is 5 then special.\n" +
		"    \"\"\"\n" +
		"    if n is 5:\n" + // real identity-vs-literal bug (flagged)
		"        return 1\n"
	if got := (PythonIsLiteral{}).MatchFile(context.Background(), ntarget(t, "mixed.py", mixed)); len(got) != 1 {
		t.Fatalf("expected 1 finding (only the real `is 5`), got %d (%+v)", len(got), got)
	}
}

// ---- bugs/self-assignment: inline block comment (FP regression) ----

func TestSelfAssignment_BlockCommentNotFlagged(t *testing.T) {
	// FP input: a self-assignment with a trailing single-line /* */ block comment
	// previously defeated the end-of-line anchor and was missed/mishandled; a
	// genuine self-assignment with such a comment SHOULD still fire as a finding.
	pos := "package x\n" +
		"func f() {\n" +
		"\ta = a /* keep */\n" + // genuine self-assignment + trailing block comment
		"}\n"
	if got := (SelfAssignment{}).MatchFile(context.Background(), ntarget(t, "bc.go", pos)); len(got) != 1 {
		t.Fatalf("expected 1 finding for self-assign w/ block comment, got %d (%+v)", len(got), got)
	}

	// Non-self-assignment with a block comment must NOT be flagged.
	neg := "package x\n" +
		"func g() {\n" +
		"\ta = b /* note */\n" +
		"\tc = \"x = x\" /* string, not a self-assign */\n" +
		"}\n"
	if got := (SelfAssignment{}).MatchFile(context.Background(), ntarget(t, "bcneg.go", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// ---- bugs/error-not-wrapped ----

func TestErrorNotWrapped(t *testing.T) {
	pos := `package x
func f() error {
	return fmt.Errorf("read failed: %v", err)
}
func g() error {
	return fmt.Errorf("write %s failed: %s", name, dbErr)
}
`
	got := (ErrorNotWrapped{}).MatchFile(context.Background(), ntarget(t, "x.go", pos))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}

	neg := `package x
func ok() error {
	return fmt.Errorf("read failed: %w", err)
}
func noErr() error {
	return fmt.Errorf("count is %d for %s", n, name)
}
`
	if got := (ErrorNotWrapped{}).MatchFile(context.Background(), ntarget(t, "y.go", neg)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}
