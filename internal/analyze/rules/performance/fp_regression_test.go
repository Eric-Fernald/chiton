package performance

import "testing"

// Regression tests for fixed false positives. Each FP input must yield zero findings while a true
// positive still fires.

// Fix 1: a balanced single-line loop must not leak its frame onto later lines at the same depth
// (regex, sleep, and await rules share braceLoopScanner).
func TestSingleLineLoopDoesNotLeak(t *testing.T) {
	// regex-compile-in-loop: single-line loop, then a compile OUTSIDE any loop.
	{
		r := RegexCompileInLoop{}
		src := `function f(items) {
	for (const x of items) { use(x); }
	const re = regexp.MustCompile("[0-9]+");
	return re;
}`
		if got := countFindings(t, r, "a.js", src); got != 0 {
			t.Fatalf("regex single-line-loop leak: expected 0, got %d", got)
		}
	}
	// sleep-in-loop: single-line loop, then a sleep OUTSIDE any loop.
	{
		r := SleepInLoop{}
		src := `package x
func f() {
	for i := 0; i < 3; i++ { work(i) }
	time.Sleep(time.Second)
}`
		if got := countFindings(t, r, "a.go", src); got != 0 {
			t.Fatalf("sleep single-line-loop leak: expected 0, got %d", got)
		}
	}
	// await-in-loop: single-line loop, then an await OUTSIDE any loop.
	{
		r := AwaitInLoop{}
		src := `async function f(items) {
	for (const x of items) { sync(x); }
	const res = await fetch(url);
	return res;
}`
		if got := countFindings(t, r, "a.ts", src); got != 0 {
			t.Fatalf("await single-line-loop leak: expected 0, got %d", got)
		}
	}
	// True positives still fire: a real multi-line loop body.
	{
		r := SleepInLoop{}
		src := `package x
func f() {
	for i := 0; i < 3; i++ { work(i) }
	for j := 0; j < 3; j++ {
		time.Sleep(time.Second)
	}
}`
		if got := countFindings(t, r, "a.go", src); got != 1 {
			t.Fatalf("sleep after single-line-loop true positive: expected 1, got %d", got)
		}
	}
}

// Fix 2: re.compile must have a left word boundary so identifiers ending in
// `re` (core.compile, genre.compile) do not match.
func TestRegexCompileLeftBoundary(t *testing.T) {
	r := RegexCompileInLoop{}

	for _, neg := range []string{
		"for s in items:\n    p = core.compile(s)\n",
		"for s in items:\n    p = genre.compile(s)\n",
	} {
		if got := countFindings(t, r, "a.py", neg); got != 0 {
			t.Fatalf("re.compile boundary negative %q: expected 0, got %d", neg, got)
		}
	}
	// True positive: standalone re module still fires.
	pos := "import re\nfor s in items:\n    p = re.compile(s)\n"
	if got := countFindings(t, r, "a.py", pos); got != 1 {
		t.Fatalf("re.compile boundary positive: expected 1, got %d", got)
	}
}

// Fix 3: time.Sleep needs a left word boundary (uptime.sleep must not match), and a for or while
// header sharing a line with a preceding token must still be recognized.
func TestSleepBoundaryAndInlineHeader(t *testing.T) {
	r := SleepInLoop{}

	// uptime.sleep inside a loop must not match (boundary).
	neg := `package x
func f() {
	for i := 0; i < 3; i++ {
		uptime.sleep(1)
	}
}`
	if got := countFindings(t, r, "a.go", neg); got != 0 {
		t.Fatalf("uptime.sleep boundary negative: expected 0, got %d", got)
	}

	// A for header that shares a line with a preceding `if {` token must still
	// be recognized as a loop, so a sleep in its body fires.
	pos := `package x
func f() {
	if cond { for i := 0; i < 3; i++ {
		time.Sleep(time.Second)
	} }
}`
	if got := countFindings(t, r, "a.go", pos); got != 1 {
		t.Fatalf("inline for-header positive: expected 1, got %d", got)
	}
}

// Fix 4: a bare `await` word in a string or block comment inside a loop must
// not be flagged; only a real await expression fires.
func TestAwaitMasksStringsAndComments(t *testing.T) {
	r := AwaitInLoop{}

	// `await` inside a log string inside a loop -> no finding.
	negStr := `async function f(items) {
	for (const x of items) {
		console.log("processing await item now");
	}
}`
	if got := countFindings(t, r, "a.ts", negStr); got != 0 {
		t.Fatalf("await-in-string negative: expected 0, got %d", got)
	}

	// `await` inside a multi-line block comment inside a loop -> no finding.
	negComment := `async function f(items) {
	for (const x of items) {
		/* we should await
		   the result here later */
		sync(x);
	}
}`
	if got := countFindings(t, r, "a.ts", negComment); got != 0 {
		t.Fatalf("await-in-block-comment negative: expected 0, got %d", got)
	}

	// Real await expressions still fire.
	posCall := `async function f(items) {
	for (const x of items) {
		const r = await fetch(x);
	}
}`
	if got := countFindings(t, r, "a.ts", posCall); got != 1 {
		t.Fatalf("await fn() positive: expected 1, got %d", got)
	}
	posIdent := `async function f(items) {
	for (const x of items) {
		const r = await promise;
	}
}`
	if got := countFindings(t, r, "a.ts", posIdent); got != 1 {
		t.Fatalf("await ident positive: expected 1, got %d", got)
	}
}

// Fix 5: indexOf whose argument contains parentheses (a nested call) must still
// be detected.
func TestIndexOfNestedParens(t *testing.T) {
	r := IndexOfNotIncludes{}

	pos := `if (arr.indexOf(getKey(x)) !== -1) doIt();`
	if got := countFindings(t, r, "a.js", pos); got != 1 {
		t.Fatalf("indexOf nested-paren positive: expected 1, got %d", got)
	}
	// Simple argument still works.
	if got := countFindings(t, r, "a.js", `if (arr.indexOf(x) !== -1) doIt();`); got != 1 {
		t.Fatalf("indexOf simple positive: expected 1, got %d", got)
	}
}

// Fix 6: a range(len(x)) inside a Python docstring must not be flagged; real
// code still fires.
func TestRangeLenMasksDocstring(t *testing.T) {
	r := RangeLen{}

	neg := "def f(items):\n" +
		"    \"\"\"Example:\n" +
		"    for i in range(len(items)):\n" +
		"        print(items[i])\n" +
		"    \"\"\"\n" +
		"    return items\n"
	if got := countFindings(t, r, "a.py", neg); got != 0 {
		t.Fatalf("range-len docstring negative: expected 0, got %d", got)
	}

	// Real code still fires.
	pos := "for i in range(len(items)):\n    print(items[i])\n"
	if got := countFindings(t, r, "a.py", pos); got != 1 {
		t.Fatalf("range-len positive: expected 1, got %d", got)
	}
}
