package performance

import (
	"context"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// ftarget builds a FileTarget for a synthesized source file. No graph store is
// required; rules are invoked directly via MatchFile.
func ftarget(path, content string) analyze.FileTarget {
	uri := "file://test/" + path
	return analyze.FileTarget{
		Path:     path,
		URI:      uri,
		Content:  []byte(content),
		FileNode: model.Node{ID: "f", Kind: model.KindFile, Label: path, SourceURI: uri},
	}
}

func countFindings(t *testing.T, r analyze.FileRule, path, content string) int {
	t.Helper()
	return len(r.MatchFile(context.Background(), ftarget(path, content)))
}

func TestRegexCompileInLoop(t *testing.T) {
	r := RegexCompileInLoop{}

	// Go: positive (inside loop) and negative (hoisted out).
	goPos := `package x
func f(items []string) {
	for _, s := range items {
		re := regexp.MustCompile("[0-9]+")
		_ = re.MatchString(s)
	}
}`
	if got := countFindings(t, r, "a.go", goPos); got != 1 {
		t.Fatalf("go positive: expected 1, got %d", got)
	}
	goNeg := `package x
var re = regexp.MustCompile("[0-9]+")
func f(items []string) {
	for _, s := range items {
		_ = re.MatchString(s)
	}
}`
	if got := countFindings(t, r, "a.go", goNeg); got != 0 {
		t.Fatalf("go negative: expected 0, got %d", got)
	}

	// Python: positive (indented under for) and negative (above loop).
	pyPos := "import re\nfor s in items:\n    p = re.compile(r'\\d+')\n    p.match(s)\n"
	if got := countFindings(t, r, "a.py", pyPos); got != 1 {
		t.Fatalf("py positive: expected 1, got %d", got)
	}
	pyNeg := "import re\np = re.compile(r'\\d+')\nfor s in items:\n    p.match(s)\n"
	if got := countFindings(t, r, "a.py", pyNeg); got != 0 {
		t.Fatalf("py negative: expected 0, got %d", got)
	}

	// JS: new RegExp inside while loop.
	jsPos := `function f(items) {
	while (items.length) {
		const re = new RegExp("[0-9]+");
		re.test(items.pop());
	}
}`
	if got := countFindings(t, r, "a.js", jsPos); got != 1 {
		t.Fatalf("js positive: expected 1, got %d", got)
	}

	// Java with brace on next line (Allman style) inside the loop.
	javaPos := `class C {
	void f(String[] xs) {
		for (int i = 0; i < xs.length; i++)
		{
			Pattern.compile("[0-9]+");
		}
	}
}`
	if got := countFindings(t, r, "a.java", javaPos); got != 1 {
		t.Fatalf("java positive: expected 1, got %d", got)
	}
}

func TestSleepInLoop(t *testing.T) {
	r := SleepInLoop{}

	goPos := `package x
func f() {
	for i := 0; i < 10; i++ {
		time.Sleep(time.Second)
	}
}`
	if got := countFindings(t, r, "a.go", goPos); got != 1 {
		t.Fatalf("go positive: expected 1, got %d", got)
	}
	goNeg := `package x
func f() {
	time.Sleep(time.Second)
	for i := 0; i < 10; i++ {
		work(i)
	}
}`
	if got := countFindings(t, r, "a.go", goNeg); got != 0 {
		t.Fatalf("go negative: expected 0, got %d", got)
	}

	pyPos := "import time\nfor i in range(10):\n    time.sleep(1)\n"
	if got := countFindings(t, r, "a.py", pyPos); got != 1 {
		t.Fatalf("py positive: expected 1, got %d", got)
	}

	javaPos := `class C {
	void f() {
		while (running) {
			Thread.sleep(100);
		}
	}
}`
	if got := countFindings(t, r, "a.java", javaPos); got != 1 {
		t.Fatalf("java positive: expected 1, got %d", got)
	}
}

func TestAwaitInLoop(t *testing.T) {
	r := AwaitInLoop{}

	pos := `async function f(urls) {
	for (const u of urls) {
		const res = await fetch(u);
		console.log(res);
	}
}`
	if got := countFindings(t, r, "a.ts", pos); got != 1 {
		t.Fatalf("positive: expected 1, got %d", got)
	}

	// Negative: await outside loop, and `for await` header must not flag.
	neg := `async function f(urls) {
	const res = await fetch(urls[0]);
	for await (const chunk of stream) {
		process(chunk);
	}
}`
	if got := countFindings(t, r, "a.ts", neg); got != 0 {
		t.Fatalf("negative: expected 0, got %d", got)
	}

	// `for await` header inside which a plain await appears -> only the body
	// await counts (the header line is skipped).
	mixed := `async function f() {
	for await (const x of gen()) {
		await sink(x);
	}
}`
	if got := countFindings(t, r, "a.ts", mixed); got != 1 {
		t.Fatalf("mixed: expected 1, got %d", got)
	}
}

func TestIndexOfNotIncludes(t *testing.T) {
	r := IndexOfNotIncludes{}

	for _, pos := range []string{
		`if (arr.indexOf(x) !== -1) doIt();`,
		`if (arr.indexOf(x) != -1) doIt();`,
		`const has = s.indexOf("a") >= 0;`,
		`const has = s.indexOf("a") > -1;`,
	} {
		if got := countFindings(t, r, "a.js", pos); got != 1 {
			t.Fatalf("positive %q: expected 1, got %d", pos, got)
		}
	}

	for _, neg := range []string{
		`const i = arr.indexOf(x);`,
		`if (arr.indexOf(x) > 5) doIt();`,
		`const has = arr.includes(x);`,
	} {
		if got := countFindings(t, r, "a.js", neg); got != 0 {
			t.Fatalf("negative %q: expected 0, got %d", neg, got)
		}
	}
}

func TestRangeLen(t *testing.T) {
	r := RangeLen{}

	pos := "for i in range(len(items)):\n    print(items[i])\n"
	if got := countFindings(t, r, "a.py", pos); got != 1 {
		t.Fatalf("positive: expected 1, got %d", got)
	}

	for _, neg := range []string{
		"for i, x in enumerate(items):\n    print(x)\n",
		"for i in range(len(items) - 1):\n    print(i)\n",
		"for i in range(10):\n    print(i)\n",
	} {
		if got := countFindings(t, r, "a.py", neg); got != 0 {
			t.Fatalf("negative %q: expected 0, got %d", neg, got)
		}
	}
}

func TestJSONDeepClone(t *testing.T) {
	r := JSONDeepClone{}

	pos := `const copy = JSON.parse(JSON.stringify(obj));`
	if got := countFindings(t, r, "a.ts", pos); got != 1 {
		t.Fatalf("positive: expected 1, got %d", got)
	}

	for _, neg := range []string{
		`const copy = structuredClone(obj);`,
		`const s = JSON.stringify(obj);`,
		`const o = JSON.parse(input);`,
		`// JSON.parse(JSON.stringify(obj)) is bad`,
	} {
		if got := countFindings(t, r, "a.ts", neg); got != 0 {
			t.Fatalf("negative %q: expected 0, got %d", neg, got)
		}
	}
}
