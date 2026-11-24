package reliability

import (
	"context"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func relTarget(path, content string) analyze.FileTarget {
	uri := "file://test/" + path
	return analyze.FileTarget{
		Path:    path,
		URI:     uri,
		Content: []byte(content),
		FileNode: model.Node{
			ID:        "f",
			Kind:      model.KindFile,
			Label:     path,
			SourceURI: uri,
		},
	}
}

// ---- empty-catch ----

func TestEmptyCatch_Positive(t *testing.T) {
	src := `function f() {
  try { risky(); }
  catch (e) {}
}
function g() {
  try { risky(); } catch {}
}
`
	got := (EmptyCatch{}).MatchFile(context.Background(), relTarget("a.ts", src))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}
}

func TestEmptyCatch_Negative(t *testing.T) {
	src := `function f() {
  try { risky(); }
  catch (e) { log(e); }
}
function g() {
  try { risky(); }
  catch (e) { /* intentionally ignored: best-effort */ }
}
// catch (e) {}  <- commented out, must not fire
const s = "catch (e) {}";
`
	got := (EmptyCatch{}).MatchFile(context.Background(), relTarget("a.js", src))
	if len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// ---- abrupt-exit ----

func TestAbruptExit_Go_Positive(t *testing.T) {
	src := `package lib
import "os"
func quit() { os.Exit(1) }
`
	got := (AbruptExit{}).MatchFile(context.Background(), relTarget("lib.go", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
}

func TestAbruptExit_Go_AllowsMain(t *testing.T) {
	src := `package main
import "os"
func main() { os.Exit(1) }
`
	got := (AbruptExit{}).MatchFile(context.Background(), relTarget("main.go", src))
	if len(got) != 0 {
		t.Fatalf("expected 0 findings in main.go, got %d (%+v)", len(got), got)
	}
}

func TestAbruptExit_JS_Positive(t *testing.T) {
	src := `function quit() { process.exit(1); }`
	got := (AbruptExit{}).MatchFile(context.Background(), relTarget("worker.js", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
}

func TestAbruptExit_Negative(t *testing.T) {
	src := `package lib
// os.Exit(1) in a comment must not fire
func ok() error { return nil }
`
	got := (AbruptExit{}).MatchFile(context.Background(), relTarget("lib.go", src))
	if len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// ---- log-fatal-in-code ----

func TestLogFatal_Positive(t *testing.T) {
	src := `package svc
import "log"
func load() {
	log.Fatalf("boom %v", err)
	log.Panic("nope")
}
`
	got := (LogFatalInCode{}).MatchFile(context.Background(), relTarget("svc.go", src))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}
}

func TestLogFatal_Negative(t *testing.T) {
	src := `package svc
import "log"
func load() {
	log.Printf("info %v", x)  // log.Fatal in comment ignored
	log.Println("ok")
}
`
	got := (LogFatalInCode{}).MatchFile(context.Background(), relTarget("svc.go", src))
	if len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
	// main.go is exempt.
	main := `package main
import "log"
func main() { log.Fatal("x") }
`
	if g := (LogFatalInCode{}).MatchFile(context.Background(), relTarget("main.go", main)); len(g) != 0 {
		t.Fatalf("expected 0 findings in main.go, got %d (%+v)", len(g), g)
	}
}

// ---- rust-unwrap ----

func TestRustUnwrap_Positive(t *testing.T) {
	src := `fn run() {
    let x = foo().unwrap();
    let y = bar().expect("must exist");
}
`
	got := (RustUnwrap{}).MatchFile(context.Background(), relTarget("src/run.rs", src))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}
}

func TestRustUnwrap_Negative(t *testing.T) {
	src := `fn run() -> Result<i32, E> {
    let x = foo()?;
    let y = bar().unwrap_or(0);
    let z = baz().unwrap_or_else(|| 1);
    // qux().unwrap(); commented out
    Ok(x + y + z)
}
`
	got := (RustUnwrap{}).MatchFile(context.Background(), relTarget("src/run.rs", src))
	if len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
	// tests/ directory is exempt.
	if g := (RustUnwrap{}).MatchFile(context.Background(), relTarget("tests/it.rs", "fn t() { x().unwrap(); }")); len(g) != 0 {
		t.Fatalf("expected 0 findings under tests/, got %d", len(g))
	}
}

// ---- missing-http-timeout ----

func TestHTTPTimeout_Go_Positive(t *testing.T) {
	src := `package c
import "net/http"
func fetch() { resp, _ := http.Get("http://x") ; _ = resp }
`
	got := (MissingHTTPTimeout{}).MatchFile(context.Background(), relTarget("c.go", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
}

func TestHTTPTimeout_Py_Positive(t *testing.T) {
	src := `import requests
def fetch():
    r = requests.get("http://x")
    return r
`
	got := (MissingHTTPTimeout{}).MatchFile(context.Background(), relTarget("c.py", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
}

func TestHTTPTimeout_Py_NegativeWithTimeout(t *testing.T) {
	src := `import requests
def fetch():
    r = requests.get(
        "http://x",
        timeout=5,
    )
    return r
`
	got := (MissingHTTPTimeout{}).MatchFile(context.Background(), relTarget("c.py", src))
	if len(got) != 0 {
		t.Fatalf("expected 0 findings (timeout present), got %d (%+v)", len(got), got)
	}
}

func TestHTTPTimeout_Negative(t *testing.T) {
	// Go: a client with a Timeout configured does not match the helper funcs.
	goSrc := `package c
import (
	"net/http"
	"time"
)
func fetch() {
	cl := &http.Client{Timeout: 5 * time.Second}
	_, _ = cl.Get("http://x")
}
`
	if g := (MissingHTTPTimeout{}).MatchFile(context.Background(), relTarget("c.go", goSrc)); len(g) != 0 {
		t.Fatalf("expected 0 Go findings, got %d (%+v)", len(g), g)
	}
	// Py: timeout inside a string must not count; the call itself has none.
	pySrc := `import requests
def fetch():
    # a previous call: requests.get(url, timeout=1)
    note = "remember timeout="
    return requests.get("http://x", headers={})
`
	if g := (MissingHTTPTimeout{}).MatchFile(context.Background(), relTarget("c.py", pySrc)); len(g) != 1 {
		t.Fatalf("expected 1 Py finding, got %d (%+v)", len(g), g)
	}
}

// ---- panic-in-source ----

func TestPanicInSource_Positive(t *testing.T) {
	src := `package lib
func must() { panic("unreachable") }
`
	got := (PanicInSource{}).MatchFile(context.Background(), relTarget("lib.go", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
}

func TestPanicInSource_Negative(t *testing.T) {
	src := `package lib
// panic("x") in a comment is fine
func ok() error {
	defer func() { _ = recover() }()
	return nil
}
`
	got := (PanicInSource{}).MatchFile(context.Background(), relTarget("lib.go", src))
	if len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
	// test files are exempt
	if g := (PanicInSource{}).MatchFile(context.Background(), relTarget("lib_test.go", `package lib
func TestX() { panic("boom") }`)); len(g) != 0 {
		t.Fatalf("expected 0 findings in _test.go, got %d", len(g))
	}
}
