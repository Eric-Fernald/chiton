package reliability

import (
	"context"
	"testing"
)

// False-positive regressions: rust-unwrap literal masking, abrupt-exit entrypoints, Go block-comment masking.

// ---- rust-unwrap: string-literal & block-comment masking ----

func TestRustUnwrap_FP_StringAndBlockComment(t *testing.T) {
	src := "fn run() {\n" +
		"    let msg = \"call .unwrap() to crash\";\n" +
		"    println!(\"use .expect(this) carefully\");\n" +
		"    let raw = r#\"text with .unwrap() inside\"#;\n" +
		"    let chr = ')';\n" +
		"    /* block comment\n" +
		"       foo().unwrap();\n" +
		"       bar().expect(\"x\"); */\n" +
		"    let _ = msg;\n" +
		"}\n"
	got := (RustUnwrap{}).MatchFile(context.Background(), relTarget("src/run.rs", src))
	if len(got) != 0 {
		t.Fatalf("expected 0 findings (all inside strings/comments), got %d (%+v)", len(got), got)
	}
}

func TestRustUnwrap_FP_TruePositiveStillFires(t *testing.T) {
	// A real .unwrap()/.expect( on code lines (alongside a benign string literal)
	// must still be flagged.
	src := "fn run() {\n" +
		"    let note = \".unwrap() is documented here\"; // not real\n" +
		"    let x = foo().unwrap();\n" +
		"    let y = bar().expect(\"must exist\");\n" +
		"    let _ = note;\n" +
		"}\n"
	got := (RustUnwrap{}).MatchFile(context.Background(), relTarget("src/run.rs", src))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings (real unwrap/expect), got %d (%+v)", len(got), got)
	}
}

func TestRustUnwrap_FP_LifetimeNotMisparsed(t *testing.T) {
	// A lifetime annotation ('a) must not start a char literal that swallows the
	// rest of the line and hides a real .unwrap().
	src := "fn run<'a>(s: &'a str) {\n" +
		"    let x = foo().unwrap();\n" +
		"    let _ = s;\n" +
		"}\n"
	got := (RustUnwrap{}).MatchFile(context.Background(), relTarget("src/run.rs", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding (unwrap after lifetime), got %d (%+v)", len(got), got)
	}
}

// ---- abrupt-exit: broadened entrypoint detection ----

func TestAbruptExit_FP_GoPackageMainAnyName(t *testing.T) {
	// A package-main file under cmd/<tool>/ with a non-main.go name is an
	// entrypoint and must be exempt.
	src := `package main
import "os"
func run() { os.Exit(1) }
`
	got := (AbruptExit{}).MatchFile(context.Background(), relTarget("cmd/chiton/serve.go", src))
	if len(got) != 0 {
		t.Fatalf("expected 0 findings (package main entrypoint), got %d (%+v)", len(got), got)
	}
	// A genuine library file (package lib) under the same tree still fires.
	lib := `package lib
import "os"
func quit() { os.Exit(1) }
`
	if g := (AbruptExit{}).MatchFile(context.Background(), relTarget("cmd/chiton/helper.go", lib)); len(g) != 1 {
		t.Fatalf("expected 1 finding (library code), got %d (%+v)", len(g), g)
	}
}

func TestAbruptExit_FP_JSEntrypoints(t *testing.T) {
	for _, name := range []string{"server.js", "app.js", "start.js", "src/bin/tool.js"} {
		src := `function quit() { process.exit(1); }`
		if g := (AbruptExit{}).MatchFile(context.Background(), relTarget(name, src)); len(g) != 0 {
			t.Fatalf("expected 0 findings for entrypoint %s, got %d (%+v)", name, len(g), g)
		}
	}
}

func TestAbruptExit_FP_ShebangFile(t *testing.T) {
	src := "#!/usr/bin/env node\nfunction quit() { process.exit(1); }\n"
	if g := (AbruptExit{}).MatchFile(context.Background(), relTarget("tool.js", src)); len(g) != 0 {
		t.Fatalf("expected 0 findings for shebang entrypoint, got %d (%+v)", len(g), g)
	}
}

func TestAbruptExit_FP_NonEntrypointStillFires(t *testing.T) {
	// A plain JS module (no shebang, not a bin path, not an entry name) must
	// still be flagged.
	src := `function quit() { process.exit(1); }`
	if g := (AbruptExit{}).MatchFile(context.Background(), relTarget("worker.js", src)); len(g) != 1 {
		t.Fatalf("expected 1 finding (non-entrypoint module), got %d (%+v)", len(g), g)
	}
}

func TestAbruptExit_FP_GoBlockComment(t *testing.T) {
	// os.Exit inside a /* ... */ block comment (commented-out code) must not fire.
	src := `package lib
import "os"
/*
func old() { os.Exit(1) }
*/
func ok() error { return nil }
`
	if g := (AbruptExit{}).MatchFile(context.Background(), relTarget("lib.go", src)); len(g) != 0 {
		t.Fatalf("expected 0 findings (os.Exit in block comment), got %d (%+v)", len(g), g)
	}
}

// ---- block-comment masking for the Go line-scan rules ----

func TestPanicInSource_FP_BlockComment(t *testing.T) {
	src := `package lib
/*
func old() { panic("dead code") }
*/
const note = "panic(here) in a string is fine"
func ok() error { return nil }
`
	if g := (PanicInSource{}).MatchFile(context.Background(), relTarget("lib.go", src)); len(g) != 0 {
		t.Fatalf("expected 0 findings (panic in block comment/string), got %d (%+v)", len(g), g)
	}
	// Real panic still fires.
	real := `package lib
func must() { panic("boom") }
`
	if g := (PanicInSource{}).MatchFile(context.Background(), relTarget("lib.go", real)); len(g) != 1 {
		t.Fatalf("expected 1 finding (real panic), got %d (%+v)", len(g), g)
	}
}

func TestLogFatal_FP_BlockComment(t *testing.T) {
	src := `package svc
import "log"
/*
func old() { log.Fatal("dead") }
*/
func ok() error { return nil }
`
	if g := (LogFatalInCode{}).MatchFile(context.Background(), relTarget("svc.go", src)); len(g) != 0 {
		t.Fatalf("expected 0 findings (log.Fatal in block comment), got %d (%+v)", len(g), g)
	}
	// Real log.Fatal still fires.
	real := `package svc
import "log"
func load() { log.Fatal("boom") }
`
	if g := (LogFatalInCode{}).MatchFile(context.Background(), relTarget("svc.go", real)); len(g) != 1 {
		t.Fatalf("expected 1 finding (real log.Fatal), got %d (%+v)", len(g), g)
	}
}

func TestHTTPTimeout_Go_FP_BlockComment(t *testing.T) {
	src := `package c
import "net/http"
/*
func old() { http.Get("http://x") }
*/
func ok() error { return nil }
`
	if g := (MissingHTTPTimeout{}).MatchFile(context.Background(), relTarget("c.go", src)); len(g) != 0 {
		t.Fatalf("expected 0 findings (http.Get in block comment), got %d (%+v)", len(g), g)
	}
	// Real http.Get still fires.
	real := `package c
import "net/http"
func fetch() { resp, _ := http.Get("http://x"); _ = resp }
`
	if g := (MissingHTTPTimeout{}).MatchFile(context.Background(), relTarget("c.go", real)); len(g) != 1 {
		t.Fatalf("expected 1 finding (real http.Get), got %d (%+v)", len(g), g)
	}
}
