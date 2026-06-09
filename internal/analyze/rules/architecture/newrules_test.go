package architecture

import (
	"context"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// fileTarget builds a FileScope target from raw source for direct rule testing.
func fileTarget(path, content string) analyze.FileTarget {
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

// --- too-many-imports ---------------------------------------------------------

func TestTooManyImports_GoBlockOverThreshold(t *testing.T) {
	var b strings.Builder
	b.WriteString("package x\n\nimport (\n")
	for i := 0; i < TooManyImportsThreshold+3; i++ {
		b.WriteString("\t\"pkg/")
		b.WriteString(itoa(i))
		b.WriteString("\"\n")
	}
	b.WriteString(")\n")
	got := (TooManyImports{}).MatchFile(context.Background(), fileTarget("x.go", b.String()))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
}

func TestTooManyImports_GoUnderThreshold(t *testing.T) {
	src := "package x\n\nimport (\n\t\"fmt\"\n\t\"strings\"\n)\n\n// import \"fake\" in a comment must not count\nfunc f() {}\n"
	if got := (TooManyImports{}).MatchFile(context.Background(), fileTarget("x.go", src)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

func TestTooManyImports_PythonOverThreshold(t *testing.T) {
	var b strings.Builder
	for i := 0; i < TooManyImportsThreshold+2; i++ {
		b.WriteString("import mod")
		b.WriteString(itoa(i))
		b.WriteString("\n")
	}
	if got := (TooManyImports{}).MatchFile(context.Background(), fileTarget("m.py", b.String())); len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d", len(got))
	}
}

func TestTooManyImports_JSUnderThreshold(t *testing.T) {
	src := "import a from 'a';\nimport b from 'b';\nconst c = require('c');\n"
	if got := (TooManyImports{}).MatchFile(context.Background(), fileTarget("m.ts", src)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d", len(got))
	}
}

// --- large-file ---------------------------------------------------------------

func TestLargeFile_OverThreshold(t *testing.T) {
	src := strings.Repeat("x\n", LargeFileLineThreshold+5)
	if got := (LargeFile{}).MatchFile(context.Background(), fileTarget("big.go", src)); len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d", len(got))
	}
}

func TestLargeFile_UnderThreshold(t *testing.T) {
	src := strings.Repeat("x\n", 50)
	if got := (LargeFile{}).MatchFile(context.Background(), fileTarget("ok.go", src)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d", len(got))
	}
}

// --- init-side-effect ---------------------------------------------------------

func TestInitSideEffect_DetectsSideEffectWork(t *testing.T) {
	// An init that reads the environment is real side-effect work and SHOULD fire.
	src := "package x\n\nfunc init() {\n\tcfg = os.Getenv(\"CFG\")\n}\n"
	got := (InitSideEffect{}).MatchFile(context.Background(), fileTarget("x.go", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
	if got[0].Span == nil || got[0].Span.StartLine != 3 {
		t.Fatalf("expected finding on line 3, got %+v", got[0].Span)
	}
}

func TestInitSideEffect_IgnoresOther(t *testing.T) {
	// A method named init-ish, an indented (nested-looking) line, and a comment
	// must not trip the rule.
	src := "package x\n\n// func init() in a comment\nfunc initialize() {}\n\tfunc init() {}\n"
	if got := (InitSideEffect{}).MatchFile(context.Background(), fileTarget("x.go", src)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// FALSE POSITIVE (HIGH NOISE): the idiomatic registration init — the exact
// pattern every rule file in this package uses — must NOT fire.
func TestInitSideEffect_IgnoresRegistrationOnly(t *testing.T) {
	cases := []string{
		// Single-line registration (note the InitSideEffect{} braces inside body).
		"package x\n\nfunc init() { analyze.Register(InitSideEffect{}) }\n",
		// Multi-line registration with a RegisterSomething call.
		"package x\n\nfunc init() {\n\tRegisterRule(myRule{})\n}\n",
		// sql.Register is a registry hook, not a connection setup.
		"package x\n\nfunc init() {\n\tsql.Register(\"drv\", &drv{})\n}\n",
	}
	for i, src := range cases {
		if got := (InitSideEffect{}).MatchFile(context.Background(), fileTarget("x.go", src)); len(got) != 0 {
			t.Fatalf("case %d: expected 0 findings for registration-only init, got %d (%+v)", i, len(got), got)
		}
	}
}

// FALSE POSITIVE: the `init` keyword inside a block comment or a backtick raw
// string must not be treated as a real init declaration.
func TestInitSideEffect_IgnoresInitInCommentOrRawString(t *testing.T) {
	cases := []string{
		// `func init()` text living inside a block comment.
		"package x\n\n/*\nfunc init() {\n\tos.Getenv(\"X\")\n}\n*/\nvar y = 1\n",
		// A backtick raw string whose contents look like an init with IO.
		"package x\n\nvar tmpl = `\nfunc init() {\n\tos.ReadFile(\"x\")\n}\n`\n",
	}
	for i, src := range cases {
		if got := (InitSideEffect{}).MatchFile(context.Background(), fileTarget("x.go", src)); len(got) != 0 {
			t.Fatalf("case %d: expected 0 findings (init in comment/raw string), got %d (%+v)", i, len(got), got)
		}
	}
}

// TRUE POSITIVE: each side-effect class still fires.
func TestInitSideEffect_DetectsEachSideEffectClass(t *testing.T) {
	cases := []string{
		// env write
		"package x\n\nfunc init() {\n\tos.Setenv(\"A\", \"b\")\n}\n",
		// file I/O
		"package x\n\nfunc init() {\n\tdata, _ = os.ReadFile(\"cfg\")\n}\n",
		// DB setup
		"package x\n\nfunc init() {\n\tdb, _ = sql.Open(\"pg\", dsn)\n}\n",
		// network setup via http call
		"package x\n\nfunc init() {\n\thttp.Get(\"http://x\")\n}\n",
		// network dial
		"package x\n\nfunc init() {\n\tconn, _ = net.Dial(\"tcp\", addr)\n}\n",
		// package-global mutation
		"package x\n\nfunc init() {\n\tGlobalCache = newCache()\n}\n",
	}
	for i, src := range cases {
		if got := (InitSideEffect{}).MatchFile(context.Background(), fileTarget("x.go", src)); len(got) != 1 {
			t.Fatalf("case %d: expected 1 finding, got %d (%+v)", i, len(got), got)
		}
	}
}

// A side-effect signature that appears only inside a STRING or COMMENT within
// an otherwise-registration-only init body must not fire (body masking).
func TestInitSideEffect_IgnoresSideEffectInBodyStringOrComment(t *testing.T) {
	cases := []string{
		"package x\n\nfunc init() {\n\tRegister(\"os.Getenv\") // not a real call\n}\n",
		"package x\n\nfunc init() {\n\t// os.ReadFile would be a side effect\n\tRegister(r)\n}\n",
	}
	for i, src := range cases {
		if got := (InitSideEffect{}).MatchFile(context.Background(), fileTarget("x.go", src)); len(got) != 0 {
			t.Fatalf("case %d: expected 0 findings (signature in string/comment), got %d (%+v)", i, len(got), got)
		}
	}
}

// FALSE POSITIVE (too-many-imports): dynamic import() calls and import.meta in
// JS/TS must not be counted as import statements.
func TestTooManyImports_JSIgnoresDynamicImportAndMeta(t *testing.T) {
	var b strings.Builder
	// A pile of dynamic imports + import.meta references, well over the limit if
	// they were (wrongly) counted.
	for i := 0; i < TooManyImportsThreshold+5; i++ {
		b.WriteString("const m")
		b.WriteString(itoa(i))
		b.WriteString(" = import('./mod")
		b.WriteString(itoa(i))
		b.WriteString("');\n")
	}
	for i := 0; i < TooManyImportsThreshold+5; i++ {
		b.WriteString("import.meta.url;\n")
	}
	if got := (TooManyImports{}).MatchFile(context.Background(), fileTarget("m.ts", b.String())); len(got) != 0 {
		t.Fatalf("expected 0 findings (dynamic import/import.meta), got %d (%+v)", len(got), got)
	}
}

// TRUE POSITIVE: real ES import statements over the threshold still fire even
// when dynamic imports are mixed in.
func TestTooManyImports_JSStaticImportsStillFire(t *testing.T) {
	var b strings.Builder
	for i := 0; i < TooManyImportsThreshold+3; i++ {
		b.WriteString("import x")
		b.WriteString(itoa(i))
		b.WriteString(" from './m")
		b.WriteString(itoa(i))
		b.WriteString("';\n")
	}
	// Dynamic import that must not change the verdict either way.
	b.WriteString("const lazy = import('./lazy');\n")
	if got := (TooManyImports{}).MatchFile(context.Background(), fileTarget("m.ts", b.String())); len(got) != 1 {
		t.Fatalf("expected 1 finding (static imports over limit), got %d (%+v)", len(got), got)
	}
}

// --- wildcard-import ----------------------------------------------------------

func TestWildcardImport_Detects(t *testing.T) {
	src := "from os import *\nfrom .pkg import *\n"
	if got := (WildcardImport{}).MatchFile(context.Background(), fileTarget("m.py", src)); len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}
}

func TestWildcardImport_IgnoresNamed(t *testing.T) {
	src := "from os import path, sep\nimport sys\n# from os import * (commented)\n"
	if got := (WildcardImport{}).MatchFile(context.Background(), fileTarget("m.py", src)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// --- deep-relative-import -----------------------------------------------------

func TestDeepRelativeImport_Detects(t *testing.T) {
	src := "import { a } from '../../../core/a';\nconst b = require(\"../../../../b\");\n"
	if got := (DeepRelativeImport{}).MatchFile(context.Background(), fileTarget("m.ts", src)); len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}
}

func TestDeepRelativeImport_IgnoresShallow(t *testing.T) {
	src := "import { a } from '../a';\nimport { b } from '../../b';\nimport c from 'pkg';\n"
	if got := (DeepRelativeImport{}).MatchFile(context.Background(), fileTarget("m.ts", src)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}

// --- dot-import ---------------------------------------------------------------

func TestDotImport_DetectsSingleAndBlock(t *testing.T) {
	src := "package x\n\nimport . \"errors\"\n\nimport (\n\t\"fmt\"\n\t. \"strings\"\n)\n"
	got := (DotImport{}).MatchFile(context.Background(), fileTarget("x.go", src))
	if len(got) != 2 {
		t.Fatalf("expected 2 findings, got %d (%+v)", len(got), got)
	}
}

func TestDotImport_IgnoresNormal(t *testing.T) {
	src := "package x\n\nimport (\n\t\"fmt\"\n\t_ \"side/effect\"\n\talias \"long/pkg\"\n)\n"
	if got := (DotImport{}).MatchFile(context.Background(), fileTarget("x.go", src)); len(got) != 0 {
		t.Fatalf("expected 0 findings, got %d (%+v)", len(got), got)
	}
}
