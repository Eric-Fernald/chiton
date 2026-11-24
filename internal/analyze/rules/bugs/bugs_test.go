package bugs

import (
	"context"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func target(t *testing.T, path, content string) analyze.FileTarget {
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

func TestDeferInLoop_Positive(t *testing.T) {
	src := `package x

func bad() {
	for i := 0; i < 10; i++ {
		f, _ := openFile()
		defer f.Close()
	}
}
`
	got := (DeferInLoop{}).MatchFile(context.Background(), target(t, "bad.go", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
	if got[0].Span == nil || got[0].Span.StartLine != 6 {
		t.Fatalf("expected finding at line 6, got %+v", got[0].Span)
	}
}

func TestDeferInLoop_Negative(t *testing.T) {
	src := `package x

func ok() {
	f, _ := openFile()
	defer f.Close()
	for i := 0; i < 10; i++ {
		println(i)
	}
}
`
	got := (DeferInLoop{}).MatchFile(context.Background(), target(t, "ok.go", src))
	if len(got) != 0 {
		t.Fatalf("expected no findings, got %+v", got)
	}
}

func TestDeferInLoop_IgnoresStringLiterals(t *testing.T) {
	// "defer" inside a string literal must not match; a real defer still does.
	src := `package x

func ok() {
	for i := 0; i < 10; i++ {
		msg := "wrap in func() { defer f() }()"
		println(msg)
	}
}

func bad() {
	for i := 0; i < 10; i++ {
		defer println(i)
	}
}
`
	got := (DeferInLoop{}).MatchFile(context.Background(), target(t, "strlit.go", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
	if got[0].Span == nil || got[0].Span.StartLine != 12 {
		t.Fatalf("expected finding at line 12, got %+v", got[0].Span)
	}
}

func TestConsoleLog(t *testing.T) {
	src := `function debug() {
  console.log("hi"); // diagnostic
  return 1;
}
`
	got := (ConsoleLog{}).MatchFile(context.Background(), target(t, "x.js", src))
	if len(got) != 1 {
		t.Fatalf("expected 1, got %d", len(got))
	}

	// Test file is skipped.
	if got := (ConsoleLog{}).MatchFile(context.Background(), target(t, "x.test.js", src)); len(got) != 0 {
		t.Fatalf("expected test file to skip; got %d", len(got))
	}

	// console.log inside a string literal — current implementation will still
	// match; document it. Comment-only mention should NOT.
	neg := `// console.log("removed")`
	if got := (ConsoleLog{}).MatchFile(context.Background(), target(t, "y.js", neg)); len(got) != 0 {
		t.Fatalf("expected comment to skip; got %d", len(got))
	}
}
