package quality

import (
	"context"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func target(t *testing.T, path string, content string, nodes []model.Node) analyze.FileTarget {
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
		Nodes: nodes,
	}
}

func TestLongFunction(t *testing.T) {
	short := model.Node{
		ID:    "s1",
		Kind:  model.KindSymbol,
		Label: "small",
		Span:  &model.SourceSpan{StartLine: 1, EndLine: 10},
		Attrs: map[string]any{"lang_kind": "func"},
	}
	long := model.Node{
		ID:    "s2",
		Kind:  model.KindSymbol,
		Label: "big",
		Span:  &model.SourceSpan{StartLine: 1, EndLine: LongFunctionLines + 5},
		Attrs: map[string]any{"lang_kind": "func"},
	}
	notFunc := model.Node{
		ID:    "s3",
		Kind:  model.KindSymbol,
		Label: "Type",
		Span:  &model.SourceSpan{StartLine: 1, EndLine: LongFunctionLines + 50},
		Attrs: map[string]any{"lang_kind": "type"},
	}

	tg := target(t, "x.go", "", []model.Node{short, long, notFunc})
	got := (LongFunction{}).MatchFile(context.Background(), tg)
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d", len(got))
	}
	if got[0].TargetNodeID != "s2" {
		t.Fatalf("expected finding on long function, got %q", got[0].TargetNodeID)
	}
}

func TestTodoComment(t *testing.T) {
	src := `package x
// TODO: refactor this
const v = 1
// also: fixme later
func ok() {} // XXX hack
`
	got := (TodoComment{}).MatchFile(context.Background(), target(t, "x.go", src, nil))
	if len(got) != 3 {
		t.Fatalf("expected 3 markers, got %d (%+v)", len(got), got)
	}
}

func TestTodoComment_IgnoresStringLiterals(t *testing.T) {
	// Markers inside string literals (e.g. a rule's own pattern list) must not
	// match; a marker in a real comment still does.
	src := `package x

var pattern = "TODO|FIXME|XXX|HACK"

// TODO: real marker
`
	got := (TodoComment{}).MatchFile(context.Background(), target(t, "x.go", src, nil))
	if len(got) != 1 {
		t.Fatalf("expected 1 marker, got %d (%+v)", len(got), got)
	}
	if got[0].Span == nil || got[0].Span.StartLine != 5 {
		t.Fatalf("expected finding at line 5, got %+v", got[0].Span)
	}

	// Test sources are skipped entirely (fixtures embed markers in raw strings).
	if got := (TodoComment{}).MatchFile(context.Background(), target(t, "x_test.go", src, nil)); len(got) != 0 {
		t.Fatalf("expected test file to skip; got %d", len(got))
	}
}
