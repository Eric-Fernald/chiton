package security

import (
	"context"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func target(t *testing.T, path string, content string) analyze.FileTarget {
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

func TestHardcodedSecret_Positive(t *testing.T) {
	cases := map[string]string{
		"aws":     `const Key = "AKIAIOSFODNN7EXAMPLE"`,
		"slack":   `token := "xoxb-1234567890-0987654321"`,
		"private": "-----BEGIN RSA PRIVATE KEY-----",
		"jwt":     `auth := "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"`,
	}
	r := HardcodedSecret{}
	for name, src := range cases {
		t.Run(name, func(t *testing.T) {
			got := r.MatchFile(context.Background(), target(t, "secret.go", src))
			if len(got) == 0 {
				t.Fatalf("expected a finding for %q", src)
			}
		})
	}
}

func TestHardcodedSecret_Negative(t *testing.T) {
	src := `
// just a regular file
const greeting = "hello world"
const id = "abc-123-def-456"
`
	got := HardcodedSecret{}.MatchFile(context.Background(), target(t, "ok.go", src))
	if len(got) != 0 {
		t.Fatalf("unexpected findings: %+v", got)
	}
}

func TestEvalUsage(t *testing.T) {
	pos := `result = eval(user_input)`
	if got := (EvalUsage{}).MatchFile(context.Background(), target(t, "x.py", pos)); len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d", len(got))
	}

	// Comment-only mention of eval should not fire.
	neg := `# we used to use eval(x) but switched to ast.literal_eval`
	if got := (EvalUsage{}).MatchFile(context.Background(), target(t, "x.py", neg)); len(got) != 0 {
		t.Fatalf("expected no finding in comment; got %d", len(got))
	}

	// 'myeval(' should not match.
	neg2 := `result = myeval(x)`
	if got := (EvalUsage{}).MatchFile(context.Background(), target(t, "x.py", neg2)); len(got) != 0 {
		t.Fatalf("expected no finding for myeval; got %d", len(got))
	}
}

func TestWeakRandom(t *testing.T) {
	prod := `package main

import (
	"fmt"
	"math/rand"
)

func main() { fmt.Println(rand.Int()) }
`
	got := (WeakRandom{}).MatchFile(context.Background(), target(t, "main.go", prod))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding; got %d", len(got))
	}
	if !strings.Contains(got[0].Message, "math/rand") {
		t.Fatalf("message missing math/rand: %q", got[0].Message)
	}

	// Test file should be skipped.
	tgt := target(t, "main_test.go", prod)
	if got := (WeakRandom{}).MatchFile(context.Background(), tgt); len(got) != 0 {
		t.Fatalf("expected test file to skip; got %d findings", len(got))
	}

	// Production file with no math/rand import shouldn't trigger.
	noImport := `package main
import "fmt"
func main() { fmt.Println("ok") }
`
	if got := (WeakRandom{}).MatchFile(context.Background(), target(t, "main.go", noImport)); len(got) != 0 {
		t.Fatalf("expected no finding; got %d", len(got))
	}
}
