package style

import (
	"context"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

func styleTarget(t *testing.T, path, content string) analyze.FileTarget {
	t.Helper()
	uri := "file://test/" + path
	return analyze.FileTarget{
		Path:     path,
		URI:      uri,
		Content:  []byte(content),
		FileNode: model.Node{ID: "f", Kind: model.KindFile, Label: path, SourceURI: uri},
	}
}

func TestLongLine(t *testing.T) {
	long := strings.Repeat("a", LongLineLimit+1)
	atLimit := strings.Repeat("b", LongLineLimit)
	src := "package x\n" + "var s = \"" + long + "\"\n" + "var ok = \"" + atLimit[:LongLineLimit-12] + "\"\n"

	got := (LongLine{}).MatchFile(context.Background(), styleTarget(t, "x.go", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 long-line finding, got %d (%+v)", len(got), got)
	}
	if got[0].Span.StartLine != 2 {
		t.Fatalf("expected finding on line 2, got %d", got[0].Span.StartLine)
	}

	// Negative: a file with only short lines must produce nothing.
	clean := "package x\nvar a = 1\nvar b = 2\n"
	if got := (LongLine{}).MatchFile(context.Background(), styleTarget(t, "x.go", clean)); len(got) != 0 {
		t.Fatalf("expected 0 findings on short-line file, got %d", len(got))
	}
}

func TestLongLineCountsRunesNotBytes(t *testing.T) {
	// A line of multi-byte runes that is under the rune limit but over the byte
	// limit must NOT fire. 100 'é' = 100 runes (under 120) but 200 bytes (over).
	line := strings.Repeat("é", 100) // each 'é' is 2 bytes
	src := line + "\n"
	if got := (LongLine{}).MatchFile(context.Background(), styleTarget(t, "x.py", src)); len(got) != 0 {
		t.Fatalf("expected 0 findings (under rune limit), got %d", len(got))
	}
}

func TestMissingFinalNewline(t *testing.T) {
	// Positive: no trailing newline.
	got := (MissingFinalNewline{}).MatchFile(context.Background(), styleTarget(t, "x.go", "package x\nvar a = 1"))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding, got %d (%+v)", len(got), got)
	}
	if got[0].Span.StartLine != 2 {
		t.Fatalf("expected finding on final line 2, got %d", got[0].Span.StartLine)
	}

	// Negative: ends with \n.
	if got := (MissingFinalNewline{}).MatchFile(context.Background(), styleTarget(t, "x.go", "package x\n")); len(got) != 0 {
		t.Fatalf("expected 0 findings on newline-terminated file, got %d", len(got))
	}
	// Negative: ends with \r\n (CRLF).
	if got := (MissingFinalNewline{}).MatchFile(context.Background(), styleTarget(t, "x.go", "package x\r\n")); len(got) != 0 {
		t.Fatalf("expected 0 findings on CRLF-terminated file, got %d", len(got))
	}
	// Negative: empty file is not flagged.
	if got := (MissingFinalNewline{}).MatchFile(context.Background(), styleTarget(t, "x.go", "")); len(got) != 0 {
		t.Fatalf("expected 0 findings on empty file, got %d", len(got))
	}
}

// ruleApplies mirrors analyze.ruleApplies (unexported) so language-filter
// regressions can be asserted from this external-ish test without exporting it.
// nil/empty filter accepts everything; otherwise the extension must match.
func ruleApplies(langs []string, ext string) bool {
	if len(langs) == 0 {
		return true
	}
	for _, l := range langs {
		if strings.EqualFold(l, ext) {
			return true
		}
	}
	return false
}

// TestTrailingWhitespaceLanguageFilter pins the fix for the false positive where
// style/trailing-whitespace ran on Markdown (two trailing spaces is a valid hard
// line break) and on image binaries. The Meta must now carry a Languages filter
// that excludes md/png/jpg while still accepting source files.
func TestTrailingWhitespaceLanguageFilter(t *testing.T) {
	langs := (TrailingWhitespace{}).Meta().Languages
	if len(langs) == 0 {
		t.Fatal("style/trailing-whitespace must declare a Languages filter, got none")
	}
	// False positives: must NOT run on Markdown or image binaries.
	for _, ext := range []string{"md", "markdown", "png", "jpg", "jpeg", "gif"} {
		if ruleApplies(langs, ext) {
			t.Errorf("trailing-whitespace must not apply to .%s files", ext)
		}
	}
	// True positives: must still run on source languages.
	for _, ext := range []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"} {
		if !ruleApplies(langs, ext) {
			t.Errorf("trailing-whitespace must still apply to .%s files", ext)
		}
	}

	// The detection logic itself must still fire on a genuine .go offender, and
	// stay silent on a clean source line.
	dirty := "package x\nvar a = 1  \n"
	if got := (TrailingWhitespace{}).MatchFile(context.Background(), styleTarget(t, "x.go", dirty)); len(got) != 1 {
		t.Fatalf("expected 1 trailing-whitespace finding on offending .go source, got %d", len(got))
	}
	clean := "package x\nvar a = 1\n"
	if got := (TrailingWhitespace{}).MatchFile(context.Background(), styleTarget(t, "x.go", clean)); len(got) != 0 {
		t.Fatalf("expected 0 findings on clean .go source, got %d", len(got))
	}
}

// TestMixedIndentLanguageFilter pins the fix for the false positive where
// style/mixed-indent ran on Markdown and image binaries.
func TestMixedIndentLanguageFilter(t *testing.T) {
	langs := (MixedIndent{}).Meta().Languages
	if len(langs) == 0 {
		t.Fatal("style/mixed-indent must declare a Languages filter, got none")
	}
	for _, ext := range []string{"md", "markdown", "png", "jpg", "jpeg", "gif"} {
		if ruleApplies(langs, ext) {
			t.Errorf("mixed-indent must not apply to .%s files", ext)
		}
	}
	for _, ext := range []string{"go", "py", "js", "ts", "jsx", "tsx", "rs", "java"} {
		if !ruleApplies(langs, ext) {
			t.Errorf("mixed-indent must still apply to .%s files", ext)
		}
	}

	// Detection must still fire on a genuine mixed tab+space indent, and stay
	// silent on clean tab-only indentation.
	mixed := "package x\nfunc f() {\n\t   return\n}\n"
	if got := (MixedIndent{}).MatchFile(context.Background(), styleTarget(t, "x.go", mixed)); len(got) != 1 {
		t.Fatalf("expected 1 mixed-indent finding on offending .go source, got %d", len(got))
	}
	clean := "package x\nfunc f() {\n\treturn\n}\n"
	if got := (MixedIndent{}).MatchFile(context.Background(), styleTarget(t, "x.go", clean)); len(got) != 0 {
		t.Fatalf("expected 0 findings on clean tab-indented .go source, got %d", len(got))
	}
}

func TestMultipleBlankLines(t *testing.T) {
	// Positive: three consecutive blank lines (one is whitespace-only).
	src := "package x\n\n\n\nvar a = 1\n"
	got := (MultipleBlankLines{}).MatchFile(context.Background(), styleTarget(t, "x.go", src))
	if len(got) != 1 {
		t.Fatalf("expected 1 finding for a run of 3 blanks, got %d (%+v)", len(got), got)
	}
	if got[0].Span.StartLine != 4 {
		t.Fatalf("expected finding at first excess blank (line 4), got %d", got[0].Span.StartLine)
	}

	// Positive: a longer run still produces exactly one finding.
	long := "a\n\n\n\n\n\nb\n"
	if got := (MultipleBlankLines{}).MatchFile(context.Background(), styleTarget(t, "x.go", long)); len(got) != 1 {
		t.Fatalf("expected exactly 1 finding for a single long run, got %d", len(got))
	}

	// Positive: whitespace-only lines count as blank.
	ws := "a\n   \n\t\n  \nb\n"
	if got := (MultipleBlankLines{}).MatchFile(context.Background(), styleTarget(t, "x.go", ws)); len(got) != 1 {
		t.Fatalf("expected 1 finding for whitespace-only blank run, got %d", len(got))
	}

	// Negative: two consecutive blanks is allowed.
	twoBlanks := "package x\n\n\nvar a = 1\n"
	if got := (MultipleBlankLines{}).MatchFile(context.Background(), styleTarget(t, "x.go", twoBlanks)); len(got) != 0 {
		t.Fatalf("expected 0 findings for 2 blank lines, got %d", len(got))
	}
}
