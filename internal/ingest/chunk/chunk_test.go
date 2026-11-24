package chunk

import (
	"slices"
	"strings"
	"testing"
	"unicode/utf8"
)

func TestParagraphs(t *testing.T) {
	cases := []struct {
		name      string
		text      string
		size, max int
		want      []string
		truncated bool
	}{
		{"fits whole", " a\n\nb ", 10, 0, []string{" a\n\nb "}, false},
		{"packs paragraphs", "aa\n\nbb\n\ncc\n\ndd", 6, 0, []string{"aa\n\nbb", "cc\n\ndd"}, false},
		{"long paragraph is hard-split", "aa\n\nbbbbbbbbbb\n\ncc", 4, 0, []string{"aa", "bbbb", "bbbb", "bb", "cc"}, false},
		{"splits at a newline first", "one two\nthree four\nfive", 12, 0, []string{"one two", "three four", "five"}, false},
		{"then at a space", "alpha beta gamma", 12, 0, []string{"alpha beta", "gamma"}, false},
		{"never inside a rune", "ééééé", 5, 0, []string{"éé", "éé", "é"}, false},
		{"cap drops the rest", "aa\n\nbb\n\ncc\n\ndd", 2, 2, []string{"aa", "bb"}, true},
		{"cap exactly reached", "aa\n\nbb", 2, 2, []string{"aa", "bb"}, false},
		{"blank oversized text yields nothing", "     ", 2, 0, nil, false},
	}
	for _, c := range cases {
		got, truncated := Paragraphs(c.text, c.size, c.max)
		if !slices.Equal(got, c.want) || truncated != c.truncated {
			t.Errorf("%s: Paragraphs = %q, %v; want %q, %v", c.name, got, truncated, c.want, c.truncated)
		}
	}
}

// Each chunk becomes one extraction task, so none may exceed size whatever the text looks like.
func TestParagraphsNeverExceedsSize(t *testing.T) {
	const size, maxChunks = 8000, 60
	cases := []struct {
		name      string
		text      string
		truncated bool
	}{
		{"transcript, one newline per turn", strings.Repeat("Host: welcome back to the show.\n", 5000), false},
		{"no whitespace at all", strings.Repeat("x", 100_000), false},
		{"multibyte without whitespace", strings.Repeat("世界", 20_000), false},
		{"past the chunk cap", strings.Repeat("Host: welcome back to the show.\n", 20_000), true},
		{"paragraphs past the chunk cap", strings.Repeat(strings.Repeat("y", 5000)+"\n\n", 100), true},
	}
	for _, c := range cases {
		chunks, truncated := Paragraphs(c.text, size, maxChunks)
		if truncated != c.truncated {
			t.Errorf("%s: truncated = %v, want %v", c.name, truncated, c.truncated)
		}
		if len(chunks) > maxChunks || (c.truncated && len(chunks) != maxChunks) {
			t.Errorf("%s: %d chunks, want at most %d and exactly that many when truncated", c.name, len(chunks), maxChunks)
		}
		for i, ch := range chunks {
			if len(ch) > size {
				t.Errorf("%s: chunk %d is %d bytes, over the %d cap", c.name, i, len(ch), size)
				break
			}
			if !utf8.ValidString(ch) {
				t.Errorf("%s: chunk %d was cut inside a rune", c.name, i)
				break
			}
		}
	}
}
