package chunk

import (
	"slices"
	"testing"
)

func TestParagraphs(t *testing.T) {
	cases := []struct {
		name      string
		text      string
		size, max int
		want      []string
	}{
		{"fits whole", " a\n\nb ", 10, 0, []string{" a\n\nb "}},
		{"packs paragraphs", "aa\n\nbb\n\ncc\n\ndd", 6, 0, []string{"aa\n\nbb", "cc\n\ndd"}},
		{"long paragraph kept whole", "aa\n\nbbbbbbbbbb\n\ncc", 4, 0, []string{"aa", "bbbbbbbbbb", "cc"}},
		{"cap absorbs rest", "aa\n\nbb\n\ncc\n\ndd", 2, 2, []string{"aa", "bb\n\ncc\n\ndd"}},
		{"blank text comes back whole", "     ", 2, 0, []string{"     "}},
	}
	for _, c := range cases {
		if got := Paragraphs(c.text, c.size, c.max); !slices.Equal(got, c.want) {
			t.Errorf("%s: Paragraphs = %q, want %q", c.name, got, c.want)
		}
	}
}
