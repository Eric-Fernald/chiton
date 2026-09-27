package analyze

import (
	"bufio"
	"bytes"
	"slices"
	"strings"
	"testing"
)

// TestLinesMatchesScanner pins Lines to the bufio.Scanner setup the line-based rules replaced.
func TestLinesMatchesScanner(t *testing.T) {
	huge := strings.Repeat("x", maxLineBytes)
	inputs := []string{
		"", "\n", "a", "a\n", "a\nb", "a\r\nb\r\n", "a\n\n", "\r", "x\r\r\n",
		"a\n" + huge[:maxLineBytes-1] + "\nb\n",
		"a\n" + huge[:maxLineBytes-1] + "\r\nb\n",
		"a\n" + huge + "\nb\n",
		"a\n" + huge[:maxLineBytes-1],
		"a\n" + huge,
	}
	for i, in := range inputs {
		var want []string
		sc := bufio.NewScanner(bytes.NewReader([]byte(in)))
		sc.Buffer(make([]byte, 0, 64*1024), maxLineBytes)
		for sc.Scan() {
			want = append(want, sc.Text())
		}
		var got []string
		for n, line := range Lines([]byte(in)) {
			if n != len(got)+1 {
				t.Fatalf("input %d: line number %d, want %d", i, n, len(got)+1)
			}
			got = append(got, line)
		}
		if !slices.Equal(got, want) {
			t.Errorf("input %d: got %d lines, want %d", i, len(got), len(want))
		}
	}
}

// TestCached checks that loaded targets compute each key once while hand-built ones always recompute.
func TestCached(t *testing.T) {
	calls := 0
	fn := func() int { calls++; return calls }
	var bare FileTarget
	if Cached(bare, "k", fn) != 1 || Cached(bare, "k", fn) != 2 {
		t.Fatalf("hand-built target should not memoize (calls=%d)", calls)
	}
	loaded := FileTarget{memo: &fileMemo{m: map[string]any{}}}
	copyOf := loaded // rules receive FileTarget by value; copies must share the memo
	if Cached(loaded, "k", fn) != 3 || Cached(copyOf, "k", fn) != 3 || Cached(loaded, "other", fn) != 4 {
		t.Fatalf("loaded target should memoize per key (calls=%d)", calls)
	}
}
