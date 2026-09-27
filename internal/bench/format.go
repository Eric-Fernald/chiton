package bench

import (
	"fmt"
	"io"
	"slices"
)

// ratio renders grep+read cost relative to chiton cost as a human phrase.
func ratio(grep, chiton int) string {
	if chiton == 0 || grep == 0 {
		return "—"
	}
	r := float64(grep) / float64(chiton)
	if r >= 1 {
		return fmt.Sprintf("%.1f× cheaper", r)
	}
	return fmt.Sprintf("%.1f× costlier", 1/r)
}

// medianInt returns the median of xs (0 for empty). xs is copied before sorting.
func medianInt(xs []int) int {
	if len(xs) == 0 {
		return 0
	}
	c := slices.Clone(xs)
	slices.Sort(c)
	n := len(c)
	if n%2 == 1 {
		return c[n/2]
	}
	return (c[n/2-1] + c[n/2]) / 2
}

// FormatMarkdown writes one Markdown section per report (corpus shape, per-query token table,
// medians), ready to paste into docs/BENCHMARK.md.
func FormatMarkdown(w io.Writer, reps []*Report) {
	for _, rep := range reps {
		fmt.Fprintf(w, "### %s\n\n", rep.Name)
		fmt.Fprintf(w, "%d files · %d nodes · %d edges · ~%d tokens in scope · depth %d\n\n",
			rep.Files, rep.Nodes, rep.Edges, rep.CorpusTokens, rep.Depth)
		fmt.Fprintln(w, "| Question | naive | grep+read | chiton | chiton vs grep+read |")
		fmt.Fprintln(w, "|---|---:|---:|---:|---|")
		grepVals := make([]int, 0, len(rep.Queries))
		chitonVals := make([]int, 0, len(rep.Queries))
		for _, q := range rep.Queries {
			fmt.Fprintf(w, "| %s | %d | %d | %d | %s |\n",
				q.Question, q.Naive, q.GrepRead, q.Chiton, ratio(q.GrepRead, q.Chiton))
			grepVals = append(grepVals, q.GrepRead)
			chitonVals = append(chitonVals, q.Chiton)
		}
		mg, mc := medianInt(grepVals), medianInt(chitonVals)
		fmt.Fprintf(w, "\n**Median:** chiton ~%d tokens vs grep+read ~%d (%s); naive ~%d.\n\n",
			mc, mg, ratio(mg, mc), rep.CorpusTokens)
	}
}
