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
		targets := "test symbols excluded"
		if rep.IncludeTests {
			targets = "test symbols included"
		}
		fmt.Fprintf(w, "### %s\n\n", rep.Name)
		fmt.Fprintf(w, "%d files · %d nodes · %d edges · ~%d tokens in scope · %s\n\n",
			rep.Files, rep.Nodes, rep.Edges, rep.CorpusTokens, targets)
		fmt.Fprintln(w, "| Question | naive | grep+read | chiton d2 | chiton d1 | query_graph | d2 vs grep+read | d1 vs grep+read |")
		fmt.Fprintln(w, "|---|---:|---:|---:|---:|---:|---|---|")
		var grepVals, d2Vals, d1Vals, qgVals []int
		for _, q := range rep.Queries {
			fmt.Fprintf(w, "| %s | %d | %d | %d | %d | %d | %s | %s |\n",
				q.Question, q.Naive, q.GrepRead, q.Chiton, q.ChitonDepth1, q.QueryGraph,
				ratio(q.GrepRead, q.Chiton), ratio(q.GrepRead, q.ChitonDepth1))
			grepVals = append(grepVals, q.GrepRead)
			d2Vals = append(d2Vals, q.Chiton)
			d1Vals = append(d1Vals, q.ChitonDepth1)
			qgVals = append(qgVals, q.QueryGraph)
		}
		mg, m2, m1, mq := medianInt(grepVals), medianInt(d2Vals), medianInt(d1Vals), medianInt(qgVals)
		fmt.Fprintf(w, "\n**Median:** grep+read ~%d tokens; chiton depth 2 ~%d (%s), depth 1 ~%d (%s), query_graph ~%d; naive ~%d.\n\n",
			mg, m2, ratio(mg, m2), m1, ratio(mg, m1), mq, rep.CorpusTokens)
	}
}
