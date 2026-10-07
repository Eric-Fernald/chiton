package main

import (
	"fmt"
	"io"
	"slices"
	"sort"
	"strings"
)

// WriteReport writes Markdown tables comparing the conditions: overall scores and costs, scores by question
// kind, and per-question F1.
func WriteReport(w io.Writer, set QuestionSet, runs []Run) {
	conds := conditionsOf(runs)
	byCond := map[string][]Run{}
	for _, r := range runs {
		byCond[r.Condition] = append(byCond[r.Condition], r)
	}
	model := ""
	for _, r := range runs {
		if r.Model != "" {
			model = r.Model
			break
		}
	}
	repeats := 0
	for _, r := range runs {
		repeats = max(repeats, r.Repeat)
	}
	fmt.Fprintf(w, "%d questions on %s@%s · model %s · %d run(s) per question per condition\n\n",
		len(set.Questions), set.Module, set.Commit, model, repeats)

	fmt.Fprintln(w, "| Condition | Mean F1 | Precision | Recall | Exact answers | Median input tokens | Median output tokens | Median turns | Median time | Total cost | Errors |")
	fmt.Fprintln(w, "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
	for _, c := range conds {
		rs := byCond[c]
		var f1, p, rc []float64
		var in, outT, turns, dur []int
		exact, errs := 0, 0
		cost := 0.0
		for _, r := range rs {
			cost += r.CostUSD
			if r.Error != "" {
				errs++
				f1, p, rc = append(f1, 0), append(p, 0), append(rc, 0)
				continue
			}
			f1, p, rc = append(f1, r.Score.F1), append(p, r.Score.Precision), append(rc, r.Score.Recall)
			if r.Score.Exact {
				exact++
			}
			in, outT, turns = append(in, r.TotalInput()), append(outT, r.OutputTokens), append(turns, r.NumTurns)
			dur = append(dur, int(r.DurationMS/1000))
		}
		fmt.Fprintf(w, "| %s | %.2f | %.2f | %.2f | %d / %d | %s | %s | %d | %ds | $%.2f | %d |\n",
			c, mean(f1), mean(p), mean(rc), exact, len(rs), thousands(median(in)), thousands(median(outT)), median(turns), median(dur), cost, errs)
	}

	fmt.Fprintln(w, "\nMean F1 by question kind:")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "| Kind | %s |\n", strings.Join(conds, " | "))
	fmt.Fprintf(w, "|---|%s\n", strings.Repeat("---:|", len(conds)))
	kindOf := map[string]string{}
	var kinds []string
	for _, q := range set.Questions {
		kindOf[q.ID] = q.Kind
		if !slices.Contains(kinds, q.Kind) {
			kinds = append(kinds, q.Kind)
		}
	}
	for _, k := range kinds {
		row := []string{}
		for _, c := range conds {
			var xs []float64
			for _, r := range byCond[c] {
				if kindOf[r.QuestionID] == k {
					xs = append(xs, r.Score.F1)
				}
			}
			row = append(row, fmt.Sprintf("%.2f", mean(xs)))
		}
		fmt.Fprintf(w, "| %s | %s |\n", k, strings.Join(row, " | "))
	}

	fmt.Fprintln(w, "\nPer question (mean F1 over repeats; expected answer size in brackets):")
	fmt.Fprintln(w)
	fmt.Fprintf(w, "| Question | Target | %s |\n", strings.Join(conds, " | "))
	fmt.Fprintf(w, "|---|---|%s\n", strings.Repeat("---:|", len(conds)))
	for _, q := range set.Questions {
		row := []string{}
		for _, c := range conds {
			var xs []float64
			for _, r := range byCond[c] {
				if r.QuestionID == q.ID {
					xs = append(xs, r.Score.F1)
				}
			}
			row = append(row, fmt.Sprintf("%.2f", mean(xs)))
		}
		fmt.Fprintf(w, "| %s | `%s` [%d] | %s |\n", q.ID, q.Target, len(q.Expected), strings.Join(row, " | "))
	}

	fmt.Fprintln(w, "\nTool calls per run (mean):")
	fmt.Fprintln(w)
	for _, c := range conds {
		totals := map[string]int{}
		for _, r := range byCond[c] {
			for t, n := range r.ToolCalls {
				totals[t] += n
			}
		}
		names := make([]string, 0, len(totals))
		for t := range totals {
			names = append(names, t)
		}
		sort.Strings(names)
		parts := []string{}
		for _, t := range names {
			parts = append(parts, fmt.Sprintf("%s %.1f", t, float64(totals[t])/float64(len(byCond[c]))))
		}
		blocked := 0
		for _, r := range byCond[c] {
			for _, n := range r.Blocked {
				blocked += n
			}
		}
		line := strings.Join(parts, ", ")
		if blocked > 0 {
			line += fmt.Sprintf(" (plus %.1f rejected calls to withheld tools)", float64(blocked)/float64(len(byCond[c])))
		}
		fmt.Fprintf(w, "- **%s:** %s\n", c, line)
	}
}

func conditionsOf(runs []Run) []string {
	var out []string
	for _, preferred := range []string{"grep", "chiton", "chiton-only"} {
		for _, r := range runs {
			if r.Condition == preferred {
				out = append(out, preferred)
				break
			}
		}
	}
	for _, r := range runs {
		if !slices.Contains(out, r.Condition) {
			out = append(out, r.Condition)
		}
	}
	return out
}

func mean(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	s := 0.0
	for _, x := range xs {
		s += x
	}
	return s / float64(len(xs))
}

func median(xs []int) int {
	if len(xs) == 0 {
		return 0
	}
	c := slices.Clone(xs)
	slices.Sort(c)
	if len(c)%2 == 1 {
		return c[len(c)/2]
	}
	return (c[len(c)/2-1] + c[len(c)/2]) / 2
}

func thousands(n int) string {
	s := fmt.Sprint(n)
	var b strings.Builder
	for i, ch := range s {
		if i > 0 && (len(s)-i)%3 == 0 {
			b.WriteByte(',')
		}
		b.WriteRune(ch)
	}
	return b.String()
}
