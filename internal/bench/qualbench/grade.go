package main

import (
	"encoding/json"
	"strings"
)

// Score is how one answer compares with the expected set.
type Score struct {
	Precision  float64  `json:"precision"`
	Recall     float64  `json:"recall"`
	F1         float64  `json:"f1"`
	Exact      bool     `json:"exact"`
	ParseError bool     `json:"parse_error"`
	Missing    []string `json:"missing,omitempty"`
	Extra      []string `json:"extra,omitempty"`
}

// ParseAnswer pulls the last JSON array of strings out of a model reply, tolerating prose or a code fence around
// it. A reply that drafts an answer, corrects itself and gives a final array is graded on the final array.
func ParseAnswer(reply string) ([]string, bool) {
	end := strings.LastIndex(reply, "]")
	if end < 0 {
		return nil, false
	}
	for start := strings.LastIndex(reply[:end], "["); start >= 0; start = strings.LastIndex(reply[:start], "[") {
		var out []string
		if json.Unmarshal([]byte(reply[start:end+1]), &out) == nil {
			return out, true
		}
	}
	return nil, false
}

// normalize forgives formatting that does not change the meaning of an answer: surrounding space and
// backticks, backslashes, a leading "./", the module path prefix, a "()" suffix, and a "(*T)." or "(T)." receiver.
func normalize(s, module string) string {
	s = strings.TrimSpace(strings.Trim(strings.TrimSpace(s), "`"))
	s = strings.ReplaceAll(s, `\`, "/")
	s = strings.TrimPrefix(s, module+"/")
	if s == module {
		s = "."
	}
	if rest, ok := strings.CutPrefix(s, "./"); ok && rest != "" {
		s = rest
	}
	s = strings.TrimSuffix(s, "()")
	if dir, name, ok := strings.Cut(s, ":"); ok {
		name = strings.TrimPrefix(name, "(*")
		name = strings.TrimPrefix(name, "(")
		name = strings.Replace(name, ").", ".", 1)
		s = dir + ":" + name
	}
	return s
}

// Grade scores reply against expected. Precision and recall are 1 for a correct empty answer.
func Grade(reply string, expected []string, module string) Score {
	got, ok := ParseAnswer(reply)
	if !ok {
		return Score{ParseError: true, Missing: expected}
	}
	want := map[string]bool{}
	for _, e := range expected {
		want[normalize(e, module)] = true
	}
	have := map[string]bool{}
	for _, g := range got {
		if n := normalize(g, module); n != "" {
			have[n] = true
		}
	}
	var sc Score
	hits := 0
	for h := range have {
		if want[h] {
			hits++
		} else {
			sc.Extra = append(sc.Extra, h)
		}
	}
	for w := range want {
		if !have[w] {
			sc.Missing = append(sc.Missing, w)
		}
	}
	switch {
	case len(have) == 0 && len(want) == 0:
		sc.Precision, sc.Recall = 1, 1
	default:
		if len(have) > 0 {
			sc.Precision = float64(hits) / float64(len(have))
		}
		if len(want) > 0 {
			sc.Recall = float64(hits) / float64(len(want))
		}
	}
	if sc.Precision+sc.Recall > 0 {
		sc.F1 = 2 * sc.Precision * sc.Recall / (sc.Precision + sc.Recall)
	}
	sc.Exact = len(sc.Missing) == 0 && len(sc.Extra) == 0
	return sc
}
