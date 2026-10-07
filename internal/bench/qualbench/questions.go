package main

import (
	"fmt"
	"sort"
	"strings"
)

// Question is one benchmark question with its ground-truth answer.
type Question struct {
	ID       string   `json:"id"`
	Kind     string   `json:"kind"` // callers, callees, imports, importers
	Target   string   `json:"target"`
	Prompt   string   `json:"prompt"`
	Expected []string `json:"expected"`
}

// QuestionSet is the generated question file: the questions plus where and how they were made.
type QuestionSet struct {
	Module    string     `json:"module"`
	Commit    string     `json:"commit"`
	Questions []Question `json:"questions"`
}

// How many questions of each kind to pick, and the answer-size window that keeps them non-trivial (a single
// grep hit) but checkable (not a 40-item list).
var plan = []struct {
	kind     string
	count    int
	min, max int
}{
	{"callers", 6, 3, 10},
	{"callees", 4, 3, 10},
	{"imports", 4, 3, 10},
	{"importers", 4, 3, 12},
}

const answerFormat = `Reply with ONLY a JSON array of strings and nothing else. Write each function as "<dir>:<Func>" and each method as "<dir>:<Type>.<Method>", and each package as "<dir>", where <dir> is the package's directory relative to the repository root with forward slashes ("." for the root). Example: ["internal/mcp:queryHandler", "cmd/chiton:Server.Run", "internal/graph/store"]. Reply [] if the answer is empty.`

const scope = "Consider every non-test Go file in the repository (files not ending in _test.go), whatever their build tags or target OS."

// Generate picks questions deterministically from the facts: for each kind, the candidates inside the size
// window, sorted by target, sampled at even intervals.
func Generate(f *Facts) []Question {
	var qs []Question
	for _, p := range plan {
		var cands []Question
		switch p.kind {
		case "callers":
			for _, fn := range f.uniqueFuncs() {
				exp := f.Callers(fn)
				if inWindow(exp, p.min, p.max) && spansPackages(exp) {
					cands = append(cands, Question{Kind: p.kind, Target: fn, Expected: exp, Prompt: fmt.Sprintf(
						"Which functions and methods call the package-level function `%s` directly? A call inside a closure counts for the named function or method that contains it. A function value used without calling it does not count. %s\n\n%s",
						fn, scope, answerFormat)})
				}
			}
		case "callees":
			for _, fn := range f.uniqueFuncs() {
				exp := f.Callees(fn)
				if inWindow(exp, p.min, p.max) && spansPackages(exp) {
					cands = append(cands, Question{Kind: p.kind, Target: fn, Expected: exp, Prompt: fmt.Sprintf(
						"Which package-level functions defined in this repository does `%s` call directly, including calls inside closures in its body? Leave out methods, builtins, and functions from the standard library or other modules. %s\n\n%s",
						fn, scope, answerFormat)})
				}
			}
		case "imports":
			for _, dir := range sortedPackages(f) {
				exp := f.ImportsOf(dir)
				if inWindow(exp, p.min, p.max) {
					cands = append(cands, Question{Kind: p.kind, Target: dir, Expected: exp, Prompt: fmt.Sprintf(
						"Which packages from this repository (not the standard library or third-party modules) does package `%s` import? %s\n\n%s",
						dir, scope, answerFormat)})
				}
			}
		case "importers":
			for _, dir := range sortedPackages(f) {
				exp := f.Importers(dir)
				if inWindow(exp, p.min, p.max) {
					cands = append(cands, Question{Kind: p.kind, Target: dir, Expected: exp, Prompt: fmt.Sprintf(
						"Which packages in this repository import package `%s`? %s\n\n%s",
						dir, scope, answerFormat)})
				}
			}
		}
		for i, q := range sample(cands, p.count) {
			q.ID = fmt.Sprintf("%s-%d", p.kind, i+1)
			qs = append(qs, q)
		}
	}
	return qs
}

// hardPlan is the multi-hop question set: every answer needs a chain of lookups, not one search.
var hardPlan = []struct {
	kind     string
	hops     int // call chain length; 0 for the transitive import kinds
	count    int
	min, max int
}{
	{"callers", 2, 6, 8, 30},
	{"callers", 3, 4, 10, 35},
	{"callees", 2, 4, 8, 30},
	{"imports-all", 0, 3, 6, 25},
	{"importers-all", 0, 3, 6, 30},
}

// GenerateHard picks multi-hop questions deterministically. Caller chains only pass through package-level
// functions (a method's callers can't be resolved from syntax alone), so for a chain of n calls the targets are
// those whose callers at every step before the last are all package-level functions.
func GenerateHard(f *Facts) []Question {
	var qs []Question
	for _, p := range hardPlan {
		var cands []Question
		switch p.kind {
		case "callers":
			for _, fn := range f.uniqueFuncs() {
				exp, ok := f.transitiveCallers(fn, p.hops)
				if ok && inWindow(exp, p.min, p.max) && packagesIn(exp) >= 3 {
					cands = append(cands, Question{Kind: fmt.Sprintf("callers-%d", p.hops), Target: fn, Expected: exp, Prompt: fmt.Sprintf(
						"List every function and method that reaches the package-level function `%s` through a chain of at most %d calls. "+
							"A chain of 1 is a direct caller of `%[1]s`; a chain of 2 is a direct caller of one of those; and so on. "+
							"Count only calls that name a package-level function directly, as `f(...)` or `pkg.F(...)`. "+
							"A call inside a closure counts for the named function or method that contains it. List each function once, and leave out `%[1]s` itself. %[3]s\n\n%[4]s",
						fn, p.hops, scope, answerFormat)})
				}
			}
		case "callees":
			for _, fn := range f.uniqueFuncs() {
				exp := f.transitiveCallees(fn, p.hops)
				if inWindow(exp, p.min, p.max) && packagesIn(exp) >= 3 {
					cands = append(cands, Question{Kind: fmt.Sprintf("callees-%d", p.hops), Target: fn, Expected: exp, Prompt: fmt.Sprintf(
						"List every package-level function defined in this repository that `%s` reaches through a chain of at most %d direct calls to package-level functions. "+
							"A chain of 1 is a function `%[1]s` calls directly; a chain of 2 is a function one of those calls directly. "+
							"Calls inside closures count for the function that contains them. Leave out methods, builtins, functions from the standard library or other modules, and `%[1]s` itself. %[3]s\n\n%[4]s",
						fn, p.hops, scope, answerFormat)})
				}
			}
		case "imports-all":
			for _, dir := range sortedPackages(f) {
				exp := f.closure(dir, f.ImportsOf)
				if inWindow(exp, p.min, p.max) {
					cands = append(cands, Question{Kind: p.kind, Target: dir, Expected: exp, Prompt: fmt.Sprintf(
						"List every package from this repository that package `%s` depends on, directly or through other packages of this repository it imports (not the standard library or third-party modules). %s\n\n%s",
						dir, scope, answerFormat)})
				}
			}
		case "importers-all":
			for _, dir := range sortedPackages(f) {
				exp := f.closure(dir, f.Importers)
				if inWindow(exp, p.min, p.max) {
					cands = append(cands, Question{Kind: p.kind, Target: dir, Expected: exp, Prompt: fmt.Sprintf(
						"List every package in this repository that depends on package `%s`, directly or through other packages that import it. %s\n\n%s",
						dir, scope, answerFormat)})
				}
			}
		}
		for i, q := range sample(cands, p.count) {
			q.ID = fmt.Sprintf("%s-%d", q.Kind, i+1)
			qs = append(qs, q)
		}
	}
	return qs
}

// transitiveCallers returns every declaration within hops calls of fn, and false when a chain would have to
// continue through a method, whose own callers can't be resolved.
func (f *Facts) transitiveCallers(fn string, hops int) ([]string, bool) {
	seen := map[string]bool{fn: true}
	frontier := []string{fn}
	var out []string
	for step := 1; step <= hops && len(frontier) > 0; step++ {
		var next []string
		for _, x := range frontier {
			for _, c := range f.Callers(x) {
				if seen[c] {
					continue
				}
				seen[c] = true
				out = append(out, c)
				if step < hops {
					if strings.Contains(c[strings.Index(c, ":")+1:], ".") {
						return nil, false // a method in the middle of a chain
					}
					next = append(next, c)
				}
			}
		}
		frontier = next
	}
	sort.Strings(out)
	return out, true
}

// transitiveCallees returns every module package-level function within hops calls of fn.
func (f *Facts) transitiveCallees(fn string, hops int) []string {
	seen := map[string]bool{fn: true}
	frontier := []string{fn}
	var out []string
	for step := 1; step <= hops && len(frontier) > 0; step++ {
		var next []string
		for _, x := range frontier {
			for _, c := range f.Callees(x) {
				if !seen[c] {
					seen[c] = true
					out = append(out, c)
					next = append(next, c)
				}
			}
		}
		frontier = next
	}
	sort.Strings(out)
	return out
}

// closure returns every package reachable from dir by repeatedly applying step, excluding dir itself.
func (f *Facts) closure(dir string, step func(string) []string) []string {
	seen := map[string]bool{dir: true}
	frontier := []string{dir}
	var out []string
	for len(frontier) > 0 {
		var next []string
		for _, x := range frontier {
			for _, y := range step(x) {
				if !seen[y] {
					seen[y] = true
					out = append(out, y)
					next = append(next, y)
				}
			}
		}
		frontier = next
	}
	sort.Strings(out)
	return out
}

// packagesIn counts the distinct packages an answer reaches into.
func packagesIn(xs []string) int {
	seen := map[string]bool{}
	for _, x := range xs {
		dir, _, _ := strings.Cut(x, ":")
		seen[dir] = true
	}
	return len(seen)
}

// uniqueFuncs lists package-level functions ("dir:Func") defined in exactly one file, so build-tag stubs that
// define the same name twice never become targets. Sorted.
func (f *Facts) uniqueFuncs() []string {
	var out []string
	for dir, fns := range f.Funcs {
		for name, files := range fns {
			if len(files) == 1 && name != "init" && name != "main" {
				out = append(out, dir+":"+name)
			}
		}
	}
	sort.Strings(out)
	return out
}

func sortedPackages(f *Facts) []string {
	out := make([]string, 0, len(f.Packages))
	for dir := range f.Packages {
		out = append(out, dir)
	}
	sort.Strings(out)
	return out
}

func inWindow(xs []string, lo, hi int) bool { return len(xs) >= lo && len(xs) <= hi }

// spansPackages reports whether the answer reaches into at least two packages, so it cannot be read off one file.
func spansPackages(xs []string) bool {
	seen := map[string]bool{}
	for _, x := range xs {
		dir, _, _ := strings.Cut(x, ":")
		seen[dir] = true
	}
	return len(seen) >= 2
}

// sample takes n items at even intervals across xs (all of them when there are fewer).
func sample(xs []Question, n int) []Question {
	if len(xs) <= n {
		return xs
	}
	out := make([]Question, n)
	for i := range n {
		out[i] = xs[i*len(xs)/n]
	}
	return out
}
