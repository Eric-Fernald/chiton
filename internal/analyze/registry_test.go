package analyze

import (
	"context"
	"testing"
)

type scopeRule struct{ scope Scope }

func (r scopeRule) Meta() Meta { return Meta{ID: "test/scope", Scope: r.scope} }

type scopeFileRule struct{ scopeRule }

func (scopeFileRule) MatchFile(context.Context, FileTarget) []Finding { return nil }

type scopeGraphRule struct{ scopeRule }

func (scopeGraphRule) MatchGraph(context.Context, GraphTarget) []Finding { return nil }

type scopeLLMRule struct{ scopeRule }

func (scopeLLMRule) Prompt() string { return "p" }

type scopeBothRule struct{ scopeFileRule }

func (scopeBothRule) MatchGraph(context.Context, GraphTarget) []Finding { return nil }

// TestRegisterScope checks that Register panics when Meta.Scope disagrees with the implemented interface.
func TestRegisterScope(t *testing.T) {
	cases := []struct {
		name  string
		rule  Rule
		panic bool
	}{
		{"file ok", scopeFileRule{scopeRule{FileScope}}, false},
		{"graph ok", scopeGraphRule{scopeRule{GraphScope}}, false},
		{"llm ok", scopeLLMRule{scopeRule{FileScope}}, false},
		{"file as graph", scopeFileRule{scopeRule{GraphScope}}, true},
		{"graph as file", scopeGraphRule{scopeRule{FileScope}}, true},
		{"llm as graph", scopeLLMRule{scopeRule{GraphScope}}, true},
		{"file and graph", scopeBothRule{scopeFileRule{scopeRule{FileScope}}}, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if got := recover() != nil; got != c.panic {
					t.Errorf("panicked=%v, want %v", got, c.panic)
				}
			}()
			NewRegistry().Register(c.rule)
		})
	}
}
