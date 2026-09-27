package analyze

import (
	"fmt"
	"maps"
	"slices"
	"sync"
)

// Registry holds the rules known to an analyzer. Rule packages populate the default one from init().
// Tests build private registries so they don't share rules with other tests.
type Registry struct {
	mu    sync.RWMutex
	rules map[string]Rule
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{rules: map[string]Rule{}}
}

// Register adds a rule to the registry. It panics on a duplicate ID or on a Meta.Scope that disagrees
// with the implemented interface, since both are programming errors that must not reach production.
func (r *Registry) Register(rule Rule) {
	r.mu.Lock()
	defer r.mu.Unlock()
	m := rule.Meta()
	id := m.ID
	if id == "" {
		panic("analyze: rule has empty ID")
	}
	if _, dup := r.rules[id]; dup {
		panic(fmt.Sprintf("analyze: duplicate rule ID %q", id))
	}
	if err := checkScope(rule, m.Scope); err != "" {
		panic(fmt.Sprintf("analyze: rule %q %s", id, err))
	}
	r.rules[id] = rule
}

// checkScope enforces rule.go's contract: FileRule and LLMRule are FileScope, GraphRule is GraphScope,
// and a rule implements at most one of them. It returns "" when rule complies.
func checkScope(rule Rule, scope Scope) string {
	_, isFile := rule.(FileRule)
	_, isGraph := rule.(GraphRule)
	_, isLLM := rule.(LLMRule)
	switch {
	case isFile && isGraph, isLLM && (isFile || isGraph):
		return "implements more than one of FileRule, GraphRule, LLMRule"
	case isGraph && scope != GraphScope:
		return "implements GraphRule but Meta.Scope is not GraphScope"
	case (isFile || isLLM) && scope != FileScope:
		return "implements FileRule/LLMRule but Meta.Scope is not FileScope"
	}
	return ""
}

// All returns every registered rule, ordered by ID for deterministic output.
func (r *Registry) All() []Rule {
	r.mu.RLock()
	defer r.mu.RUnlock()
	ids := slices.Sorted(maps.Keys(r.rules))
	out := make([]Rule, len(ids))
	for i, id := range ids {
		out[i] = r.rules[id]
	}
	return out
}

// Get returns the rule with the given ID, or nil if not registered.
func (r *Registry) Get(id string) Rule {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.rules[id]
}

// defaultRegistry is the global registry rule packages populate via init().
var defaultRegistry = NewRegistry()

// Register adds a rule to the default registry.
func Register(rule Rule) { defaultRegistry.Register(rule) }

// AllRules returns every rule in the default registry.
func AllRules() []Rule { return defaultRegistry.All() }

// DefaultRegistry returns the package-level registry, for example to pass to New or NewLLMService.
func DefaultRegistry() *Registry { return defaultRegistry }
