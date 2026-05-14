package analyze

import (
	"fmt"
	"sort"
	"sync"
)

// Registry holds all rules known to the analyzer. Rules register at package
// init() time via Register(). The default registry is the package-level
// `defaultRegistry`; tests can construct private registries to avoid cross-test
// contamination.
type Registry struct {
	mu    sync.RWMutex
	rules map[string]Rule
}

// NewRegistry returns an empty registry.
func NewRegistry() *Registry {
	return &Registry{rules: map[string]Rule{}}
}

// Register adds a rule to the registry. Panics on duplicate ID — duplicate
// rules indicate a programming error and should never happen in production.
func (r *Registry) Register(rule Rule) {
	r.mu.Lock()
	defer r.mu.Unlock()
	id := rule.Meta().ID
	if id == "" {
		panic("analyze: rule has empty ID")
	}
	if _, dup := r.rules[id]; dup {
		panic(fmt.Sprintf("analyze: duplicate rule ID %q", id))
	}
	r.rules[id] = rule
}

// All returns every registered rule, ordered by ID for deterministic output.
func (r *Registry) All() []Rule {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]Rule, 0, len(r.rules))
	for _, rule := range r.rules {
		out = append(out, rule)
	}
	sort.Slice(out, func(i, j int) bool {
		return out[i].Meta().ID < out[j].Meta().ID
	})
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

// DefaultRegistry returns the package-level registry. Exposed so callers can
// pass it into NewAnalyzer without importing internal state.
func DefaultRegistry() *Registry { return defaultRegistry }
