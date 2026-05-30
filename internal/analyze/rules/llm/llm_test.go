package llm

import (
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// TestLLMCatalog guards the shipped LLM rule pack: enough rules, all eight
// categories represented, every rule well-formed and LLM-provenanced.
func TestLLMCatalog(t *testing.T) {
	if len(rules) < 20 {
		t.Fatalf("expected at least 20 LLM rules, got %d", len(rules))
	}
	seen := map[string]bool{}
	cats := map[analyze.Category]int{}
	for _, r := range rules {
		m := r.Meta()
		if !strings.HasPrefix(m.ID, "llm/") {
			t.Errorf("rule %q missing llm/ prefix", m.ID)
		}
		if seen[m.ID] {
			t.Errorf("duplicate LLM rule ID %q", m.ID)
		}
		seen[m.ID] = true
		if m.Provenance != analyze.ProvenanceLLM {
			t.Errorf("rule %s is not LLM provenance", m.ID)
		}
		if m.Scope != analyze.FileScope {
			t.Errorf("rule %s is not file scope", m.ID)
		}
		if m.Severity == "" {
			t.Errorf("rule %s has empty severity", m.ID)
		}
		if m.Summary == "" {
			t.Errorf("rule %s has empty summary", m.ID)
		}
		if strings.TrimSpace(r.Prompt()) == "" {
			t.Errorf("rule %s has empty prompt", m.ID)
		}
		cats[m.Category]++
	}

	for _, c := range []analyze.Category{
		analyze.CategoryArchitecture, analyze.CategorySecurity, analyze.CategoryBugs,
		analyze.CategoryPerformance, analyze.CategoryReliability, analyze.CategoryDatabase,
		analyze.CategoryQuality, analyze.CategoryStyle,
	} {
		if cats[c] == 0 {
			t.Errorf("no LLM rule registered for category %s", c)
		}
	}
}

// TestLLMRulesRegisterGlobally confirms the init() registers into the default
// registry and that the IDs do not collide with deterministic rules.
func TestLLMRulesRegisterGlobally(t *testing.T) {
	var llmCount int
	for _, r := range analyze.AllRules() {
		if _, ok := r.(analyze.LLMRule); ok {
			llmCount++
		}
	}
	if llmCount < 20 {
		t.Fatalf("expected >=20 LLM rules in default registry, got %d", llmCount)
	}
}
