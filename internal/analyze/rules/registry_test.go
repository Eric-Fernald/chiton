package rules

import (
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// TestFileRulesDeclareLanguages checks every FileRule names the languages it reads, so none is left to match
// whatever else has an extractor (Markdown prose, image bytes).
func TestFileRulesDeclareLanguages(t *testing.T) {
	for _, r := range analyze.AllRules() {
		if _, ok := r.(analyze.FileRule); !ok {
			continue
		}
		if m := r.Meta(); len(m.Languages) == 0 {
			t.Errorf("rule %s declares no Languages", m.ID)
		}
	}
}

// TestRegistryPopulated guards against accidentally dropping a category from
// the barrel import or registering two rules with the same ID.
func TestRegistryPopulated(t *testing.T) {
	all := analyze.AllRules()
	if len(all) < 15 {
		t.Errorf("expected at least 15 rules registered, got %d", len(all))
	}
	seen := map[string]bool{}
	categories := map[analyze.Category]int{}
	for _, r := range all {
		m := r.Meta()
		if seen[m.ID] {
			t.Errorf("duplicate rule ID %q", m.ID)
		}
		seen[m.ID] = true
		categories[m.Category]++
		if m.Severity == "" {
			t.Errorf("rule %s has empty severity", m.ID)
		}
		if m.Summary == "" {
			t.Errorf("rule %s has empty summary", m.ID)
		}
	}
	// Every one of the eight categories must have at least one rule.
	wanted := []analyze.Category{
		analyze.CategoryArchitecture, analyze.CategorySecurity,
		analyze.CategoryBugs, analyze.CategoryQuality,
		analyze.CategoryPerformance, analyze.CategoryReliability,
		analyze.CategoryDatabase, analyze.CategoryStyle,
	}
	for _, c := range wanted {
		if categories[c] == 0 {
			t.Errorf("no rules registered for category %s", c)
		}
	}
}
