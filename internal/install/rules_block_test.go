package install

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeRules(t *testing.T, text string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "CLAUDE.md")
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatal(err)
	}
	return path
}

func readRules(t *testing.T, path string) string {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	return string(b)
}

// TestUpsertRulesBlockOrphanStartKeepsUserText covers a file whose end marker was deleted: repeated installs
// must keep the user's text after the orphan start marker and leave exactly one block.
func TestUpsertRulesBlockOrphanStartKeepsUserText(t *testing.T) {
	path := writeRules(t, "# Mine\n\n"+ChitonBlockStart+"\nold chiton text\n\nUSER NOTES\n")
	for i := range 3 {
		if _, _, err := UpsertRulesBlock(path, "fresh"); err != nil {
			t.Fatal(err)
		}
		got := readRules(t, path)
		if !strings.Contains(got, "USER NOTES") || !strings.Contains(got, "# Mine") {
			t.Fatalf("install %d dropped user text:\n%s", i+1, got)
		}
		if s, e := strings.Count(got, ChitonBlockStart), strings.Count(got, ChitonBlockEnd); s != 1 || e != 1 {
			t.Fatalf("install %d left %d start and %d end markers, want one block:\n%s", i+1, s, e, got)
		}
	}
}

// TestUpsertRulesBlockStrayEndDoesNotDuplicate covers a stray end marker ahead of the block: each install
// used to append another copy.
func TestUpsertRulesBlockStrayEndDoesNotDuplicate(t *testing.T) {
	path := writeRules(t, ChitonBlockEnd+"\n# Mine\n\n"+ChitonBlockStart+"\nold\n"+ChitonBlockEnd+"\n")
	for i := range 3 {
		if _, _, err := UpsertRulesBlock(path, "fresh"); err != nil {
			t.Fatal(err)
		}
		got := readRules(t, path)
		if s, e := strings.Count(got, ChitonBlockStart), strings.Count(got, ChitonBlockEnd); s != 1 || e != 1 {
			t.Fatalf("install %d left %d start and %d end markers, want one block:\n%s", i+1, s, e, got)
		}
		if !strings.Contains(got, "# Mine") || !strings.Contains(got, "fresh") || strings.Contains(got, "old") {
			t.Fatalf("install %d = %q, want the user heading and only the fresh block", i+1, got)
		}
	}
}

// TestRemoveRulesBlockOrphanStart checks uninstall drops an unpaired marker but keeps the text after it.
func TestRemoveRulesBlockOrphanStart(t *testing.T) {
	path := writeRules(t, "# Mine\n\n"+ChitonBlockStart+"\nUSER NOTES\n")
	changed, err := RemoveRulesBlock(path)
	if err != nil || !changed {
		t.Fatalf("RemoveRulesBlock = %v, %v", changed, err)
	}
	got := readRules(t, path)
	if strings.Contains(got, ChitonBlockStart) || !strings.Contains(got, "USER NOTES") || !strings.Contains(got, "# Mine") {
		t.Errorf("after uninstall = %q, want the marker gone and the user text kept", got)
	}
}
