package bench

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// GenerateSynthetic writes a reproducible synthetic Go corpus of `files` files
// into dir, so the small end of the benchmark is identical on every machine. It
// is a single package whose functions call each other in a ring (yielding CALLS
// edges) and import a few stdlib packages (yielding IMPORTS edges), producing a
// connected graph of realistic shape.
func GenerateSynthetic(dir string, files int) error {
	if files < 2 {
		files = 2
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return fmt.Errorf("mkdir %s: %w", dir, err)
	}
	for i := 0; i < files; i++ {
		next := (i + 1) % files
		var b strings.Builder
		fmt.Fprintf(&b, "package synth\n\n")
		fmt.Fprintf(&b, "import (\n\t\"fmt\"\n\t\"strings\"\n)\n\n")
		// A couple of functions per file; Func<i> calls into the next file to
		// form a ring, and a helper that exercises an imported package.
		fmt.Fprintf(&b, "// Func%d is a synthetic node in the benchmark ring.\n", i)
		fmt.Fprintf(&b, "func Func%d(in string) string {\n", i)
		fmt.Fprintf(&b, "\tout := strings.ToUpper(in)\n")
		fmt.Fprintf(&b, "\tfmt.Println(out)\n")
		fmt.Fprintf(&b, "\treturn Func%d(out)\n", next)
		fmt.Fprintf(&b, "}\n\n")
		fmt.Fprintf(&b, "func helper%d(n int) int {\n\treturn n * %d\n}\n", i, i+1)

		path := filepath.Join(dir, fmt.Sprintf("synth_%03d.go", i))
		if err := os.WriteFile(path, []byte(b.String()), 0o644); err != nil {
			return fmt.Errorf("write %s: %w", path, err)
		}
	}
	return nil
}
