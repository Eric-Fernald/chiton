package analyze_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
)

// TestScanSkipsUnreadableDir checks one unreadable directory is reported in Skipped instead of aborting
// the whole scan. chmod cannot deny directory reads on Windows or to root, so the test skips there.
func TestScanSkipsUnreadableDir(t *testing.T) {
	if runtime.GOOS == "windows" || os.Geteuid() == 0 {
		t.Skip("needs POSIX permissions and a non-root user")
	}
	root := t.TempDir()
	write(t, filepath.Join(root, "ok.go"), "package a\n\n// TODO: fix\nfunc A() {}\n")
	locked := filepath.Join(root, "locked")
	if err := os.Mkdir(locked, 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(locked, "b.go"), "package b\n")
	if err := os.Chmod(locked, 0o000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { os.Chmod(locked, 0o755) })

	res, err := analyze.New(nil, nil).Scan(context.Background(), root)
	if err != nil {
		t.Fatalf("Scan aborted on an unreadable dir: %v", err)
	}
	if res.FilesScanned != 1 || len(res.Skipped) != 1 || res.Skipped[0] != locked {
		t.Errorf("FilesScanned=%d Skipped=%v, want 1 file and %s skipped", res.FilesScanned, res.Skipped, locked)
	}
}

// TestScanHonoursChitonignore checks the analyze walk uses the shared ignore policy, .chitonignore included.
func TestScanHonoursChitonignore(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "a.go"), "package a\n")
	if err := os.Mkdir(filepath.Join(root, "gen"), 0o755); err != nil {
		t.Fatal(err)
	}
	write(t, filepath.Join(root, "gen", "b.go"), "package gen\n")
	write(t, filepath.Join(root, ".chitonignore"), "gen/\n")

	res, err := analyze.New(nil, nil).Scan(context.Background(), root)
	if err != nil {
		t.Fatal(err)
	}
	if res.FilesScanned != 1 {
		t.Errorf("FilesScanned=%d, want 1 (gen/ is in .chitonignore)", res.FilesScanned)
	}
}

// TestScanStopsWhenContextDone checks a done context ends the walk with ctx's error.
func TestScanStopsWhenContextDone(t *testing.T) {
	root := t.TempDir()
	write(t, filepath.Join(root, "a.go"), "package a\n")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := analyze.New(nil, nil).Scan(ctx, root); !errors.Is(err, context.Canceled) {
		t.Fatalf("Scan with a cancelled context = %v, want context.Canceled", err)
	}
}

// TestScanSourcesSkipsGraphRules checks the in-memory scan runs file rules only and never needs a store.
func TestScanSourcesSkipsGraphRules(t *testing.T) {
	dir := t.TempDir()
	res, err := analyze.New(nil, nil).ScanSources(context.Background(), []analyze.SourceFile{
		{Path: filepath.Join(dir, "a.go"), Content: []byte("package a\n\n// TODO: fix\nfunc helper() {}\n")},
		{Path: filepath.Join(dir, "notes.txt"), Content: []byte("TODO: not source\n")},
	})
	if err != nil {
		t.Fatal(err)
	}
	var todo bool
	for _, f := range res.Findings {
		todo = todo || f.RuleID == todoRule
		if f.RuleID == "architecture/dead-symbol" {
			t.Errorf("GraphScope rule ran on a partial file set: %+v", f)
		}
	}
	if !todo || res.FilesScanned != 1 {
		t.Errorf("FilesScanned=%d findings=%+v, want 1 file with a TODO finding", res.FilesScanned, res.Findings)
	}
}
