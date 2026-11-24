package analyze_test

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
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

// docBait is prose that trips the secret, http-URL, SQL-concat and TODO rules wherever they run.
const docBait = "# Notes\n\nTODO: tidy up. See http://example.com/x and key AKIAIOSFODNN7EXAMPLE.\n\n" +
	"```go\nq := \"SELECT * FROM users WHERE id = \" + id\n```\n"

// TestFileRulesSkipDocsAndImages checks the tree scan, the staged-blob scan and the single-file pass all run
// FileRules on code only. Markdown is prose, and an image has no file URI for a finding to target.
func TestFileRulesSkipDocsAndImages(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	code := filepath.Join(root, "ok.go")
	paths := []string{code, filepath.Join(root, "README.md"), filepath.Join(root, "a.png"), filepath.Join(root, "b.png")}
	var sources []analyze.SourceFile
	for _, p := range paths {
		content := docBait
		if p == code {
			content = "package a\n\n// TODO: fix\nfunc A() {}\n"
		}
		write(t, p, content)
		sources = append(sources, analyze.SourceFile{Path: p, Content: []byte(content)})
	}

	scan, err := analyze.New(nil, nil).Scan(ctx, root)
	if err != nil {
		t.Fatal(err)
	}
	staged, err := analyze.New(nil, nil).ScanSources(ctx, sources)
	if err != nil {
		t.Fatal(err)
	}
	var single []analyze.Finding
	az := analyze.New(openStore(t), nil)
	for _, p := range paths {
		res, err := az.AnalyzeFile(ctx, p)
		if err != nil {
			t.Fatal(err)
		}
		single = append(single, res.Findings...)
	}

	codeURI := model.FileURI(code)
	for name, findings := range map[string][]analyze.Finding{"Scan": scan.Findings, "ScanSources": staged.Findings, "AnalyzeFile": single} {
		onCode := 0
		for _, f := range findings {
			if f.TargetURI == codeURI {
				onCode++
				continue
			}
			t.Errorf("%s: %s fired on a non-code file (target_uri %q): %s", name, f.RuleID, f.TargetURI, f.Message)
		}
		if onCode == 0 {
			t.Errorf("%s: no finding on ok.go, so the rules did not run at all", name)
		}
	}
}

// TestAnalyzePrunesStaleDocFindings checks a full analyze and a single-file pass both still treat a Markdown
// file as scanned, so a FileRule finding an older version left on it is pruned.
func TestAnalyzePrunesStaleDocFindings(t *testing.T) {
	ctx := context.Background()
	passes := map[string]func(az *analyze.Analyzer, root, readme string) error{
		"Analyze": func(az *analyze.Analyzer, root, _ string) error {
			_, err := az.Analyze(ctx, root)
			return err
		},
		"AnalyzeFile": func(az *analyze.Analyzer, _, readme string) error {
			_, err := az.AnalyzeFile(ctx, readme)
			return err
		},
	}
	for name, pass := range passes {
		t.Run(name, func(t *testing.T) {
			root := t.TempDir()
			readme := filepath.Join(root, "README.md")
			write(t, readme, docBait)
			gs := openStore(t)
			id := "stalefinding0000000000000000cc01"
			stale := model.Node{
				ID: id, Kind: model.KindFinding, Label: todoRule, SourceURI: "finding://" + id,
				Provenance: model.ProvExtracted, Confidence: 1,
				Attrs: map[string]any{
					"rule_id": todoRule, "message": "TODO", "target_uri": model.FileURI(readme),
					"rule_provenance": string(analyze.ProvenanceDeterministic),
				},
			}
			if err := gs.UpsertNodes(ctx, []model.Node{stale}); err != nil {
				t.Fatal(err)
			}
			if err := pass(analyze.New(gs, nil), root, readme); err != nil {
				t.Fatal(err)
			}
			if views, _ := analyze.List(ctx, gs, analyze.ListOptions{}); len(views) != 0 {
				t.Errorf("findings left on a docs-only tree: %+v", views)
			}
		})
	}
}

// TestUnparsableFileIsReportedAsSkipped checks a file the extractor rejects is listed in Skipped as
// "path: reason" by every pass, is not counted as scanned, and keeps the findings recorded before it broke.
func TestUnparsableFileIsReportedAsSkipped(t *testing.T) {
	ctx := context.Background()
	root := t.TempDir()
	bad := filepath.Join(root, "bad.go")
	write(t, filepath.Join(root, "ok.go"), "package a\n")
	write(t, bad, "package a\n\n// TODO: fix\nfunc A() {}\n")
	gs := openStore(t)
	az := analyze.New(gs, nil)
	if _, err := az.Analyze(ctx, root); err != nil {
		t.Fatal(err)
	}
	if got := countFindings(t, gs, model.FileURI(bad), todoRule); got != 1 {
		t.Fatalf("setup: want 1 TODO finding on bad.go, got %d", got)
	}

	broken := "package a\n\n// TODO: fix\nfunc {\n"
	write(t, bad, broken)
	res, err := az.Analyze(ctx, root)
	if err != nil {
		t.Fatal(err)
	}
	if res.FilesScanned != 1 || len(res.Skipped) != 1 {
		t.Fatalf("FilesScanned=%d Skipped=%v, want ok.go scanned and bad.go skipped", res.FilesScanned, res.Skipped)
	}
	note := res.Skipped[0]
	if !strings.HasPrefix(note, bad+": ") || strings.Count(note, bad) != 1 || !strings.Contains(note, "expected") {
		t.Errorf("Skipped entry = %q, want %q followed once by the parse error", note, bad+": ")
	}
	if got := countFindings(t, gs, model.FileURI(bad), todoRule); got != 1 {
		t.Errorf("an unparsable file's earlier findings were pruned: %d left, want 1", got)
	}

	single, err := az.AnalyzeFile(ctx, bad)
	if err != nil {
		t.Fatal(err)
	}
	staged, err := az.ScanSources(ctx, []analyze.SourceFile{{Path: bad, Content: []byte(broken)}})
	if err != nil {
		t.Fatal(err)
	}
	for name, got := range map[string][]string{"AnalyzeFile": single.Skipped, "ScanSources": staged.Skipped} {
		if len(got) != 1 || got[0] != note {
			t.Errorf("%s Skipped = %q, want [%q]", name, got, note)
		}
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
