package database

import (
	"context"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// dbTarget builds a FileTarget for a rule test. The URI extension drives the
// per-language behaviour in stripDBComment and SQLFormatInjection.
func dbTarget(path, content string) analyze.FileTarget {
	uri := "file://test/" + path
	return analyze.FileTarget{
		Path:     path,
		URI:      uri,
		Content:  []byte(content),
		FileNode: model.Node{ID: "f", Kind: model.KindFile, Label: path, SourceURI: uri},
	}
}

func TestSelectStar(t *testing.T) {
	pos := dbTarget("q.go", "q := `SELECT * FROM users WHERE id = ?`\n")
	if got := (SelectStar{}).MatchFile(context.Background(), pos); len(got) != 1 {
		t.Fatalf("positive: expected 1 finding, got %d (%+v)", len(got), got)
	}
	// count(*) and explicit columns must not fire.
	neg := dbTarget("q.go", "a := \"SELECT count(*) FROM users\"\nb := \"SELECT id, name FROM users\"\n")
	if got := (SelectStar{}).MatchFile(context.Background(), neg); len(got) != 0 {
		t.Fatalf("negative: expected 0 findings, got %d (%+v)", len(got), got)
	}
	// commented-out SELECT * must not fire.
	cmt := dbTarget("q.go", "// SELECT * FROM users\n")
	if got := (SelectStar{}).MatchFile(context.Background(), cmt); len(got) != 0 {
		t.Fatalf("comment: expected 0 findings, got %d", len(got))
	}
}

func TestDeleteUpdateNoWhere(t *testing.T) {
	pos := dbTarget("q.go", "db.Exec(\"DELETE FROM sessions\")\ndb.Exec(\"UPDATE users SET active = false\")\n")
	if got := (DeleteUpdateNoWhere{}).MatchFile(context.Background(), pos); len(got) != 2 {
		t.Fatalf("positive: expected 2 findings, got %d (%+v)", len(got), got)
	}
	// WHERE present, or statement continues to next line → no finding.
	neg := dbTarget("q.go", "db.Exec(\"DELETE FROM sessions WHERE id = ?\")\n"+
		"q := \"UPDATE users SET active = false\" +\n    \" WHERE id = ?\"\n")
	if got := (DeleteUpdateNoWhere{}).MatchFile(context.Background(), neg); len(got) != 0 {
		t.Fatalf("negative: expected 0 findings, got %d (%+v)", len(got), got)
	}
}

func TestSQLFormatInjectionGo(t *testing.T) {
	pos := dbTarget("q.go", "q := fmt.Sprintf(\"SELECT id FROM users WHERE name = '%s'\", name)\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), pos); len(got) != 1 {
		t.Fatalf("go positive: expected 1, got %d (%+v)", len(got), got)
	}
	neg := dbTarget("q.go", "q := fmt.Sprintf(\"hello %s\", name)\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), neg); len(got) != 0 {
		t.Fatalf("go negative: expected 0, got %d (%+v)", len(got), got)
	}
}

func TestSQLFormatInjectionPy(t *testing.T) {
	fstr := dbTarget("q.py", "q = f\"SELECT id FROM users WHERE name = {name}\"\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), fstr); len(got) != 1 {
		t.Fatalf("py f-string: expected 1, got %d (%+v)", len(got), got)
	}
	pct := dbTarget("q.py", "q = \"SELECT id FROM users WHERE name = '%s'\" % name\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), pct); len(got) != 1 {
		t.Fatalf("py percent: expected 1, got %d (%+v)", len(got), got)
	}
	neg := dbTarget("q.py", "q = f\"hello {name}\"\nq2 = \"SELECT id FROM users\"\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), neg); len(got) != 0 {
		t.Fatalf("py negative: expected 0, got %d (%+v)", len(got), got)
	}
}

func TestSQLFormatInjectionJS(t *testing.T) {
	pos := dbTarget("q.ts", "const q = `SELECT id FROM users WHERE name = ${name}`;\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), pos); len(got) != 1 {
		t.Fatalf("ts positive: expected 1, got %d (%+v)", len(got), got)
	}
	neg := dbTarget("q.ts", "const q = `hello ${name}`;\nconst c = `SELECT id FROM users`;\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), neg); len(got) != 0 {
		t.Fatalf("ts negative: expected 0, got %d (%+v)", len(got), got)
	}
}

func TestDropStatement(t *testing.T) {
	pos := dbTarget("m.go", "db.Exec(\"DROP TABLE users\")\ndb.Exec(\"DROP DATABASE app\")\n")
	if got := (DropStatement{}).MatchFile(context.Background(), pos); len(got) != 2 {
		t.Fatalf("positive: expected 2, got %d (%+v)", len(got), got)
	}
	// English word "drop" / method name must not fire.
	neg := dbTarget("m.go", "list.dropFirst()\n// drop the connection\nfunc dropConn() {}\n")
	if got := (DropStatement{}).MatchFile(context.Background(), neg); len(got) != 0 {
		t.Fatalf("negative: expected 0, got %d (%+v)", len(got), got)
	}
}

func TestTruncateStatement(t *testing.T) {
	pos := dbTarget("m.go", "db.Exec(\"TRUNCATE TABLE events\")\n")
	if got := (TruncateStatement{}).MatchFile(context.Background(), pos); len(got) != 1 {
		t.Fatalf("positive: expected 1, got %d (%+v)", len(got), got)
	}
	neg := dbTarget("m.go", "s := truncate(str, 10)\nf.Truncate(0)\n")
	if got := (TruncateStatement{}).MatchFile(context.Background(), neg); len(got) != 0 {
		t.Fatalf("negative: expected 0, got %d (%+v)", len(got), got)
	}
}

func TestLikeLeadingWildcard(t *testing.T) {
	pos := dbTarget("q.go", "q := \"SELECT id FROM t WHERE name LIKE '%foo'\"\n")
	if got := (LikeLeadingWildcard{}).MatchFile(context.Background(), pos); len(got) != 1 {
		t.Fatalf("positive: expected 1, got %d (%+v)", len(got), got)
	}
	// prefix search and parameterized LIKE must not fire.
	neg := dbTarget("q.go", "a := \"... LIKE 'foo%'\"\nb := \"... LIKE ?\"\n")
	if got := (LikeLeadingWildcard{}).MatchFile(context.Background(), neg); len(got) != 0 {
		t.Fatalf("negative: expected 0, got %d (%+v)", len(got), got)
	}
}

func TestRawQueryEscapeHatch(t *testing.T) {
	pos := dbTarget("v.py", "qs = Entry.objects.raw('SELECT * FROM blog_entry')\n"+
		"qs2 = Entry.objects.extra(where=['id > 1'])\n"+
		"r = text('SELECT 1')\n")
	if got := (RawQueryEscapeHatch{}).MatchFile(context.Background(), pos); len(got) != 3 {
		t.Fatalf("py positive: expected 3, got %d (%+v)", len(got), got)
	}
	js := dbTarget("v.js", "knex.raw('SELECT 1');\nsequelize.query('SELECT 1');\n")
	if got := (RawQueryEscapeHatch{}).MatchFile(context.Background(), js); len(got) != 2 {
		t.Fatalf("js positive: expected 2, got %d (%+v)", len(got), got)
	}
	// self.text / unrelated identifiers must not fire.
	neg := dbTarget("v.py", "self.text(1)\nlabel = mytext()\nx = context(1)\n")
	if got := (RawQueryEscapeHatch{}).MatchFile(context.Background(), neg); len(got) != 0 {
		t.Fatalf("negative: expected 0, got %d (%+v)", len(got), got)
	}
}
