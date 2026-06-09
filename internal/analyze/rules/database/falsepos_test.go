package database

import (
	"context"
	"testing"
)

// --- Concern #1: stripDBComment / maskDBBlocks block-comment + triple-quote gap.

// A SQL keyword inside a single-line /* ... */ block comment must not fire.
func TestDropStatement_SingleLineBlockComment(t *testing.T) {
	fp := dbTarget("m.go", "x := 1 /* historically we used DROP TABLE users here */\n")
	if got := (DropStatement{}).MatchFile(context.Background(), fp); len(got) != 0 {
		t.Fatalf("single-line block comment: expected 0, got %d (%+v)", len(got), got)
	}
	// Genuine DROP still fires.
	tp := dbTarget("m.go", "db.Exec(\"DROP TABLE users\")\n")
	if got := (DropStatement{}).MatchFile(context.Background(), tp); len(got) != 1 {
		t.Fatalf("true positive: expected 1, got %d (%+v)", len(got), got)
	}
}

// A SQL keyword inside a MULTI-line /* ... */ block comment must not fire — the
// open/close must be tracked across lines (per-file pass).
func TestDropStatement_MultiLineBlockComment(t *testing.T) {
	fp := dbTarget("m.go", "/*\n  Migration note:\n  DROP TABLE legacy_users\n  TRUNCATE TABLE audit\n*/\nfunc f() {}\n")
	if got := (DropStatement{}).MatchFile(context.Background(), fp); len(got) != 0 {
		t.Fatalf("multi-line block comment (DROP): expected 0, got %d (%+v)", len(got), got)
	}
	if got := (TruncateStatement{}).MatchFile(context.Background(), fp); len(got) != 0 {
		t.Fatalf("multi-line block comment (TRUNCATE): expected 0, got %d (%+v)", len(got), got)
	}
}

// SQL keywords inside a Python triple-quoted docstring must not fire.
func TestDatabaseRules_PyTripleQuotedDocstring(t *testing.T) {
	doc := "def handler():\n" +
		"    \"\"\"\n" +
		"    This endpoint used to DROP TABLE sessions and run\n" +
		"    DELETE FROM cache and SELECT * FROM users.\n" +
		"    Also LIKE '%legacy' lookups were common.\n" +
		"    \"\"\"\n" +
		"    return 1\n"
	fp := dbTarget("h.py", doc)
	if got := (DropStatement{}).MatchFile(context.Background(), fp); len(got) != 0 {
		t.Fatalf("docstring DROP: expected 0, got %d (%+v)", len(got), got)
	}
	if got := (DeleteUpdateNoWhere{}).MatchFile(context.Background(), fp); len(got) != 0 {
		t.Fatalf("docstring DELETE: expected 0, got %d (%+v)", len(got), got)
	}
	if got := (SelectStar{}).MatchFile(context.Background(), fp); len(got) != 0 {
		t.Fatalf("docstring SELECT *: expected 0, got %d (%+v)", len(got), got)
	}
	if got := (LikeLeadingWildcard{}).MatchFile(context.Background(), fp); len(got) != 0 {
		t.Fatalf("docstring LIKE: expected 0, got %d (%+v)", len(got), got)
	}
	// Real DELETE/SELECT in py code still fires.
	tp := dbTarget("h.py", "cur.execute(\"DELETE FROM sessions\")\n")
	if got := (DeleteUpdateNoWhere{}).MatchFile(context.Background(), tp); len(got) != 1 {
		t.Fatalf("py true positive DELETE: expected 1, got %d (%+v)", len(got), got)
	}
}

// --- Concern #2: sql-format-injection must require the SQL skeleton.

// Interpolated strings that merely contain the bare words "update"/"select"
// (with no SET/FROM) must NOT fire.
func TestSQLFormatInjection_BareVerbNoSkeleton(t *testing.T) {
	goFP := dbTarget("a.go", "msg := fmt.Sprintf(\"update of path %s applied\", p)\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), goFP); len(got) != 0 {
		t.Fatalf("go bare 'update': expected 0, got %d (%+v)", len(got), got)
	}
	jsFP := dbTarget("a.ts", "const m = `select of a count: ${n}`;\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), jsFP); len(got) != 0 {
		t.Fatalf("ts bare 'select': expected 0, got %d (%+v)", len(got), got)
	}
	pyFP := dbTarget("a.py", "m = f\"select of a count: {n}\"\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), pyFP); len(got) != 0 {
		t.Fatalf("py bare 'select': expected 0, got %d (%+v)", len(got), got)
	}
}

// The full SELECT...FROM and UPDATE...SET skeletons must STILL fire.
func TestSQLFormatInjection_SkeletonStillFires(t *testing.T) {
	upd := dbTarget("a.go", "q := fmt.Sprintf(\"UPDATE users SET name='%s' WHERE id=1\", n)\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), upd); len(got) != 1 {
		t.Fatalf("go UPDATE...SET: expected 1, got %d (%+v)", len(got), got)
	}
	sel := dbTarget("a.ts", "const q = `SELECT id FROM users WHERE name = ${n}`;\n")
	if got := (SQLFormatInjection{}).MatchFile(context.Background(), sel); len(got) != 1 {
		t.Fatalf("ts SELECT...FROM: expected 1, got %d (%+v)", len(got), got)
	}
}

// --- Concern #3: raw-query-escape-hatch must anchor to ORM call sites.

// Generic .raw/.extra/text/.query usages that are NOT ORM calls must not fire.
func TestRawQueryEscapeHatch_GenericMethodsNoFire(t *testing.T) {
	// express raw body / request .raw() with no SQL; react/i18n text(); object
	// .extra() without django kwargs; unrelated .query().
	fp := dbTarget("app.js", "res.raw('hello world');\n"+
		"const label = text('welcome.title');\n"+
		"widget.extra('foo');\n"+
		"myEmitter.query('status');\n")
	if got := (RawQueryEscapeHatch{}).MatchFile(context.Background(), fp); len(got) != 0 {
		t.Fatalf("generic methods: expected 0, got %d (%+v)", len(got), got)
	}
}

// Real ORM escape hatches must STILL fire.
func TestRawQueryEscapeHatch_OrmCallsStillFire(t *testing.T) {
	// knex.raw with SQL, sequelize.query, django objects.raw, django .extra kwarg.
	js := dbTarget("db.js", "knex.raw('SELECT 1');\nsequelize.query('SELECT * FROM t');\n")
	if got := (RawQueryEscapeHatch{}).MatchFile(context.Background(), js); len(got) != 2 {
		t.Fatalf("js orm calls: expected 2, got %d (%+v)", len(got), got)
	}
	py := dbTarget("v.py", "qs = Entry.objects.raw('SELECT * FROM blog_entry')\n"+
		"qs2 = Entry.objects.extra(where=['id > 1'])\n")
	if got := (RawQueryEscapeHatch{}).MatchFile(context.Background(), py); len(got) != 2 {
		t.Fatalf("py orm calls: expected 2, got %d (%+v)", len(got), got)
	}
}

// sqlalchemy text() fires when the file imports sqlalchemy even if the arg is not
// obviously SQL on its own; without the import a bare i18n-style text() is quiet.
func TestRawQueryEscapeHatch_SqlAlchemyText(t *testing.T) {
	withImport := dbTarget("q.py", "from sqlalchemy import text\nstmt = text('some_sql_fragment')\n")
	if got := (RawQueryEscapeHatch{}).MatchFile(context.Background(), withImport); len(got) != 1 {
		t.Fatalf("sqlalchemy import + text(): expected 1, got %d (%+v)", len(got), got)
	}
	noImport := dbTarget("q.py", "stmt = text('some_sql_fragment')\n")
	if got := (RawQueryEscapeHatch{}).MatchFile(context.Background(), noImport); len(got) != 0 {
		t.Fatalf("no import + non-SQL text(): expected 0, got %d (%+v)", len(got), got)
	}
}
