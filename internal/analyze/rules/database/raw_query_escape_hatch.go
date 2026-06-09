package database

import (
	"bufio"
	"bytes"
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// RawQueryEscapeHatch flags ORM raw-SQL escape hatches — the methods that drop
// out of the query builder and run a literal SQL string, bypassing the ORM's
// parameterization and validation. They are not bugs by themselves, but they
// warrant review since they are where injection and schema-coupling sneak in.
//
// To stay low-false-positive the rule anchors to real ORM call sites rather than
// matching any method named raw/extra/text/query:
//   - django/knex .raw(...) — only when the call argument looks like SQL
//     (contains a SQL verb) or the receiver is a query builder (objects/knex).
//     This skips express `req.raw`, a generic `.raw` accessor, etc.
//   - django .extra(...) — the queryset escape hatch; required to carry a
//     Django-style kwarg (select=/where=/tables=/params=/order_by=) so an
//     unrelated object's .extra(...) does not match.
//   - sqlalchemy text("...") — only when the file imports sqlalchemy OR the
//     string argument begins with a SQL verb. This skips a React/i18n
//     `text('some.key')` helper.
//   - sequelize.query("...") — the receiver name anchors it.
type RawQueryEscapeHatch struct{}

func (RawQueryEscapeHatch) Meta() analyze.Meta {
	return analyze.Meta{
		ID:         "database/raw-query-escape-hatch",
		Category:   analyze.CategoryDatabase,
		Severity:   analyze.SeverityMedium,
		Scope:      analyze.FileScope,
		Languages:  []string{"py", "js", "ts"},
		Summary:    "ORM raw-SQL escape hatch (.raw()/.extra()/sqlalchemy text()/sequelize.query()).",
		Fix:        "Prefer the ORM's query builder; if raw SQL is required, bind all parameters.",
		Provenance: analyze.ProvenanceDeterministic,
	}
}

// sqlLooking matches a SQL verb skeleton anywhere on the (already masked) line,
// used to confirm a raw() call argument is actually SQL.
var sqlLooking = regexp.MustCompile(`(?i)\b(?:SELECT\b[^"'` + "`" + `]*?\bFROM|INSERT\s+INTO|UPDATE\b[^"'` + "`" + `]*?\bSET|DELETE\s+FROM|CREATE\s+TABLE|DROP\s+TABLE|TRUNCATE\s+TABLE)\b`)

// sqlalchemyImportRe detects a sqlalchemy import so a text("...") call is treated
// as the SQLAlchemy escape hatch even when the argument is not obviously SQL.
var sqlalchemyImportRe = regexp.MustCompile(`(?i)\b(?:import\s+sqlalchemy|from\s+sqlalchemy\b)`)

var (
	// .extra( — django queryset escape hatch. Require a Django-style kwarg
	// (select=/where=/tables=/params=/order_by=) so an unrelated object's
	// .extra(...) method does not match.
	extraCallRe = regexp.MustCompile(`\.extra\s*\(\s*(?:select|where|tables|params|order_by)\s*=`)
	// .raw( on a query builder (objects / knex receiver), regardless of arg.
	builderRawRe = regexp.MustCompile(`(?i)\b(?:objects|knex)\.raw\s*\(`)
	// any .raw( call — gated by a SQL-looking argument.
	rawCallRe = regexp.MustCompile(`\.raw\s*\(`)
	// text( as a call (not a method on something, e.g. self.text()).
	textCallRe = regexp.MustCompile(`(^|[^.\w])text\s*\(`)
	// text("SELECT ...") — string argument that opens with a SQL verb keyword.
	textSQLArgRe = regexp.MustCompile(`(?i)(^|[^.\w])text\s*\(\s*['"]\s*(?:SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|DROP|ALTER|TRUNCATE)\b`)
	// sequelize.query( — receiver name anchors it.
	sequelizeQueryRe = regexp.MustCompile(`(?i)\bsequelize\.query\s*\(`)
)

func (RawQueryEscapeHatch) MatchFile(_ context.Context, t analyze.FileTarget) []analyze.Finding {
	masked := maskDBBlocks(t.Content, t.URI)
	sqlalchemyImported := sqlalchemyImportRe.Match(masked)

	var out []analyze.Finding
	sc := bufio.NewScanner(bytes.NewReader(masked))
	sc.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	line := 0
	for sc.Scan() {
		line++
		text := stripDBComment(sc.Text(), t.URI)

		var name string
		switch {
		case sequelizeQueryRe.MatchString(text):
			name = "sequelize.query()"
		case extraCallRe.MatchString(text):
			name = "django .extra()"
		case builderRawRe.MatchString(text) || (rawCallRe.MatchString(text) && sqlLooking.MatchString(text)):
			name = "django/knex .raw()"
		case textCallRe.MatchString(text) && (sqlalchemyImported || textSQLArgRe.MatchString(text)):
			name = "sqlalchemy text()"
		default:
			continue
		}

		out = append(out, analyze.Finding{
			RuleID:       "database/raw-query-escape-hatch",
			Severity:     analyze.SeverityMedium,
			Category:     analyze.CategoryDatabase,
			Message:      "ORM raw-SQL escape hatch (" + name + ")",
			Fix:          "Prefer the query builder; if raw SQL is required, bind all parameters.",
			TargetURI:    t.URI,
			TargetNodeID: t.FileNode.ID,
			Span:         &model.SourceSpan{StartLine: line, EndLine: line},
		})
	}
	return out
}

func init() { analyze.Register(RawQueryEscapeHatch{}) }
