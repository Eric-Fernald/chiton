package database

import (
	"context"
	"regexp"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// RawQueryEscapeHatch flags ORM raw-SQL escape hatches (.raw, .extra, sqlalchemy text, sequelize.query),
// each anchored to a real ORM call site so generic raw, extra, text, and query methods don't match.
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
	// Django .extra( must carry a queryset kwarg so an unrelated object's .extra(...) does not match.
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
	py := analyze.FileExt(t.URI) == "py"
	masked := maskedDB(t, py)
	sqlalchemyImported := sqlalchemyImportRe.Match(masked)

	var out []analyze.Finding
	for line, raw := range analyze.LinesOf(maskedDBText(t, py)) {
		text := stripDBComment(raw, py)

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
