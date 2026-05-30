// Package llm holds chiton's LLM rules: rules evaluated by a host-assistant
// LLM via subagent dispatch rather than by Go code. Each rule is metadata plus
// a prompt; analyze emits one task per (rule, file) and ingests the structured
// findings the host returns, tagged INFERRED (or AMBIGUOUS below the confidence
// threshold). These target the cases deterministic AST rules cannot cover —
// design-rationale mismatch, abstraction leaks, undocumented side effects,
// contract-vs-comment violations, and similar semantic judgments.
package llm

import "github.com/Eric-Fernald/chiton/internal/analyze"

// rule is a data-driven LLM rule: metadata plus a prompt, no Go match logic.
type rule struct {
	meta   analyze.Meta
	prompt string
}

func (r rule) Meta() analyze.Meta { return r.meta }
func (r rule) Prompt() string     { return r.prompt }

// def builds an LLM rule. langs=nil means the rule applies to every supported
// language (most semantic rules are language-agnostic).
func def(id string, cat analyze.Category, sev analyze.Severity, summary, prompt string) rule {
	return rule{
		meta: analyze.Meta{
			ID:         id,
			Category:   cat,
			Severity:   sev,
			Scope:      analyze.FileScope,
			Summary:    summary,
			Provenance: analyze.ProvenanceLLM,
		},
		prompt: prompt,
	}
}

// rules is the LLM starter pack: 24 high-signal semantic rules across all eight
// categories. Prompts are deliberately concrete to keep false positives low.
var rules = []rule{
	// --- Architecture ---
	def("llm/abstraction-leak", analyze.CategoryArchitecture, analyze.SeverityHigh,
		"An abstraction leaks implementation details to its callers.",
		"Find places where an abstraction leaks implementation details to its callers: an interface or public function that returns a concrete type tied to one backend, exposes an internal storage format, or forces callers to know how it is implemented."),
	def("llm/design-rationale-mismatch", analyze.CategoryArchitecture, analyze.SeverityMedium,
		"Implementation contradicts the design rationale stated nearby.",
		"Find code whose behavior contradicts the design rationale stated in a nearby comment or docstring — the comment says something is done a certain way for a stated reason, but the code does something different or for a different effect."),
	def("llm/inconsistent-error-handling", analyze.CategoryArchitecture, analyze.SeverityMedium,
		"The same class of error is handled inconsistently across a file.",
		"Find functions in this file that handle the same class of error in inconsistent ways — some swallow it, some wrap it, some log-and-continue — so that callers cannot reason about how failures propagate."),
	def("llm/missing-abstraction", analyze.CategoryArchitecture, analyze.SeverityLow,
		"A repeated structural pattern should be extracted into a shared helper.",
		"Find repeated structural patterns: three or more near-identical blocks of logic that should be extracted into a shared helper or abstraction. Report the location of each cluster of duplication."),

	// --- Bugs ---
	def("llm/off-by-one-risk", analyze.CategoryBugs, analyze.SeverityHigh,
		"Boundary or index logic looks off by one.",
		"Find boundary or index logic that looks off by one — loop bounds, slice ranges, length comparisons, or termination conditions likely to read or write one element past, or one short of, the intended range."),
	def("llm/race-condition-risk", analyze.CategoryBugs, analyze.SeverityHigh,
		"Shared mutable state is accessed without synchronization.",
		"Find shared mutable state read or written without synchronization in a context that can run concurrently: goroutines, async handlers, shared maps/slices/counters, or singletons mutated from multiple call sites."),
	def("llm/incorrect-error-wrapping", analyze.CategoryBugs, analyze.SeverityMedium,
		"An error is propagated in a way that loses its cause or context.",
		"Find errors returned or logged in a way that loses the original cause or context: a swallowed error, %v instead of %w when wrapping in Go, a bare re-raise that drops the traceback, or a generic message replacing a specific one."),
	def("llm/nil-or-none-deref", analyze.CategoryBugs, analyze.SeverityHigh,
		"A path can dereference a nil/None/undefined value.",
		"Find code paths that can dereference a nil, None, null, or undefined value because a check or early return for an error result or empty value is missing before the value is used."),

	// --- Quality ---
	def("llm/contract-violation-vs-comment", analyze.CategoryQuality, analyze.SeverityMedium,
		"Implementation violates the contract its doc comment promises.",
		"Find functions whose implementation violates the contract stated in their doc comment: documented as idempotent, pure, thread-safe, non-blocking, or returning a specific value, but the body does not honor that claim."),
	def("llm/undocumented-side-effect", analyze.CategoryQuality, analyze.SeverityMedium,
		"A function has a significant side effect its name and docs don't imply.",
		"Find functions that perform a significant side effect — disk or network I/O, global mutation, process or filesystem changes — that is not implied by the function name and not documented in a comment."),
	def("llm/misleading-name", analyze.CategoryQuality, analyze.SeverityLow,
		"An identifier's name implies behavior the code doesn't have.",
		"Find identifiers whose name implies behavior or semantics the code does not actually have: an isX or hasX that mutates state, a getX that performs expensive I/O, a validateX that also writes, a constant whose value contradicts its name."),
	def("llm/dead-logic", analyze.CategoryQuality, analyze.SeverityLow,
		"A branch or condition can never be taken.",
		"Find branches or conditions that can never be true or false given the surrounding code: redundant guards, unreachable branches after a return/throw, or conditions that are constant due to earlier checks."),

	// --- Reliability ---
	def("llm/missing-timeout", analyze.CategoryReliability, analyze.SeverityMedium,
		"An external call has no timeout, deadline, or cancellation.",
		"Find external calls — network requests, RPCs, database queries, subprocess execution — issued without a timeout, deadline, or cancellation path, so they could hang indefinitely."),
	def("llm/unhandled-failure-mode", analyze.CategoryReliability, analyze.SeverityMedium,
		"An obvious failure mode is not handled.",
		"Find an obvious or commented-about failure mode the surrounding code does not handle: a partial write, a short read, a reconnect, a retry, or an empty/error response that the code assumes will succeed."),
	def("llm/resource-leak-risk", analyze.CategoryReliability, analyze.SeverityHigh,
		"A resource is acquired without a guaranteed release.",
		"Find resources — files, connections, locks, handles, sessions — acquired without a guaranteed release on every path, including early returns and error paths (missing defer/Close, finally, with-block, or unlock)."),

	// --- Security ---
	def("llm/injection-risk", analyze.CategorySecurity, analyze.SeverityCritical,
		"Untrusted input flows into a dangerous sink unsanitized.",
		"Find untrusted input — request parameters, CLI args, environment, file contents — flowing into a dangerous sink (SQL query, shell command, eval, template, or filesystem path) without validation, parameterization, or escaping."),
	def("llm/missing-authz-check", analyze.CategorySecurity, analyze.SeverityHigh,
		"A sensitive operation appears to skip an authorization check.",
		"Find an operation that mutates or returns sensitive data where an authorization or permission check appears to be missing, especially when sibling operations in the same file perform such a check."),
	def("llm/sensitive-data-in-logs", analyze.CategorySecurity, analyze.SeverityMedium,
		"Secrets or PII are written to logs or error messages.",
		"Find secrets, credentials, API tokens, or personally identifiable information written to logs, error messages, or stdout where they could be captured or leaked."),

	// --- Performance ---
	def("llm/n-plus-one", analyze.CategoryPerformance, analyze.SeverityMedium,
		"A loop issues per-iteration I/O that could be batched.",
		"Find a loop that issues a per-iteration database query, network request, or I/O call that could be batched or hoisted out of the loop — the classic N+1 pattern."),
	def("llm/unbounded-growth", analyze.CategoryPerformance, analyze.SeverityMedium,
		"A collection or buffer grows without any bound.",
		"Find a collection, cache, buffer, or queue that grows without any eviction policy or size bound, risking unbounded memory growth over the life of the process."),

	// --- Database ---
	def("llm/missing-transaction-boundary", analyze.CategoryDatabase, analyze.SeverityHigh,
		"Related writes that must be atomic run without a transaction.",
		"Find a sequence of related writes — multiple inserts or updates that must all succeed or all fail together — executed without a surrounding transaction, leaving the data in a partially-updated state on failure."),
	def("llm/read-modify-write-race", analyze.CategoryDatabase, analyze.SeverityMedium,
		"A read-modify-write on shared state can lose updates.",
		"Find a read-modify-write against shared or persistent state without isolation — a SELECT then UPDATE, or get-then-put — that can lose updates when two callers run concurrently."),

	// --- Style ---
	def("llm/comment-rot", analyze.CategoryStyle, analyze.SeverityLow,
		"A comment describes code that no longer matches the implementation.",
		"Find comments that describe code that no longer matches the adjacent implementation: stale TODOs about work that is already done, comments referencing renamed or removed symbols, or descriptions that contradict the current code."),
	def("llm/magic-policy-value", analyze.CategoryStyle, analyze.SeverityLow,
		"A hard-coded policy value should be a named constant or config.",
		"Find hard-coded policy or business values embedded inline — timeouts, limits, thresholds, retry counts, page sizes — that should be named constants or configuration rather than magic numbers in the logic."),
}

func init() {
	for _, r := range rules {
		analyze.Register(r)
	}
}
