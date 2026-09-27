package mcpsrv

import (
	"context"
	"fmt"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/analyze"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

// --- diff_findings: new/resolved/changed vs a saved baseline ---

func diffFindingsTool() mcplib.Tool {
	return mcplib.NewTool("diff_findings",
		mcplib.WithDescription("Compare the current finding set against a saved baseline (written by 'chiton analyze --diff') and return what is new, resolved, or changed. Use this to answer 'what did this change introduce?' without re-reading the whole report."),
		mcplib.WithString("baseline",
			mcplib.Description("Baseline commit SHA to compare against, or 'latest' (default) for the most recent baseline."),
		),
		mcplib.WithString("target",
			mcplib.Description("Optional target commit SHA whose saved baseline is the 'current' side. Omit to diff the baseline against the live findings in the store."),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

// resolveBaselineRef resolves a git ref (HEAD, branch, short SHA) to the full SHA LoadBaseline keys
// on; "" and "latest" pass through, and the literal ref is kept when git can't resolve it.
func resolveBaselineRef(ref string) string {
	if ref == "" || ref == "latest" {
		return ref
	}
	if sha := analyze.GitCommit(".", ref); sha != "" {
		return sha
	}
	return ref
}

func diffFindingsHandler(gs store.GraphStore, baselineDir string) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		baselineRef := resolveBaselineRef(req.GetString("baseline", "latest"))
		targetRef := resolveBaselineRef(req.GetString("target", ""))

		base, err := analyze.LoadBaseline(baselineDir, baselineRef)
		if err != nil {
			return mcplib.NewToolResultErrorFromErr("load baseline failed", err), nil
		}
		if base == nil {
			return mcplib.NewToolResultText("no baseline found — run 'chiton analyze --diff' to record one, then try again."), nil
		}

		var current []analyze.FindingView
		var targetLabel string
		if targetRef != "" {
			tb, err := analyze.LoadBaseline(baselineDir, targetRef)
			if err != nil {
				return mcplib.NewToolResultErrorFromErr("load target baseline failed", err), nil
			}
			if tb == nil {
				return mcplib.NewToolResultText(fmt.Sprintf("no baseline found for target %q.", targetRef)), nil
			}
			current = tb.Findings
			targetLabel = tb.Commit
		} else {
			current, err = analyze.CurrentFindings(ctx, gs)
			if err != nil {
				return mcplib.NewToolResultErrorFromErr("load current findings failed", err), nil
			}
			targetLabel = "working"
		}

		res := analyze.Diff(base.Findings, current)
		res.Baseline = base.Commit
		res.Target = targetLabel
		return jsonResult(res)
	}
}

// --- pending_analyses: emit LLM-rule task specs ---

func pendingAnalysesTool() mcplib.Tool {
	return mcplib.NewTool("pending_analyses",
		mcplib.WithDescription("Return LLM-rule task specs for files not yet analysed by a given rule. Run each task (inspect the code in 'input' for the issue the rule describes), then call submit_analysis with the findings. This is chiton's subagent path for LLM rules: chiton calls no LLM itself."),
		mcplib.WithString("path",
			mcplib.Description("Directory to analyse (default '.')"),
		),
		mcplib.WithString("rule",
			mcplib.Description("Limit to one LLM rule by exact ID or substring (e.g. 'llm/injection-risk')"),
		),
		mcplib.WithInteger("limit",
			mcplib.DefaultNumber(20),
			mcplib.Description("Max task specs to return (default 20)"),
			mcplib.Min(1),
			mcplib.Max(200),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func pendingAnalysesHandler(svc *analyze.LLMService) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if svc == nil {
			return mcplib.NewToolResultText("LLM analysis unavailable — the server was started without a graph store."), nil
		}
		specs, err := svc.Pending(ctx, req.GetString("path", "."), req.GetString("rule", ""), req.GetInt("limit", 20))
		return pendingResult(specs, err, "pending_analyses",
			"no pending analyses — every file is already analysed for the selected LLM rule(s).")
	}
}

// --- submit_analysis: ingest LLM-rule findings ---

func submitAnalysisTool() mcplib.Tool {
	return mcplib.NewTool("submit_analysis",
		mcplib.WithDescription("Ingest the findings of one LLM-rule task (from pending_analyses). Creates finding nodes tagged INFERRED (or AMBIGUOUS below the confidence threshold) with VIOLATES/AFFECTS edges. Idempotent: re-submitting the same task is a no-op."),
		mcplib.WithString("task_id",
			mcplib.Required(),
			mcplib.Description("The task spec's id"),
		),
		mcplib.WithString("rule_id",
			mcplib.Required(),
			mcplib.Description("The rule id from the task spec (e.g. 'llm/injection-risk')"),
		),
		mcplib.WithString("target_uri",
			mcplib.Description("The file's source_uri from the task spec (used to link findings to the file node)"),
		),
		mcplib.WithArray("findings",
			mcplib.Required(),
			mcplib.Description("Findings: objects {line, end_line, message, confidence, fix}. Empty array means the file is clean for this rule."),
			mcplib.Items(map[string]any{
				"type": "object",
				"properties": map[string]any{
					"line":       map[string]any{"type": "integer"},
					"end_line":   map[string]any{"type": "integer"},
					"message":    map[string]any{"type": "string"},
					"confidence": map[string]any{"type": "number"},
					"fix":        map[string]any{"type": "string"},
				},
				"required": []string{"message"},
			}),
		),
	)
}

func submitAnalysisHandler(svc *analyze.LLMService) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if svc == nil {
			return mcplib.NewToolResultText("LLM analysis unavailable — the server was started without a graph store."), nil
		}
		return bindAndSubmit(ctx, req, "submit_analysis", svc.Submit)
	}
}
