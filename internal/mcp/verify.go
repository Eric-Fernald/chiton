package mcpsrv

import (
	"context"

	mcplib "github.com/mark3labs/mcp-go/mcp"
	mcpsdk "github.com/mark3labs/mcp-go/server"

	"github.com/Eric-Fernald/chiton/internal/verify"
)

// --- pending_verifications / submit_verification: re-check INFERRED claims against their sources ---

// The two descriptions are kept short on purpose: every tool definition is loaded into every session.

func pendingVerificationsTool() mcplib.Tool {
	return mcplib.NewTool("pending_verifications",
		mcplib.WithDescription("Return tasks that re-check inferred concepts and decisions against their sources, for claims whose evidence changed since the last check. Judge each 'claim' against its 'evidence', then call submit_verification. chiton calls no LLM itself."),
		mcplib.WithString("kind",
			mcplib.Description("Filter to one task kind: concept_support | decision_drift (empty = both)"),
		),
		mcplib.WithInteger("limit",
			mcplib.DefaultNumber(10),
			mcplib.Description("Max task specs to return (default 10)"),
			mcplib.Min(1),
			mcplib.Max(50),
		),
		mcplib.WithReadOnlyHintAnnotation(true),
	)
}

func pendingVerificationsHandler(svc *verify.Service) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if svc == nil {
			return mcplib.NewToolResultText("verification unavailable — the server was started without a graph store."), nil
		}
		specs, err := svc.Pending(ctx, req.GetString("kind", ""), clampInt(req.GetInt("limit", 10), 1, 50))
		return pendingResult(specs, err, "pending_verifications",
			"no pending verifications — every checkable claim matches its last verdict.")
	}
}

func submitVerificationTool() mcplib.Tool {
	return mcplib.NewTool("submit_verification",
		mcplib.WithDescription("Apply the verdict for one task from pending_verifications. The quote must be copied from the task's evidence, and a verdict below the confidence threshold is recorded without changing the graph. Requires a writable store."),
		mcplib.WithString("task_id",
			mcplib.Required(),
			mcplib.Description("The task spec's id"),
		),
		mcplib.WithString("fingerprint",
			mcplib.Required(),
			mcplib.Description("The task spec's fingerprint, echoed back"),
		),
		mcplib.WithString("verdict",
			mcplib.Required(),
			mcplib.Description("concept_support: supported | revised | unsupported. decision_drift: supported | drifted | mislinked."),
		),
		mcplib.WithString("quote",
			mcplib.Description("Passage copied from the evidence. Required for supported, revised and drifted."),
		),
		mcplib.WithString("description",
			mcplib.Description("revised only: the corrected one-sentence description"),
		),
		mcplib.WithString("note",
			mcplib.Description("drifted only: what the code does now"),
		),
		mcplib.WithNumber("confidence",
			mcplib.Description("Confidence in the verdict, 0 to 1"),
		),
	)
}

func submitVerificationHandler(svc *verify.Service) mcpsdk.ToolHandlerFunc {
	return func(ctx context.Context, req mcplib.CallToolRequest) (*mcplib.CallToolResult, error) {
		if svc == nil {
			return mcplib.NewToolResultText("verification unavailable — the server was started without a graph store."), nil
		}
		return bindAndSubmit(ctx, req, "submit_verification", svc.Submit)
	}
}
