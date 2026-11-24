package verify

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/Eric-Fernald/chiton/internal/extract/meeting"
	"github.com/Eric-Fernald/chiton/internal/extract/subagent"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

func newStore(t *testing.T) *badgerstore.Store {
	t.Helper()
	gs, err := badgerstore.Open(filepath.Join(t.TempDir(), "graph.badger"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { gs.Close() })
	return gs
}

const (
	sectionID = "d1"
	conceptID = "c1"
	authText  = "We authenticate every request with a JWT signed by the gateway."
)

func setSection(t *testing.T, gs store.GraphStore, text string) {
	t.Helper()
	if err := gs.UpsertNodes(context.Background(), []model.Node{{
		ID: sectionID, Kind: model.KindDocSection, Label: "Auth", SourceURI: "file://doc.md",
		Attrs: map[string]any{"text": text}, Provenance: model.ProvExtracted, Confidence: 1,
	}}); err != nil {
		t.Fatal(err)
	}
}

// seedConcept stores one doc section describing one inferred concept.
func seedConcept(t *testing.T, gs store.GraphStore) {
	t.Helper()
	ctx := context.Background()
	setSection(t, gs, authText)
	if err := gs.UpsertNodes(ctx, []model.Node{{
		ID: conceptID, Kind: model.KindConcept, Label: "JWT", SourceURI: "concept://jwt",
		Attrs:      map[string]any{"description": "a session cookie", "extracted_from": sectionID},
		Provenance: model.ProvInferred, Confidence: 0.9,
	}}); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, []model.Edge{{
		ID: model.NewEdgeID(sectionID, conceptID, model.EdgeDescribes), Src: sectionID, Dst: conceptID,
		Kind: model.EdgeDescribes, Provenance: model.ProvInferred, Confidence: 0.9,
	}}); err != nil {
		t.Fatal(err)
	}
}

func pending(t *testing.T, svc *Service, kind string) []TaskSpec {
	t.Helper()
	specs, err := svc.Pending(context.Background(), kind, 0)
	if err != nil {
		t.Fatal(err)
	}
	return specs
}

// onlyTask returns the single pending task of kind, failing the test when there isn't exactly one.
func onlyTask(t *testing.T, svc *Service, kind string) TaskSpec {
	t.Helper()
	specs := pending(t, svc, kind)
	if len(specs) != 1 {
		t.Fatalf("want 1 pending %s task, got %d: %+v", kind, len(specs), specs)
	}
	return specs[0]
}

func node(t *testing.T, gs store.GraphStore, id string) *model.Node {
	t.Helper()
	n, err := gs.GetNode(context.Background(), id)
	if err != nil {
		t.Fatal(err)
	}
	return n
}

func edge(t *testing.T, gs store.GraphStore, src, dst string, kind model.EdgeKind) *model.Edge {
	t.Helper()
	edges, err := gs.AllEdges(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	for i := range edges {
		if edges[i].Src == src && edges[i].Dst == dst && edges[i].Kind == kind {
			return &edges[i]
		}
	}
	return nil
}

func TestPendingEmitsConceptWithItsSectionText(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	spec := onlyTask(t, New(gs, t.TempDir(), 0.5), KindConceptSupport)

	if spec.ID != conceptID || spec.Kind != KindConceptSupport || spec.Fingerprint == "" {
		t.Errorf("spec id/kind/fingerprint = %q/%q/%q", spec.ID, spec.Kind, spec.Fingerprint)
	}
	if !strings.Contains(spec.Claim, "JWT") || !strings.Contains(spec.Claim, "a session cookie") {
		t.Errorf("claim %q should carry the concept's label and description", spec.Claim)
	}
	if len(spec.Evidence) != 1 || spec.Evidence[0].NodeID != sectionID || spec.Evidence[0].Text != authText {
		t.Errorf("evidence = %+v, want the section's text", spec.Evidence)
	}
	// Ingested text must stay out of the prompt fields.
	if strings.Contains(spec.Instructions, "gateway") || strings.Contains(spec.Schema, "gateway") {
		t.Error("evidence text leaked into instructions or schema")
	}
}

// A concept with no text to check against cannot be verified by quote, so it is not a task.
func TestPendingSkipsConceptsWithoutTextEvidence(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	_ = gs.UpsertNodes(ctx, []model.Node{
		{ID: "lonely", Kind: model.KindConcept, Label: "Lonely", SourceURI: "concept://lonely", Provenance: model.ProvInferred},
		{ID: "pictured", Kind: model.KindConcept, Label: "Pictured", SourceURI: "concept://pictured", Provenance: model.ProvInferred},
		{ID: "img", Kind: model.KindImage, Label: "arch.png", SourceURI: "file://arch.png", Attrs: map[string]any{"path": "arch.png"}},
	})
	_ = gs.UpsertEdges(ctx, []model.Edge{{ID: "e", Src: "img", Dst: "pictured", Kind: model.EdgeDescribes, Provenance: model.ProvInferred}})

	if specs := pending(t, New(gs, t.TempDir(), 0.5), ""); len(specs) != 0 {
		t.Fatalf("want no tasks, got %+v", specs)
	}
}

func TestSupportedConceptIsNotRequeuedUntilItsEvidenceChanges(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport)

	rep, err := svc.Submit(context.Background(), Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Supported,
		Quote: "authenticate every request with a JWT", Confidence: 0.9,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !rep.Applied {
		t.Errorf("report = %+v, want applied", rep)
	}
	n := node(t, gs, conceptID)
	if n.Provenance != model.ProvInferred || model.AttrString(n.Attrs, "evidence_quote") == "" ||
		model.AttrString(n.Attrs, "verified_at") == "" {
		t.Errorf("verified concept = %+v", n)
	}
	if specs := pending(t, svc, ""); len(specs) != 0 {
		t.Fatalf("verified concept was requeued: %+v", specs)
	}

	setSection(t, gs, "We now authenticate with mutual TLS.")
	if again := onlyTask(t, svc, KindConceptSupport); again.Fingerprint == spec.Fingerprint {
		t.Error("fingerprint did not change with the evidence")
	}
}

// The quote is how chiton checks the verifier: it must be in the evidence chiton sent.
func TestSubmitRejectsQuoteThatIsNotInTheEvidence(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport)

	for _, quote := range []string{"", "JWTs are validated by the auth service", "a JWT"} {
		_, err := svc.Submit(context.Background(), Result{
			TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Supported, Quote: quote, Confidence: 0.9,
		})
		if err == nil {
			t.Errorf("Submit accepted quote %q", quote)
		}
	}
	if len(pending(t, svc, "")) != 1 {
		t.Error("a rejected submit must leave the task pending")
	}
}

func TestSubmitAcceptsQuoteWithReflowedWhitespace(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	setSection(t, gs, "We authenticate every request\n    with a JWT signed by the gateway.")
	svc := New(gs, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport)

	if _, err := svc.Submit(context.Background(), Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Supported,
		Quote: "every request with a JWT", Confidence: 0.9,
	}); err != nil {
		t.Fatal(err)
	}
}

func TestSubmitRejectsStaleFingerprint(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport)
	setSection(t, gs, "We now authenticate with mutual TLS.")

	_, err := svc.Submit(context.Background(), Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9,
	})
	if err == nil || !strings.Contains(err.Error(), "changed") {
		t.Fatalf("want a stale-evidence error, got %v", err)
	}
	if node(t, gs, conceptID).Provenance != model.ProvInferred {
		t.Error("a stale verdict was applied")
	}
}

func TestRevisedRewritesTheDescription(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport)

	if _, err := svc.Submit(context.Background(), Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Revised,
		Quote: "a JWT signed by the gateway", Description: "a signed token the gateway issues", Confidence: 0.9,
	}); err != nil {
		t.Fatal(err)
	}
	n := node(t, gs, conceptID)
	if got := model.AttrString(n.Attrs, "description"); got != "a signed token the gateway issues" {
		t.Errorf("description = %q", got)
	}
	if model.AttrString(n.Attrs, "extracted_from") != sectionID {
		t.Error("revising a concept changed its owner")
	}
	if specs := pending(t, svc, ""); len(specs) != 0 {
		t.Fatalf("revised concept was requeued: %+v", specs)
	}
}

func TestRevisedWithoutDescriptionIsRejected(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport)
	if _, err := svc.Submit(context.Background(), Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Revised, Quote: "a JWT signed by the gateway", Confidence: 0.9,
	}); err == nil {
		t.Fatal("revised with no description was accepted")
	}
}

// One wrong verdict must cost a reversible demotion, not the node. Deletion needs a second verdict on new evidence.
func TestUnsupportedDemotesFirstAndDeletesOnTheNextEvidence(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	ctx := context.Background()
	spec := onlyTask(t, svc, KindConceptSupport)
	unsupported := Result{TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9}

	if _, err := svc.Submit(ctx, unsupported); err != nil {
		t.Fatal(err)
	}
	if n := node(t, gs, conceptID); n == nil || n.Provenance != model.ProvAmbiguous {
		t.Fatalf("after one unsupported verdict the concept should be AMBIGUOUS, got %+v", n)
	}
	// A retry of the same verdict is not a second opinion.
	if _, err := svc.Submit(ctx, unsupported); err != nil {
		t.Fatal(err)
	}
	if node(t, gs, conceptID) == nil {
		t.Fatal("re-sending the same verdict deleted the concept")
	}

	setSection(t, gs, "We now authenticate with mutual TLS.")
	spec = onlyTask(t, svc, KindConceptSupport)
	rep, err := svc.Submit(ctx, Result{TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9})
	if err != nil {
		t.Fatal(err)
	}
	if node(t, gs, conceptID) != nil {
		t.Errorf("concept survived a second unsupported verdict on new evidence (report %+v)", rep)
	}
}

func TestSupportedRestoresADemotedConcept(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	ctx := context.Background()
	spec := onlyTask(t, svc, KindConceptSupport)
	if _, err := svc.Submit(ctx, Result{TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9}); err != nil {
		t.Fatal(err)
	}

	setSection(t, gs, "Every request carries a JWT again.")
	spec = onlyTask(t, svc, KindConceptSupport)
	if _, err := svc.Submit(ctx, Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Supported, Quote: "carries a JWT again", Confidence: 0.9,
	}); err != nil {
		t.Fatal(err)
	}
	if n := node(t, gs, conceptID); n.Provenance != model.ProvInferred {
		t.Errorf("provenance = %s, want INFERRED restored", n.Provenance)
	}
}

// A verdict the verifier is unsure of is remembered, so the task is not re-run, but it changes nothing.
func TestLowConfidenceVerdictIsRecordedButNotApplied(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport)

	rep, err := svc.Submit(context.Background(), Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.2,
	})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Applied {
		t.Errorf("report = %+v, want not applied", rep)
	}
	if node(t, gs, conceptID).Provenance != model.ProvInferred {
		t.Error("a low-confidence verdict demoted the concept")
	}
	if specs := pending(t, svc, ""); len(specs) != 0 {
		t.Fatalf("recorded verdict was requeued: %+v", specs)
	}
}

func TestSubmitRejectsUnknownTaskAndVerdict(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport)
	ctx := context.Background()

	for name, r := range map[string]Result{
		"unknown task":          {TaskID: "nope", Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9},
		"not a claim":           {TaskID: sectionID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9},
		"verdict of other kind": {TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Drifted, Quote: "a JWT signed by the gateway", Confidence: 0.9},
		"made-up verdict":       {TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: "fine", Confidence: 0.9},
	} {
		if _, err := svc.Submit(ctx, r); err == nil {
			t.Errorf("%s: Submit succeeded", name)
		}
	}
}

func TestSubmitRejectsReadOnlyStore(t *testing.T) {
	dir := filepath.Join(t.TempDir(), "graph.badger")
	rw, err := badgerstore.Open(dir)
	if err != nil {
		t.Fatal(err)
	}
	seedConcept(t, rw)
	rw.Close()
	ro, err := badgerstore.OpenReadOnly(dir)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { ro.Close() })

	svc := New(ro, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport) // Pending only reads
	_, err = svc.Submit(context.Background(), Result{TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9})
	if err == nil || !strings.Contains(err.Error(), "read-only") {
		t.Fatalf("want a read-only error, got %v", err)
	}
}

// --- decisions ---

const clientSrc = `package client

func fetchAll() {
	const maxRetries = 3
}

func other() {}
`

type decisionFixture struct {
	svc      *Service
	gs       store.GraphStore
	root     string
	file     string
	decision string
	symbol   string
}

func writeFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

// indexFile stores the file node the way a build would, recording the content hash the spans were taken from.
func indexFile(t *testing.T, gs store.GraphStore, path, content string) {
	t.Helper()
	uri := model.FileURI(path)
	if err := gs.UpsertNodes(context.Background(), []model.Node{{
		ID: model.NewNodeID(uri, uri), Kind: model.KindFile, Label: filepath.Base(path), SourceURI: uri,
		Provenance: model.ProvExtracted, Confidence: 1, ContentSHA256: model.ContentSHA([]byte(content)),
	}}); err != nil {
		t.Fatal(err)
	}
}

// seedDecision writes client.go under dir, indexes fetchAll (lines 3-5), and links a meeting decision to it.
// root is the project root the service may read from.
func seedDecision(t *testing.T, root, dir string) decisionFixture {
	t.Helper()
	ctx := context.Background()
	gs := newStore(t)
	file := filepath.Join(dir, "client.go")
	writeFile(t, file, clientSrc)
	indexFile(t, gs, file, clientSrc)
	uri := model.FileURI(file)
	sym := model.Node{
		ID: model.NewNodeID(uri, "fetchAll"), Kind: model.KindSymbol, Label: "fetchAll", SourceURI: uri,
		Span: &model.SourceSpan{StartLine: 3, EndLine: 5}, Provenance: model.ProvExtracted, Confidence: 1,
	}
	mNodes, mEdges := meeting.ToGraph(&meeting.Transcript{
		Title: "Retry sync", Turns: []meeting.Turn{{Speaker: "Alice", Text: "Cap retries at 3 in fetchAll."}},
	}, time.Now())
	if err := gs.UpsertNodes(ctx, append(mNodes, sym)); err != nil {
		t.Fatal(err)
	}
	if err := gs.UpsertEdges(ctx, mEdges); err != nil {
		t.Fatal(err)
	}
	if _, err := meeting.NewLinkService(gs, 0.5).Submit(ctx, meeting.LinkResult{TaskID: mNodes[0].ID,
		Decisions: []meeting.DecisionResult{{
			Summary: "Cap retries at 3", Rationale: "avoid hammering the upstream", Confidence: 0.9, References: []string{"fetchAll"},
		}}}); err != nil {
		t.Fatal(err)
	}
	decisions, err := store.NodesMatching(ctx, gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindDecision}})
	if err != nil || len(decisions) != 1 {
		t.Fatalf("want 1 decision, got %d (%v)", len(decisions), err)
	}
	return decisionFixture{svc: New(gs, root, 0.5), gs: gs, root: root, file: file, decision: decisions[0].ID, symbol: sym.ID}
}

func seedDecisionInRoot(t *testing.T) decisionFixture {
	root := t.TempDir()
	return seedDecision(t, root, root)
}

func TestPendingEmitsDecisionWithItsSymbolSource(t *testing.T) {
	f := seedDecisionInRoot(t)
	spec := onlyTask(t, f.svc, KindDecisionDrift)

	if spec.ID != f.decision+":"+f.symbol {
		t.Errorf("task id = %q, want decision:symbol", spec.ID)
	}
	if !strings.Contains(spec.Claim, "Cap retries at 3") || !strings.Contains(spec.Claim, "avoid hammering the upstream") {
		t.Errorf("claim %q should carry the decision and its rationale", spec.Claim)
	}
	if len(spec.Evidence) != 1 {
		t.Fatalf("evidence = %+v, want the one governed symbol", spec.Evidence)
	}
	got := spec.Evidence[0].Text
	if !strings.Contains(got, "const maxRetries = 3") || strings.Contains(got, "func other") || strings.Contains(got, "package client") {
		t.Errorf("evidence should be fetchAll's own lines only, got:\n%s", got)
	}
}

// Task specs carry file contents, so a symbol whose file lies outside the project root is not read.
func TestPendingSkipsSymbolOutsideTheProjectRoot(t *testing.T) {
	f := seedDecision(t, t.TempDir(), t.TempDir())
	if specs := pending(t, f.svc, ""); len(specs) != 0 {
		t.Fatalf("want no tasks for a file outside the root, got %+v", specs)
	}
}

// A span is only valid for the file content it was taken from. After an edit the graph has not caught up with,
// the lines at that span may belong to something else.
func TestPendingWaitsForReindexAfterTheFileChanges(t *testing.T) {
	f := seedDecisionInRoot(t)
	edited := "// a new first line shifts every span\n" + clientSrc
	writeFile(t, f.file, edited)
	if specs := pending(t, f.svc, ""); len(specs) != 0 {
		t.Fatalf("want no tasks while the graph is behind the file, got %+v", specs)
	}

	indexFile(t, f.gs, f.file, edited)
	sym := node(t, f.gs, f.symbol)
	sym.Span = &model.SourceSpan{StartLine: 4, EndLine: 6}
	if err := f.gs.UpsertNodes(context.Background(), []model.Node{*sym}); err != nil {
		t.Fatal(err)
	}
	if spec := onlyTask(t, f.svc, KindDecisionDrift); !strings.Contains(spec.Evidence[0].Text, "const maxRetries = 3") {
		t.Errorf("after re-index the evidence should be fetchAll again, got:\n%s", spec.Evidence[0].Text)
	}
}

func TestSupportedDecisionIsNotRequeuedByAnUnrelatedEdit(t *testing.T) {
	f := seedDecisionInRoot(t)
	ctx := context.Background()
	spec := onlyTask(t, f.svc, KindDecisionDrift)
	if _, err := f.svc.Submit(ctx, Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Supported, Quote: "const maxRetries = 3", Confidence: 0.9,
	}); err != nil {
		t.Fatal(err)
	}
	if specs := pending(t, f.svc, ""); len(specs) != 0 {
		t.Fatalf("verified link was requeued: %+v", specs)
	}

	// An edit below fetchAll leaves its lines alone.
	edited := clientSrc + "\nfunc third() {}\n"
	writeFile(t, f.file, edited)
	indexFile(t, f.gs, f.file, edited)
	if specs := pending(t, f.svc, ""); len(specs) != 0 {
		t.Fatalf("an edit outside the symbol requeued it: %+v", specs)
	}
}

func TestDriftedAddsAContradictsEdgeThatSupportedLaterClears(t *testing.T) {
	f := seedDecisionInRoot(t)
	ctx := context.Background()
	drifted := strings.Replace(clientSrc, "maxRetries = 3", "maxRetries = 5", 1)
	writeFile(t, f.file, drifted)
	indexFile(t, f.gs, f.file, drifted)
	spec := onlyTask(t, f.svc, KindDecisionDrift)

	rep, err := f.svc.Submit(ctx, Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Drifted,
		Quote: "const maxRetries = 5", Note: "retries are capped at 5, not 3", Confidence: 0.9,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !rep.Applied {
		t.Errorf("report = %+v, want applied", rep)
	}
	c := edge(t, f.gs, f.symbol, f.decision, model.EdgeContradicts)
	if c == nil || c.Provenance != model.ProvInferred || model.AttrString(c.Attrs, "note") != "retries are capped at 5, not 3" {
		t.Fatalf("CONTRADICTS edge = %+v", c)
	}
	if edge(t, f.gs, f.decision, f.symbol, model.EdgeDecidedIn) == nil {
		t.Error("drift removed the decision's link; the decision is still about this symbol")
	}
	if specs := pending(t, f.svc, ""); len(specs) != 0 {
		t.Fatalf("recorded drift was requeued: %+v", specs)
	}

	writeFile(t, f.file, clientSrc)
	indexFile(t, f.gs, f.file, clientSrc)
	spec = onlyTask(t, f.svc, KindDecisionDrift)
	if _, err := f.svc.Submit(ctx, Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Supported, Quote: "const maxRetries = 3", Confidence: 0.9,
	}); err != nil {
		t.Fatal(err)
	}
	if edge(t, f.gs, f.symbol, f.decision, model.EdgeContradicts) != nil {
		t.Error("CONTRADICTS edge outlived the drift")
	}
}

func TestDriftedNeedsANote(t *testing.T) {
	f := seedDecisionInRoot(t)
	spec := onlyTask(t, f.svc, KindDecisionDrift)
	if _, err := f.svc.Submit(context.Background(), Result{
		TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Drifted, Quote: "const maxRetries = 3", Confidence: 0.9,
	}); err == nil {
		t.Fatal("drifted with no note was accepted")
	}
}

func TestMislinkedRemovesTheLinkForGood(t *testing.T) {
	f := seedDecisionInRoot(t)
	ctx := context.Background()
	spec := onlyTask(t, f.svc, KindDecisionDrift)

	if _, err := f.svc.Submit(ctx, Result{TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Mislinked, Confidence: 0.9}); err != nil {
		t.Fatal(err)
	}
	if edge(t, f.gs, f.decision, f.symbol, model.EdgeDecidedIn) != nil {
		t.Fatal("mislinked verdict left the link in place")
	}
	if _, err := meeting.NewLinkService(f.gs, 0.5).Relink(ctx); err != nil {
		t.Fatal(err)
	}
	if edge(t, f.gs, f.decision, f.symbol, model.EdgeDecidedIn) != nil {
		t.Error("Relink restored a link the verifier rejected")
	}
	if specs := pending(t, f.svc, ""); len(specs) != 0 {
		t.Fatalf("want nothing pending, got %+v", specs)
	}
}

func TestLowConfidenceMislinkedKeepsTheLink(t *testing.T) {
	f := seedDecisionInRoot(t)
	spec := onlyTask(t, f.svc, KindDecisionDrift)
	rep, err := f.svc.Submit(context.Background(), Result{TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Mislinked, Confidence: 0.3})
	if err != nil {
		t.Fatal(err)
	}
	if rep.Applied || edge(t, f.gs, f.decision, f.symbol, model.EdgeDecidedIn) == nil {
		t.Errorf("a low-confidence mislinked verdict was applied (report %+v)", rep)
	}
	if specs := pending(t, f.svc, ""); len(specs) != 0 {
		t.Fatalf("recorded verdict was requeued: %+v", specs)
	}
}

func TestPendingFiltersByKindAndLimit(t *testing.T) {
	f := seedDecisionInRoot(t)
	seedConcept(t, f.gs)

	if got := pending(t, f.svc, ""); len(got) != 2 {
		t.Fatalf("want both kinds, got %+v", got)
	}
	if got := pending(t, f.svc, KindDecisionDrift); len(got) != 1 || got[0].Kind != KindDecisionDrift {
		t.Errorf("kind filter returned %+v", got)
	}
	limited, err := f.svc.Pending(context.Background(), "", 1)
	if err != nil || len(limited) != 1 {
		t.Errorf("limit 1 returned %d task(s), %v", len(limited), err)
	}
}

// --- review fixes ---

// Only the Go extractor records where a function ends. Every other symbol has a start line alone, and one
// declaration line is not enough to judge a decision against.
func TestDecisionEvidenceCoversTheBodyWhenTheSpanHasNoEnd(t *testing.T) {
	f := seedDecisionInRoot(t)
	ctx := context.Background()
	sym := node(t, f.gs, f.symbol)
	sym.Span = &model.SourceSpan{StartLine: 3}
	other := model.Node{
		ID: model.NewNodeID(sym.SourceURI, "other"), Kind: model.KindSymbol, Label: "other", SourceURI: sym.SourceURI,
		Span: &model.SourceSpan{StartLine: 7}, Provenance: model.ProvExtracted, Confidence: 1,
	}
	if err := f.gs.UpsertNodes(ctx, []model.Node{*sym, other}); err != nil {
		t.Fatal(err)
	}

	got := onlyTask(t, f.svc, KindDecisionDrift).Evidence[0].Text
	if !strings.Contains(got, "const maxRetries = 3") {
		t.Errorf("evidence stops at the declaration line:\n%s", got)
	}
	if strings.Contains(got, "func other") {
		t.Errorf("evidence runs into the next symbol:\n%s", got)
	}

	// With no later symbol in the file, the body runs to the end of it.
	if _, err := store.DeleteNodes(ctx, f.gs, []model.Node{other}); err != nil {
		t.Fatal(err)
	}
	if got := onlyTask(t, f.svc, KindDecisionDrift).Evidence[0].Text; !strings.Contains(got, "const maxRetries = 3") {
		t.Errorf("evidence for the last symbol stops at the declaration line:\n%s", got)
	}
}

func TestLowConfidenceVerdictDoesNotLoseTheDemotion(t *testing.T) {
	gs := newStore(t)
	seedConcept(t, gs)
	svc := New(gs, t.TempDir(), 0.5)
	ctx := context.Background()
	submit := func(r Result) {
		t.Helper()
		spec := onlyTask(t, svc, KindConceptSupport)
		r.TaskID, r.Fingerprint = spec.ID, spec.Fingerprint
		if _, err := svc.Submit(ctx, r); err != nil {
			t.Fatal(err)
		}
	}
	submit(Result{Verdict: Unsupported, Confidence: 0.9})

	setSection(t, gs, "Maybe a JWT is involved somewhere here.")
	submit(Result{Verdict: Supported, Quote: "a JWT is involved", Confidence: 0.3})
	if node(t, gs, conceptID).Provenance != model.ProvAmbiguous {
		t.Fatal("an unsure supported verdict lifted the demotion")
	}

	setSection(t, gs, "Every request carries a JWT again.")
	submit(Result{Verdict: Supported, Quote: "carries a JWT again", Confidence: 0.9})
	if got := node(t, gs, conceptID).Provenance; got != model.ProvInferred {
		t.Errorf("provenance = %s: the concept stayed hidden after a confident supported verdict", got)
	}
}

// The verifier only sees what the task carries. "Nothing here supports it" proves nothing about text left out.
func TestNegativeVerdictIsNotAppliedOnTruncatedEvidence(t *testing.T) {
	ctx := context.Background()
	t.Run("more sections than a task carries", func(t *testing.T) {
		gs := newStore(t)
		seedConcept(t, gs)
		for i := range maxEvidence {
			id := fmt.Sprintf("extra%d", i)
			_ = gs.UpsertNodes(ctx, []model.Node{{ID: id, Kind: model.KindDocSection, Label: id, SourceURI: "file://doc.md",
				Attrs: map[string]any{"text": "unrelated text"}, Provenance: model.ProvExtracted, Confidence: 1}})
			_ = gs.UpsertEdges(ctx, []model.Edge{{ID: model.NewEdgeID(id, conceptID, model.EdgeDescribes), Src: id, Dst: conceptID,
				Kind: model.EdgeDescribes, Provenance: model.ProvInferred}})
		}
		svc := New(gs, t.TempDir(), 0.5)
		spec := onlyTask(t, svc, KindConceptSupport)
		if !spec.Truncated || len(spec.Evidence) != maxEvidence {
			t.Fatalf("spec truncated=%v with %d evidence items, want truncated with %d", spec.Truncated, len(spec.Evidence), maxEvidence)
		}
		rep, err := svc.Submit(ctx, Result{TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9})
		if err != nil {
			t.Fatal(err)
		}
		if rep.Applied || node(t, gs, conceptID).Provenance != model.ProvInferred {
			t.Errorf("unsupported was applied on truncated evidence (report %+v)", rep)
		}
	})
	t.Run("section longer than a task carries", func(t *testing.T) {
		gs := newStore(t)
		seedConcept(t, gs)
		setSection(t, gs, strings.Repeat("filler ", maxEvidenceChars)+"and finally the JWT.")
		svc := New(gs, t.TempDir(), 0.5)
		if spec := onlyTask(t, svc, KindConceptSupport); !spec.Truncated {
			t.Error("a clipped section did not mark the task truncated")
		}
	})
	t.Run("whole evidence fits", func(t *testing.T) {
		gs := newStore(t)
		seedConcept(t, gs)
		if spec := onlyTask(t, New(gs, t.TempDir(), 0.5), KindConceptSupport); spec.Truncated {
			t.Error("a task with all of its evidence is marked truncated")
		}
	})
}

// The delete rule has to survive the normal flow, where a changed section is extracted again before it is
// verified again.
func TestSecondUnsupportedStillDeletesAfterTheOwnerExtractsAgain(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	setSection(t, gs, authText)
	extractor := subagent.New(gs, nil, 0.5)
	extract := func(description string) {
		t.Helper()
		if _, err := extractor.Submit(ctx, subagent.SubmitResult{TaskID: sectionID,
			Concepts: []subagent.ConceptResult{{Label: "OAuth", Description: description, Confidence: 0.9}}}); err != nil {
			t.Fatal(err)
		}
	}
	extract("the auth protocol")
	svc := New(gs, t.TempDir(), 0.5)
	spec := onlyTask(t, svc, KindConceptSupport)
	if _, err := svc.Submit(ctx, Result{TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9}); err != nil {
		t.Fatal(err)
	}

	setSection(t, gs, "We now authenticate with mutual TLS.")
	extract("the auth protocol in use")
	if n := node(t, gs, spec.ID); n.Provenance != model.ProvAmbiguous {
		t.Fatalf("re-extraction lifted the verifier's demotion: %+v", n)
	}
	spec = onlyTask(t, svc, KindConceptSupport)
	if _, err := svc.Submit(ctx, Result{TaskID: spec.ID, Fingerprint: spec.Fingerprint, Verdict: Unsupported, Confidence: 0.9}); err != nil {
		t.Fatal(err)
	}
	if node(t, gs, spec.ID) != nil {
		t.Error("concept survived a second unsupported verdict because it had been extracted again in between")
	}
}

// A section that merely reads like a concept (an edge from `chiton embed --link-similar`) never claimed it.
func TestSimilarSectionIsNotEvidence(t *testing.T) {
	gs := newStore(t)
	ctx := context.Background()
	seedConcept(t, gs)
	_ = gs.UpsertNodes(ctx, []model.Node{{ID: "lookalike", Kind: model.KindDocSection, Label: "Tokens", SourceURI: "file://other.md",
		Attrs: map[string]any{"text": "Tokens of every kind."}, Provenance: model.ProvExtracted, Confidence: 1}})
	_ = gs.UpsertEdges(ctx, []model.Edge{{ID: model.NewEdgeID(conceptID, "lookalike", model.EdgeSimilarTo), Src: conceptID, Dst: "lookalike",
		Kind: model.EdgeSimilarTo, Provenance: model.ProvInferred, Confidence: 0.9}})

	spec := onlyTask(t, New(gs, t.TempDir(), 0.5), KindConceptSupport)
	if len(spec.Evidence) != 1 || spec.Evidence[0].NodeID != sectionID {
		t.Errorf("evidence = %+v, want only the section that describes the concept", spec.Evidence)
	}
}
