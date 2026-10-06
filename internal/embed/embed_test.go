package embed

import (
	"context"
	"encoding/json"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
)

func TestHashEmbedderDeterministicAndNormalized(t *testing.T) {
	h := NewHashEmbedder(64)
	if h.Dim() != 64 {
		t.Fatalf("Dim = %d, want 64", h.Dim())
	}

	got, err := h.Embed(context.Background(), []string{"hello world", "hello world"})
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 vectors, got %d", len(got))
	}
	for i, v := range got {
		if len(v) != 64 {
			t.Fatalf("vector %d has dim %d, want 64", i, len(v))
		}
	}

	// Determinism: same input → identical vector.
	for i := range got[0] {
		if got[0][i] != got[1][i] {
			t.Fatalf("hash embedder not deterministic at index %d", i)
		}
	}

	// L2 norm should be ~1 (non-empty text).
	var sum float64
	for _, v := range got[0] {
		sum += float64(v) * float64(v)
	}
	if math.Abs(sum-1.0) > 1e-5 {
		t.Fatalf("vector not unit-normalized: sumsq=%v", sum)
	}
}

// TestBucketAndSignMatchesFNV pins the inline FNV-1a to hash/fnv so previously stored hash embeddings stay valid.
func TestBucketAndSignMatchesFNV(t *testing.T) {
	for _, tok := range []string{"", "a", "retry", "backoff", "ünïcode", "日本語", "x1y2"} {
		for _, dim := range []int{1, 7, 64, 256, 768} {
			hb := fnv.New32a()
			_, _ = hb.Write([]byte(tok))
			wantBucket := int(hb.Sum32()) % dim
			if wantBucket < 0 {
				wantBucket += dim
			}
			hs := fnv.New32a()
			_, _ = hs.Write([]byte("sign:" + tok))
			wantSign := float32(1)
			if hs.Sum32()&1 != 0 {
				wantSign = -1
			}
			if b, s := bucketAndSign(tok, dim); b != wantBucket || s != wantSign {
				t.Fatalf("bucketAndSign(%q, %d) = (%d, %v), want (%d, %v)", tok, dim, b, s, wantBucket, wantSign)
			}
		}
	}
}

func TestHashEmbedderLexicalOverlap(t *testing.T) {
	h := NewHashEmbedder(512)
	vecs, err := h.Embed(context.Background(), []string{
		"retry policy with exponential backoff",
		"exponential backoff retry policy",
		"unrelated banana smoothie recipe",
	})
	if err != nil {
		t.Fatal(err)
	}
	simRelated := dot(vecs[0], vecs[1])
	simUnrelated := dot(vecs[0], vecs[2])
	if simRelated <= simUnrelated {
		t.Fatalf("expected related texts more similar: related=%v unrelated=%v", simRelated, simUnrelated)
	}
}

func TestHashEmbedderEmptyText(t *testing.T) {
	h := NewHashEmbedder(32)
	vecs, err := h.Embed(context.Background(), []string{""})
	if err != nil {
		t.Fatal(err)
	}
	// Empty text → zero vector (norm 0, left as-is).
	for _, v := range vecs[0] {
		if v != 0 {
			t.Fatalf("expected zero vector for empty text, got %v", v)
		}
	}
}

func TestNewFactory(t *testing.T) {
	hashE, err := New(Config{Provider: "hash", Dim: 128})
	if err != nil {
		t.Fatal(err)
	}
	if hashE.Model() != "hash" || hashE.Dim() != 128 {
		t.Fatalf("unexpected hash embedder: model=%s dim=%d", hashE.Model(), hashE.Dim())
	}

	ollamaE, err := New(Config{Provider: ""})
	if err != nil {
		t.Fatal(err)
	}
	if ollamaE.Model() != defaultOllamaModel {
		t.Fatalf("default provider should be ollama, got model %s", ollamaE.Model())
	}

	if _, err := New(Config{Provider: "bogus"}); err == nil {
		t.Fatal("expected error for unknown provider")
	}
}

func TestTextFor(t *testing.T) {
	cases := []struct {
		name string
		node model.Node
		want string
	}{
		{
			name: "symbol with docstring",
			node: model.Node{Kind: model.KindSymbol, Label: "Retry", Attrs: map[string]any{"docstring": "retries with backoff"}},
			want: "Retry\nretries with backoff",
		},
		{
			name: "doc section uses text",
			node: model.Node{Kind: model.KindDocSection, Label: "Auth", Attrs: map[string]any{"text": "we use JWTs"}},
			want: "Auth\nwe use JWTs",
		},
		{
			name: "concept uses description",
			node: model.Node{Kind: model.KindConcept, Label: "Idempotency", Attrs: map[string]any{"description": "safe to retry"}},
			want: "Idempotency\nsafe to retry",
		},
		{
			name: "external stub not embeddable",
			node: model.Node{Kind: model.KindSymbol, Label: "fmt.Println", Attrs: map[string]any{"external": true}},
			want: "",
		},
		{
			name: "file not embeddable",
			node: model.Node{Kind: model.KindFile, Label: "main.go"},
			want: "",
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := TextFor(c.node); got != c.want {
				t.Fatalf("TextFor = %q, want %q", got, c.want)
			}
		})
	}
}

// shortEmbedder drops the last vector of every response.
type shortEmbedder struct{ Embedder }

func (s shortEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	v, err := s.Embedder.Embed(ctx, texts)
	return v[:len(v)-1], err
}

func TestNodeEmbeddingsPairsEmbeddableNodes(t *testing.T) {
	ctx := context.Background()
	nodes := []model.Node{
		{ID: "s", Kind: model.KindSymbol, Label: "Retry"},
		{ID: "f", Kind: model.KindFile, Label: "main.go"}, // not embeddable, so NodeEmbeddings skips it
		{ID: "c", Kind: model.KindConcept, Label: "Idempotency"},
	}
	h := NewHashEmbedder(16)
	embs, exact, err := NodeEmbeddings(ctx, h, nodes)
	if err != nil || !exact || len(embs) != 2 || embs[0].NodeID != "s" || embs[1].NodeID != "c" {
		t.Fatalf("got %+v exact=%v err=%v", embs, exact, err)
	}
	if want, _ := h.Embed(ctx, []string{"Idempotency"}); !slices.Equal(embs[1].Vec, want[0]) {
		t.Fatal("vector not paired with its node's text")
	}

	// NodeEmbeddings truncates and flags a short response. With no embeddable node, it never calls Embed.
	if embs, exact, err := NodeEmbeddings(ctx, shortEmbedder{h}, nodes); err != nil || exact || len(embs) != 1 || embs[0].NodeID != "s" {
		t.Fatalf("short: got %+v exact=%v err=%v", embs, exact, err)
	}
	if embs, exact, err := NodeEmbeddings(ctx, shortEmbedder{h}, nodes[1:2]); err != nil || !exact || embs != nil {
		t.Fatalf("none embeddable: got %+v exact=%v err=%v", embs, exact, err)
	}
}

func TestOllamaRequestShape(t *testing.T) {
	var gotModel string
	var gotInput []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/embed" {
			t.Errorf("unexpected path %s", r.URL.Path)
		}
		body, _ := io.ReadAll(r.Body)
		var req ollamaRequest
		if err := json.Unmarshal(body, &req); err != nil {
			t.Errorf("bad request body: %v", err)
		}
		gotModel = req.Model
		gotInput = req.Input
		// Echo back 3-dim vectors, one per input.
		embs := make([][]float32, len(req.Input))
		for i := range embs {
			embs[i] = []float32{1, 0, 0}
		}
		_ = json.NewEncoder(w).Encode(ollamaResponse{Embeddings: embs})
	}))
	defer srv.Close()

	o := NewOllama(Config{Provider: "ollama", Endpoint: srv.URL, Model: "test-model"})
	vecs, err := o.Embed(context.Background(), []string{"a", "b"})
	if err != nil {
		t.Fatal(err)
	}
	if gotModel != "test-model" {
		t.Fatalf("model = %q, want test-model", gotModel)
	}
	if len(gotInput) != 2 {
		t.Fatalf("input len = %d, want 2", len(gotInput))
	}
	if len(vecs) != 2 || len(vecs[0]) != 3 {
		t.Fatalf("unexpected response shape: %v", vecs)
	}
	// dim should self-correct to the model's true dim.
	if o.Dim() != 3 {
		t.Fatalf("Dim = %d, want 3 after self-correction", o.Dim())
	}
}

func TestOllamaErrorStatus(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusNotFound)
		_ = json.NewEncoder(w).Encode(ollamaResponse{Error: "model not found"})
	}))
	defer srv.Close()

	o := NewOllama(Config{Endpoint: srv.URL, Model: "missing"})
	if _, err := o.Embed(context.Background(), []string{"x"}); err == nil {
		t.Fatal("expected error on non-200 status")
	}
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func TestNodeEmbeddingsTagsModelAndText(t *testing.T) {
	embs, _, err := NodeEmbeddings(context.Background(), NewHashEmbedder(16), []model.Node{{ID: "s", Kind: model.KindSymbol, Label: "Retry"}})
	if err != nil || len(embs) != 1 || embs[0].Model != "hash" || embs[0].TextSHA != TextSHA("Retry") {
		t.Fatalf("got %+v, %v; want model hash and the text's SHA", embs, err)
	}
}

func TestPlan(t *testing.T) {
	sym := func(id, label string) model.Node { return model.Node{ID: id, Kind: model.KindSymbol, Label: label} }
	nodes := []model.Node{
		sym("new", "New"),                                    // no vector yet
		sym("same", "Same"),                                  // current vector
		sym("edited", "Edited"),                              // text changed since its vector
		sym("othermodel", "Other"),                           // embedded by another model
		sym("legacy", "Legacy"),                              // stored before models were recorded
		{ID: "file", Kind: model.KindFile, Label: "main.go"}, // not embeddable, yet has a vector
	}
	have := []store.EmbeddingMeta{
		{NodeID: "same", Model: "m", Dim: 2, TextSHA: TextSHA("Same")},
		{NodeID: "edited", Model: "m", Dim: 2, TextSHA: TextSHA("Old text")},
		{NodeID: "othermodel", Model: "x", Dim: 2, TextSHA: TextSHA("Other")},
		{NodeID: "legacy", Dim: 2},
		{NodeID: "file", Model: "m", Dim: 2},
		{NodeID: "gone", Model: "m", Dim: 2}, // its node was deleted
	}
	work, stale := Plan(nodes, have, "m", false)
	if got := idsOf(work); !slices.Equal(got, []string{"new", "edited", "othermodel", "legacy"}) {
		t.Errorf("work = %v", got)
	}
	if !slices.Equal(stale, []string{"file", "gone"}) {
		t.Errorf("stale = %v, want [file gone]", stale)
	}
	if work, _ := Plan(nodes, have, "m", true); len(work) != 5 {
		t.Errorf("--all work = %v, want every embeddable node", idsOf(work))
	}
}

func TestKeepModel(t *testing.T) {
	embs := []store.Embedding{
		{NodeID: "a", Model: "m", Vec: []float32{1, 0}},
		{NodeID: "b", Vec: []float32{0, 1}},                 // legacy, same dim: kept
		{NodeID: "c", Model: "other", Vec: []float32{1, 1}}, // another model: dropped
		{NodeID: "d", Model: "m", Vec: []float32{1, 0, 0}},  // minority dimension: dropped
		{NodeID: "e", Model: "m"},                           // empty: dropped
	}
	kept, rep := KeepModel(embs, "m")
	var ids []string
	for _, e := range kept {
		ids = append(ids, e.NodeID)
	}
	if !slices.Equal(ids, []string{"a", "b"}) {
		t.Errorf("kept = %v, want [a b]", ids)
	}
	if rep.Legacy != 1 || rep.WrongDim != 1 || len(rep.Other) != 1 || rep.Other["other"] != 1 {
		t.Errorf("report = %+v", rep)
	}
}

func idsOf(nodes []model.Node) []string {
	ids := make([]string, len(nodes))
	for i, n := range nodes {
		ids[i] = n.ID
	}
	return ids
}

func TestPlanPrunesExternalStubVectors(t *testing.T) {
	stub := model.Node{ID: "stub", Kind: model.KindSymbol, Label: "fmt.Println", Attrs: map[string]any{"external": true}}
	have := []store.EmbeddingMeta{{NodeID: "stub", Model: "m", Dim: 2, TextSHA: TextSHA("fmt.Println")}}
	if work, stale := Plan([]model.Node{stub}, have, "m", false); len(work) != 0 || !slices.Equal(stale, []string{"stub"}) {
		t.Fatalf("work=%v stale=%v, want the stub's vector pruned", idsOf(work), stale)
	}
}
