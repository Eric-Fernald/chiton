package vector_test

import (
	"context"
	"fmt"
	"math/rand"
	"strings"
	"testing"

	"github.com/Eric-Fernald/chiton/internal/embed"
	"github.com/Eric-Fernald/chiton/internal/vector"
)

// TestHNSWRecallHashText is T2's text half: recall@8 against Flat on short documents embedded by the
// offline hash embedder. Documents draw most words from one of 60 topics, like code and doc sections do.
func TestHNSWRecallHashText(t *testing.T) {
	if testing.Short() {
		t.Skip("embeds and indexes 6k documents")
	}
	rng := rand.New(rand.NewSource(7))
	doc := func() string {
		topic := rng.Intn(60)
		words := make([]string, 14)
		for i := range words {
			if rng.Float64() < 0.7 {
				words[i] = fmt.Sprintf("t%dw%d", topic, rng.Intn(40))
			} else {
				words[i] = fmt.Sprintf("common%d", rng.Intn(300))
			}
		}
		return strings.Join(words, " ")
	}
	texts := make([]string, 6000)
	for i := range texts {
		texts[i] = doc()
	}
	queries := make([]string, 150)
	for i := range queries {
		queries[i] = doc()
	}

	emb := embed.NewHashEmbedder(embed.DefaultHashDim)
	vecs, err := emb.Embed(context.Background(), texts)
	if err != nil {
		t.Fatal(err)
	}
	qvecs, err := emb.Embed(context.Background(), queries)
	if err != nil {
		t.Fatal(err)
	}
	items := make([]vector.Item, len(vecs))
	for i, v := range vecs {
		items[i] = vector.Item{ID: fmt.Sprintf("d%05d", i), Vec: v}
	}
	f, h := vector.NewFlat(), vector.NewHNSW(vector.HNSWConfig{Seed: 1})
	f.Add(items)
	h.Add(items)

	var hit, total int
	for _, q := range qvecs {
		want := map[string]bool{}
		for _, r := range f.Search(q, 8) {
			want[r.ID] = true
		}
		for _, r := range h.Search(q, 8) {
			if want[r.ID] {
				hit++
			}
		}
		total += len(want)
	}
	if r := float64(hit) / float64(total); r < 0.95 {
		t.Fatalf("recall@8 = %.3f, want >= 0.95", r)
	}
}
