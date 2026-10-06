package hnswlib

import (
	"bytes"
	"fmt"
	"math"
	"math/rand"
	"reflect"
	"strings"
	"testing"
)

func TestWriteToReadFromRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	g1 := New[string]()
	g1.Metric, g1.EfConstruction, g1.LevelMult = UnitCosine, 77, 0.3
	for i := range 400 {
		v := randVec(rng, 12)
		Cosine.Distance(v, v)
		g1.Insert(fmt.Sprintf("node-%d", i), v)
	}
	for _, k := range []string{"node-0", "node-17", "node-399"} {
		g1.Remove(k)
	}

	var buf bytes.Buffer
	n, err := g1.WriteTo(&buf)
	if err != nil || n != int64(buf.Len()) {
		t.Fatalf("WriteTo = %d, %v; buffer holds %d", n, err, buf.Len())
	}
	g2 := &Graph[string]{}
	if _, err := g2.ReadFrom(&buf); err != nil {
		t.Fatal(err)
	}

	if g2.Metric != UnitCosine || g2.M != g1.M || g2.LevelMult != 0.3 || g2.EfSearch != g1.EfSearch ||
		g2.EfConstruction != 77 {
		t.Fatalf("parameters not restored: %v %d %v %d %d", g2.Metric, g2.M, g2.LevelMult, g2.EfSearch, g2.EfConstruction)
	}
	if !reflect.DeepEqual(g1.keys, g2.keys) || !reflect.DeepEqual(g1.vecs, g2.vecs) ||
		!reflect.DeepEqual(g1.links, g2.links) || !reflect.DeepEqual(g1.dead, g2.dead) ||
		!reflect.DeepEqual(g1.index, g2.index) || g1.entry != g2.entry || g1.top != g2.top {
		t.Fatal("decoded graph structure differs")
	}
	for range 20 {
		q := randVec(rng, 12)
		if a, b := g1.Nearest(q, 8), g2.Nearest(q, 8); !reflect.DeepEqual(a, b) {
			t.Fatalf("decoded graph searches differently:\n%v\n%v", a, b)
		}
	}
	checkInvariants(t, g2)
}

func TestRoundTripIntKeys(t *testing.T) {
	g1 := lineGraph(40)
	var buf bytes.Buffer
	if _, err := g1.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	g2 := &Graph[int]{}
	if _, err := g2.ReadFrom(&buf); err != nil {
		t.Fatal(err)
	}
	if got := keys(g2.Nearest([]float32{20.2}, 2)); !reflect.DeepEqual(got, []int{20, 21}) {
		t.Fatalf("Nearest after round trip = %v, want [20 21]", got)
	}
}

func TestReadFromRejectsBadInput(t *testing.T) {
	g := lineGraph(30)
	var buf bytes.Buffer
	if _, err := g.WriteTo(&buf); err != nil {
		t.Fatal(err)
	}
	full := buf.Bytes()

	bad := map[string][]byte{
		"empty":        nil,
		"wrong magic":  append([]byte("not-a-graph"), full[len(magic):]...),
		"version 9":    append([]byte(magic), append([]byte{9}, full[len(magic)+1:]...)...),
		"truncated":    full[:len(full)/3],
		"one byte off": full[:len(full)-1],
	}
	for name, data := range bad {
		target := lineGraph(3)
		if _, err := target.ReadFrom(bytes.NewReader(data)); err == nil {
			t.Errorf("%s: ReadFrom succeeded", name)
		}
		if target.Len() != 3 {
			t.Errorf("%s: a failed ReadFrom changed the graph", name)
		}
	}
	if _, err := (&Graph[string]{}).ReadFrom(bytes.NewReader(full)); err == nil ||
		!strings.Contains(err.Error(), "key kind") {
		t.Errorf("reading int keys into a string graph: err = %v, want a key kind error", err)
	}
}

// TestReadFromRejectsInconsistentGraphs writes graphs whose internal state was corrupted after building
// and checks ReadFrom refuses each one instead of accepting a graph that would panic or misbehave later.
func TestReadFromRejectsInconsistentGraphs(t *testing.T) {
	build := func() *Graph[int] {
		rng := rand.New(rand.NewSource(5))
		g := New[int]()
		g.M = 4
		for i := range 300 {
			g.Insert(i, randVec(rng, 6))
		}
		return g
	}
	var low, high int // a node with only layer 0, and one with layer 1
	for id, levels := range build().links {
		if len(levels) == 1 {
			low = id
		} else {
			high = id
		}
	}
	cases := map[string]func(g *Graph[int]){
		"link to a node lacking that level": func(g *Graph[int]) {
			g.links[high][1] = append(g.links[high][1], int32(low))
		},
		"self link":             func(g *Graph[int]) { g.links[7][0] = append(g.links[7][0], 7) },
		"too many links":        func(g *Graph[int]) { g.links[7][0] = make([]int32, 2*g.M+1) },
		"ragged vectors":        func(g *Graph[int]) { g.vecs[9] = []float32{1, 2} },
		"empty vector":          func(g *Graph[int]) { g.vecs[9] = nil },
		"duplicate live key":    func(g *Graph[int]) { g.keys[9] = g.keys[10] },
		"NaN level multiplier":  func(g *Graph[int]) { g.LevelMult = math.NaN() },
		"Inf level multiplier":  func(g *Graph[int]) { g.LevelMult = math.Inf(1) },
		"huge level multiplier": func(g *Graph[int]) { g.LevelMult = 1e300 },
		"zero M":                func(g *Graph[int]) { g.M = 0 },
		"huge M":                func(g *Graph[int]) { g.M = 1 << 40 },
		"huge EfSearch":         func(g *Graph[int]) { g.EfSearch = 1 << 40 },
	}
	for name, corrupt := range cases {
		g := build()
		corrupt(g)
		var buf bytes.Buffer
		if _, err := g.WriteTo(&buf); err != nil {
			t.Fatalf("%s: WriteTo: %v", name, err)
		}
		target := lineGraph(3)
		if _, err := target.ReadFrom(&buf); err == nil {
			t.Errorf("%s: ReadFrom accepted it", name)
		}
		if target.Len() != 3 || target.M != 6 {
			t.Errorf("%s: a failed ReadFrom changed the graph", name)
		}
	}
}

// FuzzReadFrom checks that no byte stream ReadFrom accepts yields a graph that panics when searched or
// extended. Run it with `go test -fuzz FuzzReadFrom ./internal/vector/hnswlib/`.
func FuzzReadFrom(f *testing.F) {
	for _, g := range []*Graph[int]{lineGraph(1), lineGraph(40)} {
		var buf bytes.Buffer
		if _, err := g.WriteTo(&buf); err != nil {
			f.Fatal(err)
		}
		f.Add(buf.Bytes())
	}
	f.Fuzz(func(t *testing.T, data []byte) {
		g := &Graph[int]{}
		if _, err := g.ReadFrom(bytes.NewReader(data)); err != nil {
			return
		}
		if d := g.Dim(); d > 0 {
			q := make([]float32, d)
			q[0] = 1
			g.Nearest(q, 3)
			g.Insert(-1, q)
			g.Nearest(q, 3)
		}
	})
}

func TestWriteToRejectsUnsupportedKeys(t *testing.T) {
	g := New[float64]()
	g.Insert(1.5, []float32{1})
	if _, err := g.WriteTo(&bytes.Buffer{}); err == nil {
		t.Fatal("WriteTo with float64 keys succeeded")
	}
}
