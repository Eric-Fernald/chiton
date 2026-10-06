//go:build windows

// Command ramprobe measures what chiton's build, embed and serve processes hold in memory for a corpus, for
// the memory-capacity table in docs/BENCHMARK.md. Windows only: it reads process counters through
// K32GetProcessMemoryInfo.
//
//	ramprobe <chiton.exe> <corpus dir> <store dir> <label>
//
// It deletes the parent of <store dir>, builds and embeds the corpus there (hash provider, 768 dimensions),
// swaps in dense embedding-shaped vectors, re-runs embed to save the HNSW index, then starts serve, runs
// semantic_search and query_graph over MCP stdio, and prints a JSON summary. RAMPROBE_SERVE_ONLY=1 measures
// serve on an existing store; RAMPROBE_NO_SETTLE=1 skips the five idle minutes before the final sample.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math/rand"
	"net/url"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"
	"unsafe"

	"github.com/Eric-Fernald/chiton/internal/graph/model"
	"github.com/Eric-Fernald/chiton/internal/graph/store"
	badgerstore "github.com/Eric-Fernald/chiton/internal/graph/store/badger"
)

const (
	killAt = 30 << 30 // stop a child whose working set passes 30 GiB
	settle = 5 * time.Minute
)

type memCounters struct {
	CB                         uint32
	PageFaultCount             uint32
	PeakWorkingSetSize         uintptr
	WorkingSetSize             uintptr
	QuotaPeakPagedPoolUsage    uintptr
	QuotaPagedPoolUsage        uintptr
	QuotaPeakNonPagedPoolUsage uintptr
	QuotaNonPagedPoolUsage     uintptr
	PagefileUsage              uintptr
	PeakPagefileUsage          uintptr
	PrivateUsage               uintptr
}

var procGetMem = syscall.NewLazyDLL("kernel32.dll").NewProc("K32GetProcessMemoryInfo")

func memOf(h syscall.Handle) memCounters {
	var m memCounters
	m.CB = uint32(unsafe.Sizeof(m))
	procGetMem.Call(uintptr(h), uintptr(unsafe.Pointer(&m)), uintptr(m.CB))
	return m
}

func gib(b uintptr) string { return fmt.Sprintf("%.2f", float64(b)/(1<<30)) }

type stage struct {
	Name      string
	Wall      time.Duration
	PeakWS    uintptr // peak working set
	PeakPriv  uintptr // peak committed private bytes
	SteadyWS  uintptr // serve only: after queries
	SteadyPrv uintptr
}

func env(store string) []string {
	return append(os.Environ(),
		"CHITON_GRAPH_PATH="+store,
		"CHITON_PROJECT_DIR="+filepath.Dir(store),
		"CHITON_EMBED_PROVIDER=hash",
		"CHITON_EMBED_DIM=768",
		"CHITON_PLAINTEXT_OK=1",
	)
}

// run runs bin args to completion and reports its peak memory.
func run(name, bin, storeDir string, args ...string) stage {
	cmd := exec.Command(bin, args...)
	cmd.Env = env(storeDir)
	cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
	start := time.Now()
	must(cmd.Start())
	h, err := syscall.OpenProcess(0x1000, false, uint32(cmd.Process.Pid))
	must(err)
	defer syscall.CloseHandle(h)
	done := make(chan struct{})
	go watchdog(h, cmd.Process, done)
	err = cmd.Wait()
	close(done)
	if err != nil {
		fail("%s: %v", name, err)
	}
	m := memOf(h)
	return stage{Name: name, Wall: time.Since(start), PeakWS: m.PeakWorkingSetSize, PeakPriv: m.PeakPagefileUsage}
}

func watchdog(h syscall.Handle, p *os.Process, done <-chan struct{}) {
	t := time.NewTicker(500 * time.Millisecond)
	defer t.Stop()
	for {
		select {
		case <-done:
			return
		case <-t.C:
			if memOf(h).WorkingSetSize > killAt {
				fmt.Fprintln(os.Stderr, "ramprobe: child passed the memory cap, killing it")
				p.Kill()
				return
			}
		}
	}
}

var queries = []string{
	"where do we retry failed http requests",
	"parse a configuration file",
	"compress data with gzip",
	"handle authentication tokens",
	"open a database connection",
	"format a timestamp",
	"walk a directory tree",
	"encode json",
	"read environment variables",
	"cancel a context on timeout",
}

// serve starts chiton serve, waits for its vector index, runs semantic_search queries over MCP stdio, and
// reports memory after them.
func serve(bin, storeDir string) (stage, []time.Duration, string) {
	cmd := exec.Command(bin, "serve", "--stdio")
	cmd.Env = env(storeDir)
	stdin, _ := cmd.StdinPipe()
	stdout, _ := cmd.StdoutPipe()
	stderr, _ := cmd.StderrPipe()
	start := time.Now()
	must(cmd.Start())
	h, err := syscall.OpenProcess(0x1000, false, uint32(cmd.Process.Pid))
	must(err)
	defer syscall.CloseHandle(h)
	done := make(chan struct{})
	go watchdog(h, cmd.Process, done)
	defer func() { close(done); stdin.Close(); cmd.Process.Kill(); cmd.Wait() }()

	ready := make(chan string, 1)
	var once sync.Once
	go func() {
		sc := bufio.NewScanner(stderr)
		background := false
		for sc.Scan() {
			line := sc.Text()
			fmt.Fprintln(os.Stderr, "  serve:", line)
			switch {
			case strings.Contains(line, "semantic_search enabled") && strings.Contains(line, "in the background"):
				background = true
			case strings.Contains(line, "semantic_search enabled") && !background,
				strings.Contains(line, "hnsw index ready"),
				strings.Contains(line, "semantic_search disabled"):
				once.Do(func() { ready <- line })
			}
		}
	}()
	var idxLine string
	select {
	case idxLine = <-ready:
	case <-time.After(2 * time.Hour):
		fail("serve never became ready")
	}
	startup := time.Since(start)

	rd := bufio.NewReaderSize(stdout, 1<<20)
	send := func(v any) {
		b, _ := json.Marshal(v)
		stdin.Write(append(b, '\n'))
	}
	recv := func() map[string]any {
		line, err := rd.ReadBytes('\n')
		if err != nil && err != io.EOF {
			fail("read: %v", err)
		}
		var m map[string]any
		json.Unmarshal(line, &m)
		return m
	}
	send(map[string]any{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": map[string]any{
		"protocolVersion": "2024-11-05", "capabilities": map[string]any{},
		"clientInfo": map[string]any{"name": "ramprobe", "version": "0"}}})
	recv()
	send(map[string]any{"jsonrpc": "2.0", "method": "notifications/initialized"})
	var lat []time.Duration
	for i, q := range queries {
		t0 := time.Now()
		send(map[string]any{"jsonrpc": "2.0", "id": i + 1, "method": "tools/call",
			"params": map[string]any{"name": "semantic_search", "arguments": map[string]any{"query": q}}})
		resp := recv()
		lat = append(lat, time.Since(t0))
		if res, ok := resp["result"].(map[string]any); !ok || res["isError"] == true {
			b, _ := json.Marshal(resp)
			fmt.Fprintf(os.Stderr, "ramprobe: query %q failed: %.300s\n", q, b)
		}
	}
	// query_graph builds the read-only label index on first use, so include it in the steady state.
	t0 := time.Now()
	send(map[string]any{"jsonrpc": "2.0", "id": 100, "method": "tools/call",
		"params": map[string]any{"name": "query_graph", "arguments": map[string]any{"query": "Retry"}}})
	recv()
	fmt.Fprintf(os.Stderr, "ramprobe: first query_graph %s\n", time.Since(t0))

	// Go hands freed heap back to the OS gradually, so sample again once serve has sat idle.
	for t := time.Duration(0); t < settle && os.Getenv("RAMPROBE_NO_SETTLE") == ""; t += 30 * time.Second {
		m := memOf(h)
		fmt.Fprintf(os.Stderr, "ramprobe: idle %s ws %s GiB priv %s GiB\n", t, gib(m.WorkingSetSize), gib(m.PrivateUsage))
		time.Sleep(30 * time.Second)
	}
	m := memOf(h)
	return stage{Name: "serve", Wall: startup, PeakWS: m.PeakWorkingSetSize, PeakPriv: m.PeakPagefileUsage,
		SteadyWS: m.WorkingSetSize, SteadyPrv: m.PrivateUsage}, lat, idxLine
}

type counts struct {
	Nodes, Edges, Files, Vectors int
	SourceBytes                  int64
	ByKind                       map[model.NodeKind]int
}

func count(storeDir string) counts {
	ctx := context.Background()
	gs, err := badgerstore.OpenReadOnlyWith(storeDir, badgerstore.Options{})
	must(err)
	defer gs.Close()
	nk, ek, err := store.CountByKind(ctx, gs)
	must(err)
	c := counts{ByKind: nk}
	for _, n := range nk {
		c.Nodes += n
	}
	for _, n := range ek {
		c.Edges += n
	}
	files, err := store.NodesMatching(ctx, gs, store.NodeFilter{Kinds: []model.NodeKind{model.KindFile}})
	must(err)
	c.Files = len(files)
	for _, f := range files {
		if fi, err := os.Stat(uriPath(f.SourceURI)); err == nil {
			c.SourceBytes += fi.Size()
		}
	}
	ids, err := store.EmbeddingIDs(ctx, gs)
	must(err)
	c.Vectors = len(ids)
	return c
}

// densify replaces every stored vector with a dense, embedding-shaped one (see internal/vector/datagen_test.go).
// The hash embedder's vectors are sparse and compress on disk, so they would understate disk use and give
// HNSW unrealistic duplicates; dense vectors of the same dimension cost what real embeddings cost.
func densify(storeDir string) {
	ctx := context.Background()
	gs, err := badgerstore.OpenWith(storeDir, badgerstore.Options{})
	must(err)
	defer gs.Close()
	ids, err := store.EmbeddingIDs(ctx, gs)
	must(err)
	rng := rand.New(rand.NewSource(1))
	const dim, latent = 768, 48
	proj := make([][]float32, dim)
	for i := range proj {
		proj[i] = make([]float32, latent)
		for j := range proj[i] {
			proj[i][j] = float32(rng.NormFloat64())
		}
	}
	centers := make([][]float64, 2000)
	for i := range centers {
		centers[i] = make([]float64, latent)
		for j := range centers[i] {
			centers[i][j] = rng.NormFloat64() * 2
		}
	}
	z := make([]float64, latent)
	batch := make([]store.Embedding, 0, 4096)
	for k, id := range ids {
		c := centers[rng.Intn(len(centers))]
		for j := range z {
			z[j] = c[j] + rng.NormFloat64()
		}
		v := make([]float32, dim)
		for i, row := range proj {
			var s float64
			for j, a := range row {
				s += float64(a) * z[j]
			}
			v[i] = float32(s) + float32(rng.NormFloat64())*4
		}
		batch = append(batch, store.Embedding{NodeID: id, Vec: v})
		if len(batch) == cap(batch) || k == len(ids)-1 {
			must(gs.UpsertEmbeddings(ctx, batch))
			batch = batch[:0]
		}
	}
}

func uriPath(uri string) string {
	if u, err := url.Parse(uri); err == nil && u.Scheme == "file" {
		p := u.Path
		if len(p) > 2 && p[0] == '/' && p[2] == ':' {
			p = p[1:]
		}
		return filepath.FromSlash(p)
	}
	return uri
}

func dirSize(dir string) (n int64) {
	filepath.WalkDir(dir, func(_ string, d os.DirEntry, err error) error {
		if err == nil && !d.IsDir() {
			if fi, err := d.Info(); err == nil {
				n += fi.Size()
			}
		}
		return nil
	})
	return n
}

func main() {
	if len(os.Args) != 5 {
		fail("usage: ramprobe <chiton.exe> <corpus> <store> <label>")
	}
	bin, corpus, storeDir, label := os.Args[1], os.Args[2], os.Args[3], os.Args[4]
	if os.Getenv("RAMPROBE_SERVE_ONLY") != "" {
		s, lat, idxLine := serve(bin, storeDir)
		fmt.Println(idxLine, "peak ws", gib(s.PeakWS), "peak priv", gib(s.PeakPriv), "now ws", gib(s.SteadyWS), "now priv", gib(s.SteadyPrv), lat)
		return
	}
	must(os.RemoveAll(filepath.Dir(storeDir)))
	must(os.MkdirAll(filepath.Dir(storeDir), 0o755))

	b := run("build", bin, storeDir, "build", corpus)
	e := run("embed", bin, storeDir, "embed")
	densify(storeDir)
	e2 := run("embed-index", bin, storeDir, "embed")
	e.Wall += e2.Wall
	e.PeakWS, e.PeakPriv = max(e.PeakWS, e2.PeakWS), max(e.PeakPriv, e2.PeakPriv)
	c := count(storeDir)
	diskBytes := dirSize(filepath.Dir(storeDir))
	s, lat, idxLine := serve(bin, storeDir)

	var sum time.Duration
	worst := time.Duration(0)
	for _, l := range lat {
		sum += l
		worst = max(worst, l)
	}
	out := map[string]any{
		"label": label, "corpus": corpus,
		"nodes": c.Nodes, "edges": c.Edges, "files": c.Files, "vectors": c.Vectors, "by_kind": c.ByKind,
		"source_bytes": c.SourceBytes, "source_tokens": c.SourceBytes / 4,
		"tokens_per_vector": float64(c.SourceBytes/4) / float64(max(c.Vectors, 1)),
		"disk_gib":          float64(diskBytes) / (1 << 30),
		"build_s":           b.Wall.Seconds(), "build_peak_ws_gib": gib(b.PeakWS), "build_peak_priv_gib": gib(b.PeakPriv),
		"embed_s": e.Wall.Seconds(), "embed_peak_ws_gib": gib(e.PeakWS), "embed_peak_priv_gib": gib(e.PeakPriv),
		"serve_startup_s": s.Wall.Seconds(), "serve_ws_gib": gib(s.SteadyWS), "serve_priv_gib": gib(s.SteadyPrv),
		"serve_peak_ws_gib": gib(s.PeakWS), "serve_peak_priv_gib": gib(s.PeakPriv),
		"query_mean_ms": float64(sum.Microseconds()) / float64(len(lat)) / 1000, "query_max_ms": float64(worst.Microseconds()) / 1000,
		"index": idxLine,
	}
	j, _ := json.MarshalIndent(out, "", "  ")
	fmt.Println(string(j))
}

func must(err error) {
	if err != nil {
		fail("%v", err)
	}
}

func fail(f string, a ...any) {
	fmt.Fprintf(os.Stderr, "ramprobe: "+f+"\n", a...)
	os.Exit(1)
}
