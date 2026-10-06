package cluster

import (
	"cmp"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/Eric-Fernald/chiton/internal/fsutil"
)

// Results holds a cluster pass's per-node communities and centrality metrics. It is a JSON sidecar next to
// the graph store so the serve process can load it without write access to the store.
type Results struct {
	GeneratedAt    time.Time             `json:"generated_at"`
	NodeCount      int                   `json:"node_count"`
	CommunityCount int                   `json:"community_count"`
	Nodes          map[string]NodeResult `json:"nodes"`
}

// NodeResult is the per-node output of a cluster pass.
type NodeResult struct {
	Label       string  `json:"label"`
	SourceURI   string  `json:"source_uri"`
	CommunityID int     `json:"community_id"`
	Betweenness float64 `json:"betweenness"`
	Degree      int     `json:"degree"`
}

// CompareByBetweenness orders results by descending betweenness, then label and source URI, so every god-node
// listing (CLI, report, MCP) breaks ties the same way.
func CompareByBetweenness(a, b NodeResult) int {
	return cmp.Or(cmp.Compare(b.Betweenness, a.Betweenness), cmp.Compare(a.Label, b.Label),
		cmp.Compare(a.SourceURI, b.SourceURI))
}

// SidecarPath returns the cluster sidecar path for a graph store path. For example, ./chiton-out/graph.kuzu
// maps to ./chiton-out/cluster.json.
func SidecarPath(graphPath string) string {
	return filepath.Join(filepath.Dir(graphPath), "cluster.json")
}

// WriteSidecar atomically writes r to path as compact JSON, so a concurrent reader never sees a partial file.
func WriteSidecar(path string, r *Results) error {
	b, err := marshalSidecar(r)
	if err != nil {
		return err
	}
	if err := fsutil.WriteFileAtomic(path, b, 0o644); err != nil {
		return fmt.Errorf("write cluster sidecar %s: %w", path, err)
	}
	return nil
}

// LoadSidecar reads the cluster sidecar at path. Returns (nil, nil) when the
// file does not exist so callers can treat a missing sidecar as "not yet run".
func LoadSidecar(path string) (*Results, error) {
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read cluster sidecar %s: %w", path, err)
	}
	r, err := unmarshalSidecar(data)
	if err != nil {
		return nil, fmt.Errorf("parse cluster sidecar %s: %w", path, err)
	}
	return r, nil
}

// marshalSidecar is the sidecar's one encoding seam: compact JSON, which reads back older indented files too.
func marshalSidecar(r *Results) ([]byte, error) {
	b, err := json.Marshal(r)
	if err != nil {
		return nil, fmt.Errorf("marshal cluster results: %w", err)
	}
	return b, nil
}

// unmarshalSidecar decodes marshalSidecar's output.
func unmarshalSidecar(data []byte) (*Results, error) {
	var r Results
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, err
	}
	return &r, nil
}
