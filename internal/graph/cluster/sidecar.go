package cluster

import (
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

// SidecarPath returns the cluster sidecar path for a graph store path
// (e.g. ./chiton-out/graph.kuzu → ./chiton-out/cluster.json).
func SidecarPath(graphPath string) string {
	return filepath.Join(filepath.Dir(graphPath), "cluster.json")
}

// WriteSidecar atomically writes r to path as indented JSON, so a concurrent reader never sees a partial file.
func WriteSidecar(path string, r *Results) error {
	b, err := json.MarshalIndent(r, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal cluster results: %w", err)
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
	var r Results
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, fmt.Errorf("parse cluster sidecar %s: %w", path, err)
	}
	return &r, nil
}
