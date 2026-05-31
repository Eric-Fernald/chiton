package report

import (
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// Options controls how the report is generated.
type Options struct {
	MaxGodNodes     int
	MaxPerCommunity int
}

// Default is the standard report configuration.
var Default = Options{
	MaxGodNodes:     20,
	MaxPerCommunity: 10,
}

// Generate writes a Markdown graph report to w.
func Generate(w io.Writer, results *cluster.Results, edges []model.Edge, opts Options) error {
	fmt.Fprintf(w, "# Graph Report\n\n")
	fmt.Fprintf(w, "Generated: %s\n\n", results.GeneratedAt.Format("2006-01-02 15:04:05"))
	fmt.Fprintf(w, "**Nodes:** %d  **Communities:** %d\n\n", results.NodeCount, results.CommunityCount)

	type entry struct {
		id string
		r  cluster.NodeResult
	}
	all := make([]entry, 0, len(results.Nodes))
	for id, r := range results.Nodes {
		all = append(all, entry{id, r})
	}
	sort.Slice(all, func(i, j int) bool {
		if all[i].r.Betweenness != all[j].r.Betweenness {
			return all[i].r.Betweenness > all[j].r.Betweenness
		}
		return all[i].id < all[j].id // stable tie-break so output is reproducible
	})

	limit := opts.MaxGodNodes
	if limit > len(all) {
		limit = len(all)
	}

	fmt.Fprintf(w, "## God Nodes (top %d by betweenness)\n\n", limit)
	fmt.Fprintf(w, "| Rank | Label | Community | Betweenness | Degree |\n")
	fmt.Fprintf(w, "|------|-------|-----------|-------------|--------|\n")
	for i := 0; i < limit; i++ {
		e := all[i]
		fmt.Fprintf(w, "| %d | `%s` | %d | %.4f | %d |\n",
			i+1, e.r.Label, e.r.CommunityID, e.r.Betweenness, e.r.Degree)
	}
	fmt.Fprintln(w)

	byComm := map[int][]cluster.NodeResult{}
	for _, r := range results.Nodes {
		byComm[r.CommunityID] = append(byComm[r.CommunityID], r)
	}
	commIDs := make([]int, 0, len(byComm))
	for id := range byComm {
		commIDs = append(commIDs, id)
	}
	sort.Ints(commIDs)

	fmt.Fprintf(w, "## Communities\n\n")
	for _, cid := range commIDs {
		members := byComm[cid]
		sort.Slice(members, func(i, j int) bool {
			if members[i].Betweenness != members[j].Betweenness {
				return members[i].Betweenness > members[j].Betweenness
			}
			return members[i].SourceURI < members[j].SourceURI // stable tie-break
		})
		fmt.Fprintf(w, "### Community %d (%d nodes)\n\n", cid, len(members))
		cap := opts.MaxPerCommunity
		if cap > len(members) {
			cap = len(members)
		}
		for _, m := range members[:cap] {
			fmt.Fprintf(w, "- `%s` (%s)\n", m.Label, m.SourceURI)
		}
		if len(members) > cap {
			fmt.Fprintf(w, "- … and %d more\n", len(members)-cap)
		}
		fmt.Fprintln(w)
	}

	nodeComm := map[string]int{}
	nodeLabel := map[string]string{}
	for id, r := range results.Nodes {
		nodeComm[id] = r.CommunityID
		nodeLabel[id] = r.Label
	}

	type surprisingEdge struct {
		srcLabel string
		dstLabel string
		srcComm  int
		dstComm  int
		kind     model.EdgeKind
	}
	var surprising []surprisingEdge
	seen := map[string]bool{}
	for _, e := range edges {
		sc, ok1 := nodeComm[e.Src]
		dc, ok2 := nodeComm[e.Dst]
		if !ok1 || !ok2 || sc == dc {
			continue
		}
		key := fmt.Sprintf("%s-%s-%s", e.Src, e.Dst, e.Kind)
		if seen[key] {
			continue
		}
		seen[key] = true
		surprising = append(surprising, surprisingEdge{nodeLabel[e.Src], nodeLabel[e.Dst], sc, dc, e.Kind})
	}

	if len(surprising) > 0 {
		const maxSurprising = 20
		if len(surprising) > maxSurprising {
			surprising = surprising[:maxSurprising]
		}
		fmt.Fprintf(w, "## Surprising Cross-Community Connections\n\n")
		fmt.Fprintf(w, "| From | To | Edge | Communities |\n")
		fmt.Fprintf(w, "|------|----|------|-------------|\n")
		for _, s := range surprising {
			fmt.Fprintf(w, "| `%s` | `%s` | %s | %d → %d |\n",
				s.srcLabel, s.dstLabel, strings.ToLower(string(s.kind)), s.srcComm, s.dstComm)
		}
		fmt.Fprintln(w)
	}

	return nil
}
