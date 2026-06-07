package report

import (
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// VisualOptions controls the graph visual exports (interactive HTML, Mermaid).
type VisualOptions struct {
	// MaxNodes caps how many nodes are rendered. When the graph is larger,
	// the highest-degree (then highest-betweenness) nodes are kept and the
	// rest dropped along with their dangling edges. 0 means no cap.
	MaxNodes int
	// Title is shown in the HTML <title>/heading and as a Mermaid comment.
	Title string
}

// DefaultVisual is the standard configuration for visual exports. The HTML
// renderer (vis-network) stays interactive into the low thousands of nodes;
// Mermaid is capped far lower by GenerateMermaid because the dagre layout and
// browser SVG choke well before that.
var DefaultVisual = VisualOptions{MaxNodes: 600, Title: "chiton graph"}

// mermaidMaxNodes is a hard ceiling for Mermaid output regardless of
// VisualOptions.MaxNodes — Mermaid's layout engine degrades badly past a few
// hundred nodes, so a dense graph is trimmed to its most central nodes.
const mermaidMaxNodes = 160

// viewNode is the render-ready projection of a graph node: only the fields the
// visualizers need, with community/centrality overlaid from a cluster pass when
// one is available.
type viewNode struct {
	ID          string
	Label       string
	Kind        model.NodeKind
	SourceURI   string
	Community   int
	Betweenness float64
	Degree      int
}

// buildView projects nodes+edges (and an optional cluster overlay) into a
// render-ready node set plus the edges whose endpoints both survive the
// MaxNodes cap. Degree is always computed from the supplied edges so the view
// is correct even without a prior cluster run; community/betweenness come from
// the overlay when present (community -1 and betweenness 0 otherwise).
func buildView(nodes []model.Node, edges []model.Edge, overlay *cluster.Results, maxNodes int) ([]viewNode, []model.Edge) {
	degree := make(map[string]int, len(nodes))
	for _, e := range edges {
		degree[e.Src]++
		degree[e.Dst]++
	}

	views := make([]viewNode, 0, len(nodes))
	for _, n := range nodes {
		v := viewNode{
			ID:        n.ID,
			Label:     n.Label,
			Kind:      n.Kind,
			SourceURI: n.SourceURI,
			Community: -1,
			Degree:    degree[n.ID],
		}
		if overlay != nil {
			if nr, ok := overlay.Nodes[n.ID]; ok {
				v.Community = nr.CommunityID
				v.Betweenness = nr.Betweenness
				if nr.Degree > v.Degree {
					// Trust the overlay's degree when it is richer (a full-graph
					// pass may have seen edges a scoped edge slice does not).
					v.Degree = nr.Degree
				}
			}
		}
		views = append(views, v)
	}

	// Rank by degree, then betweenness, then ID for a stable, reproducible cap.
	sort.Slice(views, func(i, j int) bool {
		if views[i].Degree != views[j].Degree {
			return views[i].Degree > views[j].Degree
		}
		if views[i].Betweenness != views[j].Betweenness {
			return views[i].Betweenness > views[j].Betweenness
		}
		return views[i].ID < views[j].ID
	})

	if maxNodes > 0 && len(views) > maxNodes {
		views = views[:maxNodes]
	}

	kept := make(map[string]bool, len(views))
	for _, v := range views {
		kept[v.ID] = true
	}
	keptEdges := make([]model.Edge, 0, len(edges))
	for _, e := range edges {
		if kept[e.Src] && kept[e.Dst] {
			keptEdges = append(keptEdges, e)
		}
	}
	return views, keptEdges
}

// GenerateMermaid writes a Mermaid flowchart of the graph to w, grouping nodes
// into per-community subgraphs. Suitable for embedding in Markdown docs that a
// renderer (GitHub, mkdocs, Obsidian) draws inline. Large graphs are trimmed to
// their most central nodes (see mermaidMaxNodes).
func GenerateMermaid(w io.Writer, nodes []model.Node, edges []model.Edge, overlay *cluster.Results, opts VisualOptions) error {
	cap := opts.MaxNodes
	if cap == 0 || cap > mermaidMaxNodes {
		cap = mermaidMaxNodes
	}
	views, keptEdges := buildView(nodes, edges, overlay, cap)

	// Stable short IDs (n0, n1, …) keyed by node ID.
	id := make(map[string]string, len(views))
	for i, v := range views {
		id[v.ID] = fmt.Sprintf("n%d", i)
	}

	title := opts.Title
	if title == "" {
		title = "chiton graph"
	}
	fmt.Fprintf(w, "%%%% %s — %d nodes, %d edges\n", title, len(views), len(keptEdges))
	fmt.Fprintln(w, "flowchart LR")

	// Group nodes by community so the layout clusters them visually.
	byComm := map[int][]viewNode{}
	commOrder := []int{}
	for _, v := range views {
		if _, seen := byComm[v.Community]; !seen {
			commOrder = append(commOrder, v.Community)
		}
		byComm[v.Community] = append(byComm[v.Community], v)
	}
	sort.Ints(commOrder)

	for _, c := range commOrder {
		members := byComm[c]
		if c >= 0 {
			fmt.Fprintf(w, "  subgraph c%d[\"Community %d\"]\n", c, c)
		} else {
			fmt.Fprintf(w, "  subgraph cNA[\"Unclustered\"]\n")
		}
		for _, v := range members {
			fmt.Fprintf(w, "    %s[\"%s\"]\n", id[v.ID], mermaidLabel(v))
		}
		fmt.Fprintln(w, "  end")
	}

	for _, e := range keptEdges {
		s, ok1 := id[e.Src]
		d, ok2 := id[e.Dst]
		if !ok1 || !ok2 {
			continue
		}
		fmt.Fprintf(w, "  %s -->|%s| %s\n", s, mermaidEdgeLabel(e.Kind), d)
	}
	return nil
}

// mermaidLabel renders a node's display text, escaped for Mermaid's
// quoted-label syntax.
func mermaidLabel(v viewNode) string {
	return mermaidEscape(v.Label)
}

// mermaidEscape neutralises characters that break Mermaid's "..." labels.
// Double quotes become &quot; (Mermaid honours HTML entities inside quotes);
// '#' becomes &#35; because Mermaid otherwise reads a '#…;' run as an HTML
// numeric entity and mangles the label; characters that confuse the parser are
// replaced with spaces. '#' must be replaced before any entity is introduced
// so we don't re-encode the '#' in '&quot;' / '&#35;' (NewReplacer scans the
// input left-to-right and never re-scans its own replacements, so ordering the
// '#' rule first is safe regardless).
func mermaidEscape(s string) string {
	r := strings.NewReplacer(
		"#", "&#35;",
		"\"", "&quot;",
		"`", "'",
		"\n", " ",
		"\r", " ",
	)
	return r.Replace(s)
}

func mermaidEdgeLabel(k model.EdgeKind) string {
	return strings.ToLower(string(k))
}

// htmlNode is the JSON shape embedded into the interactive HTML export, one per
// rendered node. Field names are short because they are repeated per node.
type htmlNode struct {
	ID    string  `json:"id"`
	Label string  `json:"label"`
	Kind  string  `json:"kind"`
	URI   string  `json:"uri"`
	Comm  int     `json:"comm"`
	Deg   int     `json:"deg"`
	Btw   float64 `json:"btw"`
}

type htmlEdge struct {
	From string `json:"from"`
	To   string `json:"to"`
	Kind string `json:"kind"`
}

// GenerateHTML writes a self-contained interactive graph page to w. The node
// and edge data are embedded inline; only the vis-network library is loaded
// from a CDN, so the file is portable and works offline once the library is
// cached. Nodes are coloured by community and sized by degree.
func GenerateHTML(w io.Writer, nodes []model.Node, edges []model.Edge, overlay *cluster.Results, opts VisualOptions) error {
	views, keptEdges := buildView(nodes, edges, overlay, opts.MaxNodes)

	hn := make([]htmlNode, len(views))
	for i, v := range views {
		hn[i] = htmlNode{
			ID:    v.ID,
			Label: v.Label,
			Kind:  string(v.Kind),
			URI:   v.SourceURI,
			Comm:  v.Community,
			Deg:   v.Degree,
			Btw:   v.Betweenness,
		}
	}
	he := make([]htmlEdge, 0, len(keptEdges))
	for _, e := range keptEdges {
		he = append(he, htmlEdge{From: e.Src, To: e.Dst, Kind: strings.ToLower(string(e.Kind))})
	}

	nodesJSON, err := marshalForScript(hn)
	if err != nil {
		return fmt.Errorf("marshal nodes: %w", err)
	}
	edgesJSON, err := marshalForScript(he)
	if err != nil {
		return fmt.Errorf("marshal edges: %w", err)
	}

	title := opts.Title
	if title == "" {
		title = "chiton graph"
	}

	page := strings.NewReplacer(
		"__TITLE__", htmlTextEscape(title),
		"__NODE_COUNT__", fmt.Sprintf("%d", len(hn)),
		"__EDGE_COUNT__", fmt.Sprintf("%d", len(he)),
		"__NODES__", nodesJSON,
		"__EDGES__", edgesJSON,
	).Replace(htmlTemplate)

	_, err = io.WriteString(w, page)
	return err
}

// marshalForScript JSON-encodes v for embedding inside a <script> block.
// json.Marshal escapes '<', '>', and '&' to \u00XX by default (SetEscapeHTML is
// on), so a node label containing "</script>" cannot break out of the block —
// it serialises as "</script>". The extra "</" → "<\/" replacement is
// belt-and-suspenders in case the encoder's HTML escaping is ever disabled.
func marshalForScript(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return strings.ReplaceAll(string(b), "</", "<\\/"), nil
}

// htmlTextEscape escapes the handful of characters that matter for text placed
// in element content / the <title>.
func htmlTextEscape(s string) string {
	r := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;")
	return r.Replace(s)
}

// htmlTemplate is the self-contained page. Placeholders (__NODES__ etc.) are
// substituted by GenerateHTML. The vis-network UMD bundle is the only external
// resource; everything else (data, styling, interaction) is inline.
const htmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  html, body { margin: 0; height: 100%; font-family: system-ui, sans-serif; background: #14161a; color: #e6e6e6; }
  #graph { width: 100%; height: 100%; }
  #panel {
    position: absolute; top: 10px; left: 10px; z-index: 10;
    background: rgba(20,22,26,0.92); border: 1px solid #2a2e36; border-radius: 8px;
    padding: 10px 12px; font-size: 13px; max-width: 320px;
  }
  #panel h1 { font-size: 14px; margin: 0 0 6px; }
  #panel .stats { color: #9aa0aa; margin-bottom: 8px; }
  #search { width: 100%; box-sizing: border-box; padding: 5px 7px; border-radius: 5px;
    border: 1px solid #2a2e36; background: #0e1013; color: #e6e6e6; }
  #legend { margin-top: 8px; line-height: 1.6; }
  #legend span.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 5px; }
  #detail { margin-top: 8px; color: #9aa0aa; min-height: 1.4em; word-break: break-all; }
  a { color: #6db3f2; }
</style>
</head>
<body>
<div id="panel">
  <h1>__TITLE__</h1>
  <div class="stats">__NODE_COUNT__ nodes · __EDGE_COUNT__ edges</div>
  <input id="search" type="text" placeholder="Search node label… (Enter to focus)">
  <div id="legend"></div>
  <div id="detail">Click a node for details.</div>
</div>
<div id="graph"></div>
<script>
  var rawNodes = __NODES__;
  var rawEdges = __EDGES__;

  // Deterministic colour per community id (golden-angle hue spread); -1 = gray.
  function commColor(c) {
    if (c < 0) return "#6b7280";
    var hue = (c * 137) % 360;
    return "hsl(" + hue + ",62%,58%)";
  }
  // Per-kind border colour so node kinds remain distinguishable within a community.
  var kindBorder = {
    file: "#ffffff", symbol: "#9ad0ff", concept: "#ffd27f", doc_section: "#c4f0c5",
    image: "#f5b3ff", meeting: "#ffb3b3", decision: "#ffe08a", finding: "#ff7a7a",
    rule: "#d0d0d0", url: "#b3d9ff", person: "#e0e0e0"
  };

  var maxDeg = 1;
  rawNodes.forEach(function (n) { if (n.deg > maxDeg) maxDeg = n.deg; });

  var nodes = rawNodes.map(function (n) {
    var size = 8 + 22 * Math.sqrt(n.deg / maxDeg);
    return {
      id: n.id, label: n.label, value: n.deg,
      shape: "dot", size: size,
      color: { background: commColor(n.comm), border: kindBorder[n.kind] || "#888",
               highlight: { background: commColor(n.comm), border: "#fff" } },
      title: n.kind + " · " + n.label + "\n" + n.uri + "\ncommunity " + n.comm + " · degree " + n.deg,
      _kind: n.kind, _uri: n.uri, _comm: n.comm, _deg: n.deg, _btw: n.btw
    };
  });
  var edges = rawEdges.map(function (e) {
    return { from: e.from, to: e.to, label: e.kind, arrows: "to",
             font: { size: 9, color: "#7a808a", strokeWidth: 0 }, color: { color: "#3a3f48", opacity: 0.7 } };
  });

  var nodesDS = new vis.DataSet(nodes);
  var edgesDS = new vis.DataSet(edges);
  var container = document.getElementById("graph");
  var network = new vis.Network(container, { nodes: nodesDS, edges: edgesDS }, {
    physics: { stabilization: { iterations: 200 }, barnesHut: { gravitationalConstant: -8000, springLength: 120 } },
    interaction: { hover: true, tooltipDelay: 120 },
    nodes: { font: { color: "#e6e6e6", size: 12 } }
  });

  // Legend: kinds present in this graph.
  var kinds = {};
  rawNodes.forEach(function (n) { kinds[n.kind] = true; });
  var legend = document.getElementById("legend");
  Object.keys(kinds).sort().forEach(function (k) {
    var row = document.createElement("div");
    row.innerHTML = '<span class="dot" style="background:' + (kindBorder[k] || "#888") + '"></span>' + k;
    legend.appendChild(row);
  });

  var detail = document.getElementById("detail");
  network.on("click", function (params) {
    if (params.nodes.length) {
      var n = nodesDS.get(params.nodes[0]);
      detail.innerHTML = "<b>" + n.label + "</b><br>" + n._kind + "<br>" + n._uri +
        "<br>community " + n._comm + " · degree " + n._deg + " · betweenness " + n._btw.toFixed(4);
    } else {
      detail.textContent = "Click a node for details.";
    }
  });

  // Search: focus the first node whose label contains the query (case-insensitive).
  document.getElementById("search").addEventListener("keydown", function (ev) {
    if (ev.key !== "Enter") return;
    var q = ev.target.value.trim().toLowerCase();
    if (!q) return;
    var hit = nodes.find(function (n) { return n.label.toLowerCase().indexOf(q) >= 0; });
    if (hit) {
      network.selectNodes([hit.id]);
      network.focus(hit.id, { scale: 1.2, animation: true });
      network.emit("click", { nodes: [hit.id], edges: [] });
    }
  });
</script>
</body>
</html>
`
