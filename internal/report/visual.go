package report

import (
	"cmp"
	"encoding/json"
	"fmt"
	"io"
	"sort"
	"strconv"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// VisualOptions controls the graph visual exports (interactive HTML, Mermaid).
type VisualOptions struct {
	// MaxNodes caps rendered nodes (0 = no cap), keeping the highest-degree then highest-betweenness
	// nodes and dropping the rest with their dangling edges.
	MaxNodes int
	// Title is shown in the HTML <title>/heading and as a Mermaid comment.
	Title string
}

// DefaultVisual is the standard configuration for visual exports. vis-network stays interactive
// into the low thousands of nodes; Mermaid gets its own, far lower cap (mermaidMaxNodes).
var DefaultVisual = VisualOptions{MaxNodes: 600, Title: "chiton graph"}

// mermaidMaxNodes is a hard ceiling for Mermaid output regardless of VisualOptions.MaxNodes:
// Mermaid's layout degrades badly past a few hundred nodes.
const mermaidMaxNodes = 160

// viewNode is the render-ready projection of a graph node, with community/centrality overlaid
// from a cluster pass when one is available.
type viewNode struct {
	ID          string
	Label       string
	Kind        model.NodeKind
	SourceURI   string
	Community   int
	Betweenness float64
	Degree      int
}

// buildView projects nodes+edges into capped view nodes plus the edges whose endpoints both survive.
// Degree comes from the edges (so no cluster run is needed); overlay-less nodes get community -1.
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

// GenerateMermaid writes a Mermaid flowchart of the graph to w, one subgraph per community, for
// Markdown renderers that draw it inline. Large graphs are trimmed (see mermaidMaxNodes).
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

	fmt.Fprintf(w, "%%%% %s — %d nodes, %d edges\n", cmp.Or(opts.Title, "chiton graph"), len(views), len(keptEdges))
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
			fmt.Fprintf(w, "    %s[\"%s\"]\n", id[v.ID], mermaidEscaper.Replace(v.Label))
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

// mermaidEscaper neutralises characters that break Mermaid's "..." labels. '#' → &#35; because Mermaid
// reads a '#…;' run as an entity; a Replacer never re-scans its output, so &quot; is not re-encoded.
var mermaidEscaper = strings.NewReplacer(
	"#", "&#35;",
	"\"", "&quot;",
	"`", "'",
	"\n", " ",
	"\r", " ",
)

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

// toHTMLData converts the view nodes and kept edges to the JSON shapes embedded in the HTML pages.
func toHTMLData(views []viewNode, edges []model.Edge) ([]htmlNode, []htmlEdge) {
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
	he := make([]htmlEdge, len(edges))
	for i, e := range edges {
		he[i] = htmlEdge{From: e.Src, To: e.Dst, Kind: strings.ToLower(string(e.Kind))}
	}
	return hn, he
}

// GenerateHTML writes a self-contained interactive graph page to w: data inline, only vis-network
// from a CDN. Nodes are coloured by community and sized by degree.
func GenerateHTML(w io.Writer, nodes []model.Node, edges []model.Edge, overlay *cluster.Results, opts VisualOptions) error {
	hn, he := toHTMLData(buildView(nodes, edges, overlay, opts.MaxNodes))
	return writePage(w, htmlTemplate, cmp.Or(opts.Title, "chiton graph"), hn, he)
}

// writePage fills tmpl's graph placeholders (__TITLE__, __NODE_COUNT__, __EDGE_COUNT__, __NODES__, __EDGES__)
// plus the extra old/new pairs, and writes the page to w.
func writePage(w io.Writer, tmpl, title string, hn []htmlNode, he []htmlEdge, extra ...string) error {
	nodesJSON, err := marshalForScript(hn)
	if err != nil {
		return fmt.Errorf("marshal nodes: %w", err)
	}
	edgesJSON, err := marshalForScript(he)
	if err != nil {
		return fmt.Errorf("marshal edges: %w", err)
	}
	// No placeholder is a prefix of another, so pair order can't change the output.
	pairs := append([]string{
		"__TITLE__", htmlTextEscape(title),
		"__NODE_COUNT__", strconv.Itoa(len(hn)),
		"__EDGE_COUNT__", strconv.Itoa(len(he)),
		"__NODES__", nodesJSON,
		"__EDGES__", edgesJSON,
	}, extra...)
	_, err = io.WriteString(w, strings.NewReplacer(pairs...).Replace(tmpl))
	return err
}

// marshalForScript JSON-encodes v for a <script> block. json.Marshal already escapes '<' '>' '&' as
// \u00XX, so "</script>" can't break out; the "</" → "<\/" rewrite is belt-and-braces.
func marshalForScript(v any) (string, error) {
	b, err := json.Marshal(v)
	if err != nil {
		return "", err
	}
	return strings.ReplaceAll(string(b), "</", "<\\/"), nil
}

// htmlTextEscaper escapes the characters that matter for element content and the <title>.
var htmlTextEscaper = strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;")

func htmlTextEscape(s string) string { return htmlTextEscaper.Replace(s) }

// htmlTemplate is the self-contained page GenerateHTML fills in (__NODES__ etc.); the
// vis-network bundle is its only external resource.
const htmlTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- Same defence-in-depth CSP as the dashboard: the graph data is already
     HTML-safe and the DOM is never built from innerHTML, so this is a second
     layer. It pins the only allowed remote script to the vis-network CDN,
     forbids object/base-uri, and blocks network egress so an exported page can
     never phone the inlined graph home. 'unsafe-inline' is required because
     the page's script and styles are inline. -->
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src https://unpkg.com 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; object-src 'none'">
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
  #physics-toggle { display: block; margin-top: 8px; color: #9aa0aa; }
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
  <label id="physics-toggle"><input id="physics" type="checkbox"> physics (unfreeze to rearrange)</label>
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
    physics: {
      solver: "barnesHut",
      barnesHut: {
        // -8000 overpowered central gravity, so low-degree nodes were pushed to
        // an equilibrium far outside the viewport ("floating off into space").
        gravitationalConstant: -3000,
        centralGravity: 0.6,
        springLength: 110,
        springConstant: 0.05,
        damping: 0.6,
        avoidOverlap: 0.15
      },
      maxVelocity: 30,
      minVelocity: 0.75,
      // 200 iterations never converged at this size, so the layout was still
      // mid-flight when it went live and kept jittering.
      stabilization: { enabled: true, iterations: 1500, updateInterval: 50, fit: true }
    },
    interaction: { hover: true, tooltipDelay: 120 },
    nodes: { font: { color: "#e6e6e6", size: 12 } }
  });

  // vis integrates forever by default: every hover, drag or click re-heats the
  // solver and the whole graph drifts. Freeze once stabilized; the checkbox
  // re-heats it deliberately when you do want to rearrange nodes.
  var physicsBox = document.getElementById("physics");
  network.once("stabilizationIterationsDone", function () {
    network.setOptions({ physics: false });
    network.fit();
  });
  physicsBox.addEventListener("change", function () {
    network.setOptions({ physics: physicsBox.checked });
  });

  // Legend: kinds present in this graph.
  var kinds = {};
  rawNodes.forEach(function (n) { kinds[n.kind] = true; });
  var legend = document.getElementById("legend");
  Object.keys(kinds).sort().forEach(function (k) {
    var row = document.createElement("div");
    var dot = document.createElement("span");
    dot.className = "dot";
    dot.style.background = kindBorder[k] || "#888";
    row.appendChild(dot);
    row.appendChild(document.createTextNode(k)); // k is a node kind — never innerHTML it
    legend.appendChild(row);
  });

  var detail = document.getElementById("detail");
  network.on("click", function (params) {
    if (params.nodes.length) {
      var n = nodesDS.get(params.nodes[0]);
      // Build with textContent/createTextNode — node label/uri/kind are
      // ingested, untrusted strings and must never reach innerHTML.
      detail.textContent = "";
      var b = document.createElement("b"); b.textContent = n.label; detail.appendChild(b);
      detail.appendChild(document.createElement("br"));
      detail.appendChild(document.createTextNode(n._kind));
      detail.appendChild(document.createElement("br"));
      detail.appendChild(document.createTextNode(n._uri));
      detail.appendChild(document.createElement("br"));
      detail.appendChild(document.createTextNode(
        "community " + n._comm + " · degree " + n._deg + " · betweenness " + n._btw.toFixed(4)));
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
