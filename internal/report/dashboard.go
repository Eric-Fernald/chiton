package report

import (
	"fmt"
	"io"
	"strings"

	"github.com/Eric-Fernald/chiton/internal/graph/cluster"
	"github.com/Eric-Fernald/chiton/internal/graph/model"
)

// DashboardFinding is the render-ready projection of an analyze finding for the
// dashboard's findings explorer. It mirrors analyze.FindingView but is declared
// here so the report package stays free of an analyze dependency (report only
// knows graph model + cluster, exactly like the other visual exports). The CLI
// converts analyze.FindingView into this shape, supplying the community of the
// affected node so the explorer can filter by it.
type DashboardFinding struct {
	ID         string
	RuleID     string
	Category   string
	Severity   string
	Message    string
	TargetURI  string
	Line       int    // start line; 0 when the finding has no span
	Fix        string // optional fix suggestion
	Provenance string // EXTRACTED | INFERRED | AMBIGUOUS
	Community  int    // community of the affected node; -1 when unknown
	NodeID     string // affected node id, for cross-linking into the graph view
}

// DashboardData is the full payload rendered into the static dashboard: the
// graph (optionally overlaid with a cluster pass) plus the findings explorer.
type DashboardData struct {
	// Title is shown in the <title>/header.
	Title string
	// Nodes/Edges are the graph; they are projected through buildView and
	// capped by MaxNodes exactly like the interactive HTML export.
	Nodes []model.Node
	Edges []model.Edge
	// Overlay is an optional cluster pass for community colours + betweenness.
	Overlay *cluster.Results
	// Findings is the (already filtered/sorted) finding set to explore. May be
	// empty — the explorer renders an empty state.
	Findings []DashboardFinding
	// MaxNodes caps the rendered graph (0 = no cap), same semantics as
	// VisualOptions.MaxNodes.
	MaxNodes int
}

// GenerateDashboard writes a self-contained, read-only dashboard page to w: an
// interactive graph view (vis-network) and a findings explorer with severity /
// category / community / file / provenance filters. As with GenerateHTML the
// node, edge, and finding data are embedded inline (so the file opens straight
// from disk over file://), and only the vis-network library is loaded from a
// CDN. Findings cross-link into the graph: clicking a finding focuses its
// affected node when that node survived the MaxNodes cap.
func GenerateDashboard(w io.Writer, data DashboardData) error {
	views, keptEdges := buildView(data.Nodes, data.Edges, data.Overlay, data.MaxNodes)

	// Community count reflects the full scoped graph (pre-cap), not the rendered
	// view: so a small --max-nodes doesn't shrink the reported total, and the
	// header stays consistent with the graph-wide community each finding is
	// tagged with in the explorer. A finding may still reference a community
	// whose nodes were all trimmed from the rendered graph — its [graph] link
	// then degrades gracefully (see focusNode).
	commSet := map[int]struct{}{}
	if data.Overlay != nil {
		for _, n := range data.Nodes {
			if r, ok := data.Overlay.Nodes[n.ID]; ok && r.CommunityID >= 0 {
				commSet[r.CommunityID] = struct{}{}
			}
		}
	}

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

	df := make([]dashFinding, 0, len(data.Findings))
	for _, f := range data.Findings {
		df = append(df, dashFinding{
			ID:   f.ID,
			Rule: f.RuleID,
			Cat:  f.Category,
			Sev:  f.Severity,
			Msg:  f.Message,
			URI:  f.TargetURI,
			Line: f.Line,
			Fix:  f.Fix,
			Prov: f.Provenance,
			Comm: f.Community,
			Node: f.NodeID,
		})
	}

	nodesJSON, err := marshalForScript(hn)
	if err != nil {
		return fmt.Errorf("marshal nodes: %w", err)
	}
	edgesJSON, err := marshalForScript(he)
	if err != nil {
		return fmt.Errorf("marshal edges: %w", err)
	}
	findingsJSON, err := marshalForScript(df)
	if err != nil {
		return fmt.Errorf("marshal findings: %w", err)
	}

	title := data.Title
	if title == "" {
		title = "chiton dashboard"
	}

	page := strings.NewReplacer(
		"__TITLE__", htmlTextEscape(title),
		"__NODE_COUNT__", fmt.Sprintf("%d", len(hn)),
		"__EDGE_COUNT__", fmt.Sprintf("%d", len(he)),
		"__COMM_COUNT__", fmt.Sprintf("%d", len(commSet)),
		"__FINDING_COUNT__", fmt.Sprintf("%d", len(df)),
		"__NODES__", nodesJSON,
		"__EDGES__", edgesJSON,
		"__FINDINGS__", findingsJSON,
	).Replace(dashboardTemplate)

	_, err = io.WriteString(w, page)
	return err
}

// dashFinding is the compact JSON shape embedded for the findings explorer.
// Field names are short because the array can be large.
type dashFinding struct {
	ID   string `json:"id"`
	Rule string `json:"rule"`
	Cat  string `json:"cat"`
	Sev  string `json:"sev"`
	Msg  string `json:"msg"`
	URI  string `json:"uri"`
	Line int    `json:"line"`
	Fix  string `json:"fix,omitempty"`
	Prov string `json:"prov,omitempty"`
	Comm int    `json:"comm"`
	Node string `json:"node,omitempty"`
}

// dashboardTemplate is the self-contained dashboard page. Placeholders
// (__NODES__, __FINDINGS__, …) are substituted by GenerateDashboard. The
// vis-network UMD bundle is the only external resource; all data, styling, and
// interaction are inline. The data placeholders are filled with HTML-safe JSON
// (see marshalForScript) so untrusted node/finding text cannot break out of the
// <script> block.
const dashboardTemplate = `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<!-- Defense-in-depth: data is already HTML-safe (marshalForScript) and the DOM
     is built with textContent/createTextNode, never innerHTML, so there is no
     active injection sink. This CSP is a second layer: it pins the only allowed
     remote script to the vis-network CDN, forbids object/base-uri, and blocks
     network egress. 'unsafe-inline' is required because the page's one script
     and styles are inline. -->
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src https://unpkg.com 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; connect-src 'none'; base-uri 'none'; object-src 'none'">
<title>__TITLE__</title>
<script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
<style>
  :root { --bg:#14161a; --panel:#1b1e24; --border:#2a2e36; --fg:#e6e6e6; --muted:#9aa0aa; --accent:#6db3f2; }
  html, body { margin:0; height:100%; font-family:system-ui, sans-serif; background:var(--bg); color:var(--fg); }
  body { display:flex; flex-direction:column; }
  header { padding:10px 16px; border-bottom:1px solid var(--border); background:var(--panel); }
  header h1 { font-size:16px; margin:0 0 6px; }
  .stats { color:var(--muted); font-size:13px; display:flex; flex-wrap:wrap; gap:6px 14px; align-items:center; }
  .pill { display:inline-block; padding:1px 8px; border-radius:10px; font-size:12px; color:#14161a; font-weight:600; }
  .sev-critical { background:#ff5c5c; } .sev-high { background:#ff9e3d; } .sev-medium { background:#ffd166; }
  .sev-low { background:#7ec4ff; } .sev-info { background:#b8bec9; }
  nav { display:flex; gap:4px; padding:8px 16px 0; border-bottom:1px solid var(--border); background:var(--panel); }
  nav button { background:none; border:none; color:var(--muted); padding:8px 14px; font-size:14px; cursor:pointer;
    border-bottom:2px solid transparent; }
  nav button.active { color:var(--fg); border-bottom-color:var(--accent); }
  main { flex:1; position:relative; overflow:hidden; }
  .tab { position:absolute; inset:0; display:none; }
  .tab.active { display:block; }
  /* graph tab */
  #graph { width:100%; height:100%; }
  #gpanel { position:absolute; top:10px; left:10px; z-index:10; background:rgba(20,22,26,0.92);
    border:1px solid var(--border); border-radius:8px; padding:10px 12px; font-size:13px; max-width:320px; }
  #gpanel input { width:100%; box-sizing:border-box; padding:5px 7px; border-radius:5px; border:1px solid var(--border);
    background:#0e1013; color:var(--fg); }
  #legend { margin-top:8px; line-height:1.6; }
  #legend span.dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:5px; }
  #detail { margin-top:8px; color:var(--muted); min-height:1.4em; word-break:break-all; }
  /* findings tab */
  #findings { display:flex; flex-direction:column; height:100%; }
  #filters { display:flex; flex-wrap:wrap; gap:8px 12px; padding:12px 16px; border-bottom:1px solid var(--border);
    align-items:center; }
  #filters label { font-size:12px; color:var(--muted); display:flex; flex-direction:column; gap:3px; }
  #filters select, #filters input { background:#0e1013; color:var(--fg); border:1px solid var(--border);
    border-radius:5px; padding:4px 6px; font-size:13px; }
  #fcount { margin-left:auto; color:var(--muted); font-size:13px; align-self:flex-end; }
  #ftablewrap { flex:1; overflow:auto; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th, td { text-align:left; padding:6px 12px; border-bottom:1px solid var(--border); vertical-align:top; }
  th { position:sticky; top:0; background:var(--panel); color:var(--muted); font-weight:600; cursor:pointer;
    white-space:nowrap; }
  tbody tr:hover { background:rgba(109,179,242,0.07); }
  td.loc { color:var(--muted); font-family:ui-monospace, monospace; word-break:break-all; max-width:360px; }
  td.msg { max-width:520px; }
  td .gobtn { color:var(--accent); cursor:pointer; font-size:12px; }
  td .fix { display:block; color:var(--muted); margin-top:3px; font-size:12px; }
  .empty { padding:24px; color:var(--muted); }
  .prov { font-size:11px; color:var(--muted); }
  a { color:var(--accent); }
</style>
</head>
<body>
<header>
  <h1>__TITLE__</h1>
  <div class="stats" id="stats">
    <span>__NODE_COUNT__ nodes</span><span>__EDGE_COUNT__ edges</span>
    <span>__COMM_COUNT__ communities</span><span>__FINDING_COUNT__ findings</span>
    <span id="sevpills"></span>
  </div>
</header>
<nav>
  <button id="tab-graph-btn" class="active" data-tab="graph">Graph</button>
  <button id="tab-findings-btn" data-tab="findings">Findings</button>
</nav>
<main>
  <section id="tab-graph" class="tab active">
    <div id="gpanel">
      <input id="search" type="text" placeholder="Search node label… (Enter to focus)">
      <div id="legend"></div>
      <div id="detail">Click a node for details.</div>
    </div>
    <div id="graph"></div>
  </section>
  <section id="tab-findings" class="tab">
    <div id="findings">
      <div id="filters">
        <label>Severity
          <select id="f-sev"><option value="">all</option><option>critical</option><option>high</option>
            <option>medium</option><option>low</option><option>info</option></select>
        </label>
        <label>Category <select id="f-cat"><option value="">all</option></select></label>
        <label>Community <select id="f-comm"><option value="">all</option></select></label>
        <label>Provenance <select id="f-prov"><option value="">all</option></select></label>
        <label>File <input id="f-file" type="text" placeholder="path substring"></label>
        <span id="fcount"></span>
      </div>
      <div id="ftablewrap">
        <table id="ftable">
          <thead><tr>
            <th data-sort="sev">Severity</th><th data-sort="rule">Rule</th>
            <th data-sort="cat">Category</th><th data-sort="uri">Location</th>
            <th data-sort="comm">Comm</th><th>Message</th>
          </tr></thead>
          <tbody id="fbody"></tbody>
        </table>
        <div id="fempty" class="empty" style="display:none">No findings match the current filters.</div>
      </div>
    </div>
  </section>
</main>
<script>
  var rawNodes = __NODES__;
  var rawEdges = __EDGES__;
  var rawFindings = __FINDINGS__;

  var SEV_RANK = { critical:4, high:3, medium:2, low:1, info:0 };

  // ---- header severity pills -------------------------------------------------
  (function () {
    var counts = {};
    rawFindings.forEach(function (f) { counts[f.sev] = (counts[f.sev] || 0) + 1; });
    var order = ["critical", "high", "medium", "low", "info"];
    var el = document.getElementById("sevpills");
    order.forEach(function (s) {
      if (!counts[s]) return;
      var span = document.createElement("span");
      span.className = "pill sev-" + s;
      span.textContent = counts[s] + " " + s;
      el.appendChild(span);
    });
  })();

  // ---- tab switching ---------------------------------------------------------
  var graphReady = false;
  function showTab(name) {
    document.querySelectorAll("nav button").forEach(function (b) {
      b.classList.toggle("active", b.dataset.tab === name);
    });
    document.getElementById("tab-graph").classList.toggle("active", name === "graph");
    document.getElementById("tab-findings").classList.toggle("active", name === "findings");
    if (name === "graph") { initGraph(); if (network) { network.redraw(); network.fit(); } }
  }
  document.querySelectorAll("nav button").forEach(function (b) {
    b.addEventListener("click", function () { showTab(b.dataset.tab); });
  });

  // ---- graph view (vis-network) ---------------------------------------------
  var network = null;
  function commColor(c) {
    if (c < 0) return "#6b7280";
    var hue = (c * 137) % 360;
    return "hsl(" + hue + ",62%,58%)";
  }
  var kindBorder = {
    file:"#ffffff", symbol:"#9ad0ff", concept:"#ffd27f", doc_section:"#c4f0c5",
    image:"#f5b3ff", meeting:"#ffb3b3", decision:"#ffe08a", finding:"#ff7a7a",
    rule:"#d0d0d0", url:"#b3d9ff", person:"#e0e0e0"
  };
  var nodesDS = null;
  function initGraph() {
    if (graphReady) return;
    graphReady = true;
    var maxDeg = 1;
    rawNodes.forEach(function (n) { if (n.deg > maxDeg) maxDeg = n.deg; });
    var nodes = rawNodes.map(function (n) {
      var size = 8 + 22 * Math.sqrt(n.deg / maxDeg);
      return {
        id:n.id, label:n.label, value:n.deg, shape:"dot", size:size,
        color:{ background:commColor(n.comm), border:kindBorder[n.kind] || "#888",
                highlight:{ background:commColor(n.comm), border:"#fff" } },
        title:n.kind + " · " + n.label + "\n" + n.uri + "\ncommunity " + n.comm + " · degree " + n.deg,
        _kind:n.kind, _uri:n.uri, _comm:n.comm, _deg:n.deg, _btw:n.btw
      };
    });
    var edges = rawEdges.map(function (e) {
      return { from:e.from, to:e.to, label:e.kind, arrows:"to",
               font:{ size:9, color:"#7a808a", strokeWidth:0 }, color:{ color:"#3a3f48", opacity:0.7 } };
    });
    nodesDS = new vis.DataSet(nodes);
    var edgesDS = new vis.DataSet(edges);
    network = new vis.Network(document.getElementById("graph"), { nodes:nodesDS, edges:edgesDS }, {
      physics:{
        solver:"barnesHut",
        // Weaker repulsion + stronger central gravity keeps sparse components
        // in frame; 200 iterations never converged, so the layout used to go
        // live mid-flight and jitter. Freeze below once it settles.
        barnesHut:{ gravitationalConstant:-3000, centralGravity:0.6, springLength:110,
                    springConstant:0.05, damping:0.6, avoidOverlap:0.15 },
        maxVelocity:30, minVelocity:0.75,
        stabilization:{ enabled:true, iterations:1500, updateInterval:50, fit:true }
      },
      interaction:{ hover:true, tooltipDelay:120 },
      nodes:{ font:{ color:"#e6e6e6", size:12 } }
    });
    network.once("stabilizationIterationsDone", function () {
      network.setOptions({ physics:false });
      network.fit();
    });

    var kinds = {};
    rawNodes.forEach(function (n) { kinds[n.kind] = true; });
    var legend = document.getElementById("legend");
    legend.innerHTML = "";
    Object.keys(kinds).sort().forEach(function (k) {
      var row = document.createElement("div");
      row.innerHTML = '<span class="dot" style="background:' + (kindBorder[k] || "#888") + '"></span>';
      row.appendChild(document.createTextNode(k));
      legend.appendChild(row);
    });

    var detail = document.getElementById("detail");
    network.on("click", function (params) {
      if (params.nodes.length) {
        var n = nodesDS.get(params.nodes[0]);
        detail.textContent = "";
        var b = document.createElement("b"); b.textContent = n.label; detail.appendChild(b);
        detail.appendChild(document.createElement("br"));
        detail.appendChild(document.createTextNode(
          n._kind + " · " + n._uri + " · community " + n._comm +
          " · degree " + n._deg + " · betweenness " + n._btw.toFixed(4)));
      } else {
        detail.textContent = "Click a node for details.";
      }
    });

    document.getElementById("search").addEventListener("keydown", function (ev) {
      if (ev.key !== "Enter") return;
      var q = ev.target.value.trim().toLowerCase();
      if (!q) return;
      var hit = nodes.find(function (n) { return n.label.toLowerCase().indexOf(q) >= 0; });
      if (hit) { focusNode(hit.id); }
    });
  }
  function focusNode(id) {
    initGraph();
    if (!nodesDS || !nodesDS.get(id)) return false;
    network.selectNodes([id]);
    network.focus(id, { scale:1.2, animation:true });
    network.emit("click", { nodes:[id], edges:[] });
    return true;
  }

  // ---- findings explorer -----------------------------------------------------
  var sortKey = "sev", sortAsc = false;
  function buildSelect(id, values, isComm) {
    var sel = document.getElementById(id);
    values.forEach(function (v) {
      var o = document.createElement("option");
      o.value = String(v);
      // value stays raw so the filter comparison matches f.cat/f.prov exactly;
      // display is lowercased so provenance (EXTRACTED/INFERRED) reads like the
      // lowercase pill in the rows (categories are already lowercase → no-op).
      o.textContent = isComm ? (v < 0 ? "none" : "community " + v) : String(v).toLowerCase();
      sel.appendChild(o);
    });
  }
  (function initFilters() {
    var cats = {}, provs = {}, comms = {};
    rawFindings.forEach(function (f) {
      if (f.cat) cats[f.cat] = true;
      if (f.prov) provs[f.prov] = true;
      comms[f.comm] = true;
    });
    buildSelect("f-cat", Object.keys(cats).sort(), false);
    buildSelect("f-prov", Object.keys(provs).sort(), false);
    var commVals = Object.keys(comms).map(Number).sort(function (a, b) { return a - b; });
    buildSelect("f-comm", commVals, true);
    ["f-sev", "f-cat", "f-comm", "f-prov"].forEach(function (id) {
      document.getElementById(id).addEventListener("change", renderFindings);
    });
    document.getElementById("f-file").addEventListener("input", renderFindings);
    document.querySelectorAll("#ftable th[data-sort]").forEach(function (th) {
      th.addEventListener("click", function () {
        var k = th.dataset.sort;
        if (sortKey === k) { sortAsc = !sortAsc; } else { sortKey = k; sortAsc = (k !== "sev"); }
        renderFindings();
      });
    });
  })();

  function currentFilter() {
    return {
      sev: document.getElementById("f-sev").value,
      cat: document.getElementById("f-cat").value,
      comm: document.getElementById("f-comm").value,
      prov: document.getElementById("f-prov").value,
      file: document.getElementById("f-file").value.trim().toLowerCase()
    };
  }
  function renderFindings() {
    var fl = currentFilter();
    var rows = rawFindings.filter(function (f) {
      if (fl.sev && f.sev !== fl.sev) return false;
      if (fl.cat && f.cat !== fl.cat) return false;
      if (fl.comm !== "" && String(f.comm) !== fl.comm) return false;
      if (fl.prov && f.prov !== fl.prov) return false;
      if (fl.file && (f.uri || "").toLowerCase().indexOf(fl.file) < 0) return false;
      return true;
    });
    rows.sort(function (a, b) {
      var av, bv;
      if (sortKey === "sev") { av = SEV_RANK[a.sev] || 0; bv = SEV_RANK[b.sev] || 0; }
      else if (sortKey === "comm") { av = a.comm; bv = b.comm; }
      else if (sortKey === "uri") {
        // The Location column shows uri:line — sort by uri, tie-break by line
        // so same-file findings read top-to-bottom.
        av = (a.uri || "").toLowerCase(); bv = (b.uri || "").toLowerCase();
        if (av === bv) { av = a.line; bv = b.line; }
      }
      else { av = (a[sortKey] || "").toLowerCase(); bv = (b[sortKey] || "").toLowerCase(); }
      if (av < bv) return sortAsc ? -1 : 1;
      if (av > bv) return sortAsc ? 1 : -1;
      return 0;
    });

    var body = document.getElementById("fbody");
    body.textContent = "";
    rows.forEach(function (f) {
      var tr = document.createElement("tr");
      tr.appendChild(sevCell(f.sev));
      tr.appendChild(textCell(f.rule));
      tr.appendChild(textCell(f.cat));
      tr.appendChild(locCell(f));
      tr.appendChild(textCell(f.comm < 0 ? "—" : String(f.comm)));
      tr.appendChild(msgCell(f));
      body.appendChild(tr);
    });
    document.getElementById("fcount").textContent = "showing " + rows.length + " of " + rawFindings.length;
    document.getElementById("fempty").style.display = rows.length ? "none" : "block";
  }
  function sevCell(sev) {
    var td = document.createElement("td");
    var span = document.createElement("span");
    span.className = "pill sev-" + sev;
    span.textContent = sev;
    td.appendChild(span);
    return td;
  }
  function textCell(text) {
    var td = document.createElement("td");
    td.textContent = text == null ? "" : String(text);
    return td;
  }
  function locCell(f) {
    var td = document.createElement("td");
    td.className = "loc";
    var loc = f.uri || "";
    if (f.line) loc += ":" + f.line;
    td.textContent = loc;
    return td;
  }
  function msgCell(f) {
    var td = document.createElement("td");
    td.className = "msg";
    td.appendChild(document.createTextNode(f.msg || ""));
    if (f.node) {
      td.appendChild(document.createTextNode(" "));
      var btn = document.createElement("span");
      btn.className = "gobtn";
      btn.textContent = "[graph]";
      btn.title = "Show the affected node in the graph";
      btn.addEventListener("click", function () {
        showTab("graph");
        if (!focusNode(f.node)) {
          document.getElementById("detail").textContent =
            "That node was trimmed from the rendered graph (raise --max-nodes to include it).";
        }
      });
      td.appendChild(btn);
    }
    if (f.fix) {
      var fix = document.createElement("span");
      fix.className = "fix";
      fix.textContent = "fix: " + f.fix;
      td.appendChild(fix);
    }
    if (f.prov && f.prov !== "EXTRACTED") {
      var prov = document.createElement("span");
      prov.className = "fix prov";
      prov.textContent = f.prov.toLowerCase();
      td.appendChild(prov);
    }
    return td;
  }

  // initial render
  initGraph();
  renderFindings();
</script>
</body>
</html>
`
