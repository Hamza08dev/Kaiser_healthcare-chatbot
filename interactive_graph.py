"""
Interactive strategy mind‑map rendering using vis-network.

This module converts the structured strategy data (root, pillars,
initiatives, KPIs) into an interactive, clickable graph rendered
inside Streamlit via components.html.
"""

from typing import Dict, Any, List
import json


def _build_nodes_and_edges(structure: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert strategy structure into vis-network node/edge lists."""
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    root_label = structure.get("root", "2026 Strategy")
    root_id = "root"

    # Root node
    nodes.append(
        {
            "id": root_id,
            "label": root_label,
            "level": 0,
            "shape": "box",
            "color": {"background": "#1f2937", "border": "#60a5fa"},
            "font": {"color": "#e5e7eb", "size": 18, "face": "Segoe UI"},
        }
    )

    pillars = structure.get("pillars", [])
    for idx, pillar in enumerate(pillars):
        pillar_id = f"pillar_{idx}"
        pillar_name = pillar.get("name", "Pillar")

        nodes.append(
            {
                "id": pillar_id,
                "label": pillar_name,
                "level": 1,
                "shape": "box",
                "color": {"background": "#111827", "border": "#34d399"},
                "font": {"color": "#e5e7eb", "size": 16},
            }
        )
        edges.append({"from": root_id, "to": pillar_id})

        # Initiatives
        for jdx, initiative in enumerate(pillar.get("initiatives", [])[:4]):
            init_id = f"init_{idx}_{jdx}"
            init_label = str(initiative)
            nodes.append(
                {
                    "id": init_id,
                    "label": init_label,
                    "level": 2,
                    "shape": "box",
                    "color": {"background": "#020617", "border": "#fbbf24"},
                    "font": {"color": "#e5e7eb", "size": 13},
                }
            )
            edges.append({"from": pillar_id, "to": init_id})

        # KPIs
        for kdx, kpi in enumerate(pillar.get("kpis", [])[:4]):
            kpi_id = f"kpi_{idx}_{kdx}"
            kpi_label = str(kpi)
            nodes.append(
                {
                    "id": kpi_id,
                    "label": kpi_label,
                    "level": 3,
                    "shape": "box",
                    "color": {"background": "#0b1120", "border": "#f97316"},
                    "font": {"color": "#e5e7eb", "size": 12},
                }
            )
            edges.append({"from": pillar_id, "to": kpi_id})

    return {"nodes": nodes, "edges": edges}


def build_interactive_mindmap_html(structure: Dict[str, Any]) -> str:
    """
    Build HTML/JS for an interactive mind‑map style strategy graph.

    - Click nodes to see their details in a side panel.
    - Uses vis-network from CDN (no extra Python dependencies).
    """
    data = _build_nodes_and_edges(structure)
    data_json = json.dumps(data)

    # HTML with dark theme to match the main app
    html = f"""
    <div style="display:flex; flex-direction:row; gap:16px; height:650px;">
      <div id="mindmap" style="flex:2; height:100%; background-color:#020617; border-radius:12px; border:1px solid #1f2937;"></div>
      <div id="node-info" style="flex:1; height:100%; background-color:#020617; border-radius:12px; border:1px solid #1f2937; padding:16px; color:#e5e7eb; font-family: 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif; font-size:14px; overflow-y:auto;">
        <h3 style="margin-top:0; font-size:16px; color:#93c5fd;">Details</h3>
        <p>Click any pillar, initiative, or KPI to see more details here.</p>
      </div>
    </div>

    <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script type="text/javascript">
      const rawData = {data_json};
      const container = document.getElementById('mindmap');
      const nodes = new vis.DataSet(rawData.nodes);
      const edges = new vis.DataSet(rawData.edges);

      const options = {{
        layout: {{
          hierarchical: {{
            enabled: true,
            direction: 'LR',
            levelSeparation: 180,
            nodeSpacing: 140,
            sortMethod: 'hubsize'
          }}
        }},
        physics: false,
        interaction: {{
          hover: true,
          tooltipDelay: 200
        }},
        edges: {{
          color: '#4b5563',
          width: 1.5,
          smooth: true
        }},
        nodes: {{
          borderWidth: 1.5,
          shape: 'box',
          margin: 8
        }}
      }};

      const network = new vis.Network(container, {{ nodes, edges }}, options);

      const infoEl = document.getElementById('node-info');

      function renderInfo(nodeId) {{
        const node = nodes.get(nodeId);
        if (!node) return;

        let levelLabel = 'Element';
        if (node.level === 0) levelLabel = 'Strategy';
        else if (node.level === 1) levelLabel = 'Pillar';
        else if (node.level === 2) levelLabel = 'Initiative';
        else if (node.level === 3) levelLabel = 'KPI';

        infoEl.innerHTML = `
          <h3 style="margin-top:0; font-size:16px; color:#93c5fd;">${{levelLabel}}</h3>
          <p style="white-space:pre-wrap;">${{node.label}}</p>
        `;
      }}

      network.on('selectNode', function(params) {{
        if (params.nodes && params.nodes.length > 0) {{
          renderInfo(params.nodes[0]);
        }}
      }});
    </script>
    """

    return html


