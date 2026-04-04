from collections import defaultdict
from nodes.registry import NODE_REGISTRY

NODE_PRIORITY = {
    "document_upload":    0,
    "document_intelligence_ocr": 0,
    "document_parser":    0,
    "google_vision_ocr":  0,
    "text_input":         0,
    "ocr_confidence_gate": 1,
    "phi_detector":       1,
    "compliance_enforcer": 1,
    "rag_tm":             2,
    "vector_db":          2,
    "glossary":           3,
    "llm_agent":          4,
    "azure_translate":    4,
    "translation":        4,
    "phi_restore":        5,
    "compliance":         5,
    "comet_qe":           5,
    "document_rebuilder": 6,
    "output":             7,
    "document_output":    7,
    "learning":           8,
}

# These node types must always run BEFORE llm_agent
SUPPORT_NODE_TYPES = {"rag_tm", "vector_db"}
TERMINAL_NODE_TYPES = {"output", "document_output"}


def build_execution_order(nodes: list[dict], edges: list[dict]) -> list[str]:
    node_ids = {n["id"] for n in nodes}
    node_type_map = {
        n["id"]: n.get("data", {}).get("nodeType", "") for n in nodes
    }

    # Separate main-flow edges from sub-handle edges
    main_edges = []
    sub_edges = []
    for edge in edges:
        if edge.get("sourceHandle"):
            sub_edges.append(edge)
        else:
            main_edges.append(edge)

    # Build adjacency only from MAIN flow edges
    # Sub-handle edges (llm→rag) encode support relationships, not execution order
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    adjacency: dict[str, list[str]] = defaultdict(list)

    for edge in main_edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src in node_ids and tgt in node_ids:
            # Skip edges where target is a support node — they don't depend on LLM
            if node_type_map.get(tgt) in SUPPORT_NODE_TYPES:
                continue
            # Skip edges where source is a support node in main flow
            if node_type_map.get(src) in SUPPORT_NODE_TYPES:
                continue
            adjacency[src].append(tgt)
            in_degree[tgt] += 1

    # Support nodes have no in-degree — they start ready and priority sorts them first
    def priority(nid: str) -> int:
        return NODE_PRIORITY.get(node_type_map.get(nid, ""), 99)

    ready = sorted(
        [nid for nid, deg in in_degree.items() if deg == 0],
        key=priority,
    )

    order = []
    while ready:
        nid = ready.pop(0)
        order.append(nid)
        for neighbour in adjacency[nid]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                ready.append(neighbour)
                ready.sort(key=priority)

    if len(order) != len(node_ids):
        raise ValueError("Workflow graph has a cycle — cannot execute.")

    return order


async def execute_workflow(
    nodes: list[dict],
    edges: list[dict],
    initial_context: dict,
) -> dict:
    order = build_execution_order(nodes, edges)
    node_map = {n["id"]: n for n in nodes}
    context = {**initial_context, "_logs": []}

    for node_id in order:
        node_def = node_map[node_id]
        node_data = node_def.get("data", {})
        node_type = node_data.get("nodeType", "")
        node_config = node_data.get("config", {})

        if context.get("_stop_workflow") and node_type not in TERMINAL_NODE_TYPES:
            context["_logs"].append({
                "node_id": node_id,
                "node_type": node_type,
                "status": "skipped",
                "reason": "Workflow was stopped by an earlier gate node.",
            })
            continue

        NodeClass = NODE_REGISTRY.get(node_type)

        if NodeClass is None:
            context["_logs"].append({
                "node_id": node_id,
                "node_type": node_type,
                "status": "skipped",
                "reason": f"No handler for nodeType '{node_type}'",
            })
            continue

        instance = NodeClass(node_id=node_id, config=node_config)
        context = await instance.execute(context)

    return context
