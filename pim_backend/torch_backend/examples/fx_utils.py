"""Write exported and partitioned FX graphs without executing them."""

import json


def write_fx_graph(module, destination, labels):
    destination.with_suffix(".txt").write_text(
        str(module.graph) + "\n\n" + module.code, encoding="utf-8",
    )
    lines = ["digraph FX {", "rankdir=LR;", "node [shape=box];"]
    for node in module.graph.nodes:
        label = f"{node.name}\n{node.op}: {node.target}"
        if node.name in labels:
            label += "\n" + labels[node.name]
        lines.append(f"{json.dumps(node.name)} [label={json.dumps(label)}];")
        for dependency in node.all_input_nodes:
            lines.append(f"{json.dumps(dependency.name)} -> {json.dumps(node.name)};")
    lines.append("}")
    destination.with_suffix(".dot").write_text("\n".join(lines) + "\n", encoding="utf-8")
