"""One CSV row per layer/ or e2e/ report: per-layer time and its splits, sample medians.

Columns per layer: weight_gemm/activation_gemm/cpu (the graph of claim 5), and for the
upload and run phases the transfer/layout/compute/setup/other slot groups. Upload reports
also carry the weight-stationary prediction for R requests.

  python summarize_reports.py layer/report_*/report.json e2e/report_*/report.json --output summary.csv
"""

import argparse
import csv
import json
from pathlib import Path
import statistics

from experiment_common import SLOT_GROUPS


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        samples = [call for call in report["calls"] if call["phase"] == "sample"]
        layers = report["layers"]
        row = {"report": path.parent.name, "config_source": report["config_source"],
               "slots": report["memory"]["slots"], "groups": report["memory"]["groups"],
               "layers": layers, "kv": report["kv"],
               "qk_heads": report["qk_heads"], "pv_heads": report["pv_heads"],
               "per_layer_ms": statistics.median(call["layers_ms"] for call in samples) / layers}
        for name in samples[0]["kinds"]:
            row[name] = statistics.median(call["kinds"][name] for call in samples) / layers
        for phase in ("upload", "run"):
            for group in (*SLOT_GROUPS, "other"):
                row[f"{phase}_{group}_ms"] = statistics.median(
                    call["phases"][phase][group] for call in samples) / layers
        if "prediction" in report["summary"]:
            for requests, value in report["summary"]["prediction"]["layers_ms"].items():
                row[f"predicted_R{requests}_ms"] = value
        rows.append(row)

    columns = list(dict.fromkeys(column for row in rows for column in row))
    with args.output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['report']}: per_layer_ms={row['per_layer_ms']:.1f} weight_gemm={row['weight_gemm_ms']:.1f} "
              f"activation_gemm={row['activation_gemm_ms']:.1f} cpu={row['cpu_ms']:.1f}", flush=True)
    print(f"csv={args.output}", flush=True)


if __name__ == "__main__":
    main()
