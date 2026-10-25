"""Prefill and decode of TritonPIM on UPMEM next to PIM-LLM's run_LLM.py, per model and prompt length.

Reads the CSVs of run_upmem_compare.sh (<ours>/<model>_s<seq>_prefill/forward_breakdown.csv
from run_upmem.py, <ours>/<model>_s<seq>_decode/step_breakdown.csv and report.json from
e2e/benchmark_model_llama_generate.py) and those of PIM-LLM's export_run_llm_breakdown.py
(<pimllm>/<name>_<seq>_<opt>_2048_t8/{forward_breakdown,gemm}.csv). A missing result is "-".

Prefill, both without tokenization: PIM-LLM is the mean measured forward; TritonPIM is the
warm_mean row (embedding to the first token). PIM-LLM also runs lm_head on every prompt row
on the DPUs, TritonPIM on the last row on the CPU, so the layers columns compare the decoder
layers alone: PIM-LLM forward - lm_head, TritonPIM the sum of its layer times. Both use
PIM-LLM's items: h2d (TritonPIM adds the weight upload, 0 for a warm request), pim = kernel,
d2h, cpu = the rest. Decode has no PIM-LLM counterpart: TritonPIM's mean step after the
first, in the items of step_breakdown.csv.

Prints markdown tables and writes <ours>/compare_prefill.csv and <ours>/decode.csv.

  python summarize_compare.py [--ours e2e/compare] [--pimllm <PIMLLM_artifact>/results_run_llm]
"""

import argparse
import csv
import json
from pathlib import Path


SEQUENCES = (128, 256, 512, 1024)
# model -> PIM-LLM 결과 이름과 최선 설정 (scripts/run_llm_seq.sh, run_llm3_seq.sh)
PIMLLM_RUNS = {"llama2": ("llama-2-7b", "ts_adj_qkv_coop"), "llama3": ("llama-3-8b", "ts_adj_coop")}
PREFILL_COLUMNS = ("first_token_ms", "layers_ms", "h2d_ms", "pim_ms", "d2h_ms", "cpu_ms")
DECODE_ITEMS = ("cpu_ms", "weight_upload_ms", "program_load_ms", "kv_push_ms", "h2d_ms", "kernel_ms", "d2h_ms",
                "reduction_ms", "host_layout_ms", "invoke_outside_native_ms", "other_ms")
PIMLLM_ITEMS = ("h2d_ms", "pim_ms", "d2h_ms", "cpu_ms")
DECODE_COLUMNS = ("step_ms", "first_step_ms", "first_token_cold_ms", "kv_push_per_flush_ms",
                  *("pimllm_" + column for column in PIMLLM_ITEMS), *DECODE_ITEMS)


def read_rows(path, key, value):
    with open(path, newline="") as file:
        return [row for row in csv.DictReader(file) if row[key] == value]


def pimllm_prefill(root, model, seq):
    name, opt = PIMLLM_RUNS[model]
    directory = root / f"{name}_{seq}_{opt}_2048_t8"
    if not directory.is_dir():
        return None
    mean, = read_rows(directory / "forward_breakdown.csv", "iter", "mean")
    lm_head, = read_rows(directory / "gemm.csv", "gemm", "lm_head")
    forward = float(mean["forward_ms"])
    h2d, pim, d2h = (float(mean[column]) for column in ("h2d_ms", "pim_ms", "d2h_ms"))
    return {"first_token_ms": forward, "layers_ms": forward - float(lm_head["sum_per_forward_ms"]),
            "h2d_ms": h2d, "pim_ms": pim, "d2h_ms": d2h, "cpu_ms": forward - h2d - pim - d2h}


def ours_prefill(root, model, seq):
    """(warm_mean row, cold first token) of forward_breakdown.csv, or None."""
    path = root / f"{model}_s{seq}_prefill" / "forward_breakdown.csv"
    if not path.is_file():
        return None
    warm, = read_rows(path, "iter", "warm_mean")
    cold, = read_rows(path, "iter", "0")
    return ({column: float(warm[column]) for column in PREFILL_COLUMNS}, float(cold["first_token_ms"]))


def ours_decode(root, model, seq):
    directory = root / f"{model}_s{seq}_decode"
    if not (directory / "step_breakdown.csv").is_file():
        return None
    mean, = read_rows(directory / "step_breakdown.csv", "step", "mean_after_first")
    first, = read_rows(directory / "step_breakdown.csv", "step", "0")
    report = json.loads((directory / "report.json").read_text())
    return {"step_ms": float(mean["step_ms"]), "first_step_ms": float(first["step_ms"]),
            "first_token_cold_ms": report["first_token_ms"],
            "kv_push_per_flush_ms": report["decode_step_breakdown"]["kv_push_ms_per_flush"],
            **{"pimllm_" + column: float(mean[column]) for column in PIMLLM_ITEMS},
            **{name: float(mean["detail_" + name]) for name in DECODE_ITEMS}}


def table(rows):
    """Markdown table of rows (dicts with the same keys); floats with one decimal, None as "-"."""
    names = list(rows[0])
    lines = ["| " + " | ".join(names) + " |", "|" + "---|" * len(names)]
    for row in rows:
        lines.append("| " + " | ".join("-" if value is None else f"{value:.1f}" if isinstance(value, float)
                                       else str(value) for value in row.values()) + " |")
    return "\n".join(lines)


def write_csv(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({name: "" if value is None else f"{value:.3f}" if isinstance(value, float) else value
                             for name, value in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--ours", type=Path, default=here / "e2e" / "compare")
    parser.add_argument("--pimllm", type=Path,
                        default=Path("/home/dlrkdals/PGEMMlib/PIMLLM_artifact/results_run_llm"))
    args = parser.parse_args()

    prefill = []
    for model in PIMLLM_RUNS:
        for seq in SEQUENCES:
            pimllm = pimllm_prefill(args.pimllm, model, seq)
            ours = ours_prefill(args.ours, model, seq)
            for who, result in (("PIM-LLM", pimllm), ("TritonPIM", None if ours is None else ours[0])):
                row = {"model": model, "seq": seq, "who": who,
                       **{column: None if result is None else result[column] for column in PREFILL_COLUMNS},
                       "cold_first_token_ms": None, "speedup_first_token": None, "speedup_layers": None}
                if who == "TritonPIM" and ours is not None:
                    row["cold_first_token_ms"] = ours[1]
                    if pimllm is not None:
                        row["speedup_first_token"] = pimllm["first_token_ms"] / ours[0]["first_token_ms"]
                        row["speedup_layers"] = pimllm["layers_ms"] / ours[0]["layers_ms"]
                prefill.append(row)

    decode = []
    for model in PIMLLM_RUNS:
        for seq in SEQUENCES:
            result = ours_decode(args.ours, model, seq)
            decode.append({"model": model, "seq": seq, **(
                result if result is not None else dict.fromkeys(DECODE_COLUMNS, None))})

    print("## Prefill (ms, warm, tokenization 제외; speedup = PIM-LLM / TritonPIM)\n")
    print(table(prefill))
    print("\n## TritonPIM decode (ms/step, 첫 step 제외 평균; PIM-LLM은 decode 없음)\n")
    print(table(decode))
    args.ours.mkdir(parents=True, exist_ok=True)
    write_csv(args.ours / "compare_prefill.csv", prefill)
    write_csv(args.ours / "decode.csv", decode)
    print(f"\nSaved {args.ours / 'compare_prefill.csv'} and {args.ours / 'decode.csv'}")



if __name__ == "__main__":
    main()
