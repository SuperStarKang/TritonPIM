"""Prefill and decode of TritonPIM on UPMEM next to PIM-LLM's run_LLM.py, per model and prompt length.

Reads the reports of run_upmem_compare.sh (<ours>/<model>_s<seq>_prefill/report.json from
run_upmem.py and <ours>/<model>_s<seq>_decode/report.json from
e2e/benchmark_model_llama_generate.py) and the CSVs of PIM-LLM's export_run_llm_breakdown.py
(<pimllm>/<name>_<seq>_<opt>_2048_t8/{forward_breakdown,gemm}.csv). A missing result is
printed as "-".

Prefill, both without tokenization: PIM-LLM is the mean measured forward; TritonPIM is the
mean warm request from the embedding to the first token (model_to_token_ms). PIM-LLM also
runs lm_head on every prompt row on the DPUs, TritonPIM on the last row on the CPU, so the
layers columns compare the decoder layers alone: PIM-LLM forward - lm_head, TritonPIM the
sum of its layer times. The breakdown uses PIM-LLM's items: h2d (TritonPIM adds the weight
upload, 0 for a warm request), pim = kernel, d2h, and cpu = the rest.
Decode has no PIM-LLM counterpart: the mean step after the first, split like
benchmark_model_llama_generate.py.

  python summarize_compare.py [--ours e2e/compare] [--pimllm <PIMLLM_artifact>/results_run_llm]
"""

import argparse
import csv
import json
from pathlib import Path
import statistics


SEQUENCES = (128, 256, 512, 1024)
# model -> PIM-LLM 결과 이름과 최선 설정 (scripts/run_llm_seq.sh, run_llm3_seq.sh)
PIMLLM_RUNS = {"llama2": ("llama-2-7b", "ts_adj_qkv_coop"), "llama3": ("llama-3-8b", "ts_adj_coop")}
DECODE_ITEMS = ("cpu_ms", "weight_upload_ms", "program_load_ms", "kv_push_ms", "h2d_ms", "kernel_ms", "d2h_ms",
                "reduction_ms", "host_layout_ms", "invoke_outside_native_ms", "other_ms")


def pimllm_prefill(root, model, seq):
    name, opt = PIMLLM_RUNS[model]
    directory = root / f"{name}_{seq}_{opt}_2048_t8"
    if not directory.is_dir():
        return None
    with open(directory / "forward_breakdown.csv", newline="") as file:
        mean, = [row for row in csv.DictReader(file) if row["iter"] == "mean"]
    with open(directory / "gemm.csv", newline="") as file:
        lm_head, = [row for row in csv.DictReader(file) if row["gemm"] == "lm_head"]
    forward = float(mean["forward_ms"])
    h2d, pim, d2h = (float(mean[name]) for name in ("h2d_ms", "pim_ms", "d2h_ms"))
    return {"first_token_ms": forward, "layers_ms": forward - float(lm_head["sum_per_forward_ms"]),
            "h2d_ms": h2d, "pim_ms": pim, "d2h_ms": d2h, "cpu_ms": forward - h2d - pim - d2h}


def ours_prefill(root, model, seq):
    path = root / f"{model}_s{seq}_prefill" / "report.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    warm = report["requests"][1:]

    def mean(value):
        return statistics.mean(value(request) for request in warm)

    first_token = mean(lambda request: request["model_to_token_ms"])
    h2d = mean(lambda request: request["breakdown"]["h2d_ms"] + request["breakdown"]["weight_upload_ms"])
    pim = mean(lambda request: request["breakdown"]["kernel_ms"])
    d2h = mean(lambda request: request["breakdown"]["d2h_ms"])
    return {"first_token_ms": first_token, "layers_ms": mean(lambda request: request["layers_ms"]),
            "h2d_ms": h2d, "pim_ms": pim, "d2h_ms": d2h, "cpu_ms": first_token - h2d - pim - d2h,
            "cold_first_token_ms": report["requests"][0]["model_to_token_ms"]}


def ours_decode(root, model, seq):
    path = root / f"{model}_s{seq}_decode" / "report.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def cell(value, digits=1):
    return "-" if value is None else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    here = Path(__file__).resolve().parent
    parser.add_argument("--ours", type=Path, default=here / "e2e" / "compare")
    parser.add_argument("--pimllm", type=Path,
                        default=Path("/home/dlrkdals/PGEMMlib/PIMLLM_artifact/results_run_llm"))
    args = parser.parse_args()

    print("## Prefill (ms, warm, tokenization 제외)\n")
    print("| model | seq | who | first token | layers | h2d | pim | d2h | cpu | speedup (first token) "
          "| speedup (layers) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for model in PIMLLM_RUNS:
        for seq in SEQUENCES:
            pimllm = pimllm_prefill(args.pimllm, model, seq)
            ours = ours_prefill(args.ours, model, seq)
            for who, result in (("PIM-LLM", pimllm), ("TritonPIM", ours)):
                speedups = ("", "")
                if who == "TritonPIM" and pimllm is not None and ours is not None:
                    speedups = (f"{pimllm['first_token_ms'] / ours['first_token_ms']:.2f}x",
                                f"{pimllm['layers_ms'] / ours['layers_ms']:.2f}x")
                values = [None if result is None else result[name]
                          for name in ("first_token_ms", "layers_ms", "h2d_ms", "pim_ms", "d2h_ms", "cpu_ms")]
                print(f"| {model} | {seq} | {who} | " + " | ".join(cell(value) for value in values)
                      + f" | {speedups[0]} | {speedups[1]} |")

    print("\n## TritonPIM prefill cold (weight 첫 업로드 포함, ms)\n")
    print("| model | " + " | ".join(str(seq) for seq in SEQUENCES) + " |")
    print("|---|" + "---|" * len(SEQUENCES))
    for model in PIMLLM_RUNS:
        results = [ours_prefill(args.ours, model, seq) for seq in SEQUENCES]
        print(f"| {model} | " + " | ".join(cell(None if result is None else result["cold_first_token_ms"])
                                         for result in results) + " |")

    print("\n## TritonPIM decode (ms/step, 첫 step 제외 평균; PIM-LLM은 decode 없음)\n")
    print("| model | seq | step | first step | first token (cold) | kv push/flush | "
          + " | ".join(name[:-len("_ms")] for name in DECODE_ITEMS) + " |")
    print("|---|---|---|---|---|---|" + "---|" * len(DECODE_ITEMS))
    for model in PIMLLM_RUNS:
        for seq in SEQUENCES:
            report = ours_decode(args.ours, model, seq)
            if report is None:
                print(f"| {model} | {seq} | " + " | ".join("-" for _ in range(4 + len(DECODE_ITEMS))) + " |")
                continue
            breakdown = report["decode_step_breakdown"]
            print(f"| {model} | {seq} | {breakdown['step_ms']:.1f} | {report['first_step_ms']:.1f} | "
                  f"{report['first_token_ms']:.1f} | {breakdown['kv_push_ms_per_flush']:.1f} | "
                  + " | ".join(f"{breakdown[name]:.1f}" for name in DECODE_ITEMS) + " |")


if __name__ == "__main__":
    main()
