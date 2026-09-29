"""Check FP32 mm and INT8 _int_mm through generated PIM wrappers."""

import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
from torch._inductor import config as inductor_config
from torch._inductor.codecache import PyCodeCache

from autotuner import PIMConfig
from torch_backend import PIMInductorBackend, PIMTorchConfig, target_spec
from attention.profile_attention import trace_kernels


def fp32_mm(a, b):
    return torch.mm(a, b)


def int8_mm(a, b):
    return torch._int_mm(a, b)


def verify_case(name, function, a, b, reference, config, rtol, atol):
    with torch.no_grad(), inductor_config.patch({"cpp_wrapper": False}):
        previous_modules = {id(module) for module in PyCodeCache.modules}
        compiled = torch.compile(
            function,
            backend=PIMInductorBackend(config),
            fullgraph=True,
            dynamic=False,
        )
        compiled(a, b)

        modules = [
            module
            for module in PyCodeCache.modules
            if id(module) not in previous_modules
        ]
        with trace_kernels(modules) as records:
            output = compiled(a, b)

    torch.testing.assert_close(output, reference, rtol=rtol, atol=atol)

    pim_calls = [
        record
        for record in records
        if record["pim_stages_ms"] is not None
    ]
    assert len(pim_calls) == 1

    options = pim_calls[0]["pim_config"]["backend_options"]
    assert options["pim_target"] == "upmem"
    assert options["num_units"] == 2048
    assert options["nr_tasklets"] == 16
    assert options["p_m"] * options["p_n"] * options["p_k"] == 2048
    assert [
        tensor["dtype"]
        for tensor in pim_calls[0]["tensor_arguments"]
    ] == [
        str(a.dtype).removeprefix("torch."),
        str(b.dtype).removeprefix("torch."),
        str(reference.dtype).removeprefix("torch."),
    ]

    print(
        f"{name}: correctness=PASS "
        f"PIM_wrapper_calls={len(pim_calls)} "
        f"output_dtype={output.dtype}"
    )


def main():
    logging.basicConfig(level=logging.INFO)
    torch.manual_seed(0)

    config = PIMTorchConfig(
        spec=target_spec("upmem", num_units=2048, nr_tasklets=16),
        config=PIMConfig(
            block_m=16,
            block_n=16,
            block_k=32,
            group_m=1,
            group_n=1,
            group_k=1,
            tile_m=1,
            tile_n=1,
            tile_k=2,
            p_m=32,
            p_n=64,
            p_k=1,
        ),
    )

    a = torch.randn(512, 64, dtype=torch.float32)
    b = torch.randn(64, 1024, dtype=torch.float32)
    verify_case(
        "FP32", fp32_mm, a, b, torch.mm(a, b), config,
        rtol=1e-4, atol=1e-4,
    )

    a_int = torch.randint(-4, 5, (512, 64), dtype=torch.int8)
    b_int = torch.randint(-4, 5, (64, 1024), dtype=torch.int8)
    verify_case(
        "INT8", int8_mm, a_int, b_int,
        a_int.to(torch.int32) @ b_int.to(torch.int32), config,
        rtol=0, atol=0,
    )


if __name__ == "__main__":
    main()
