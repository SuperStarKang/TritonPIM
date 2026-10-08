#ifndef PIM_TO_AIMX_CONVERSION_PASSES
#define PIM_TO_AIMX_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertPIMToAiMX : Pass<"convert-pim-to-aimx", "mlir::ModuleOp"> {
  let summary = "Lower PIM SSA computations and hierarchy directly to AiMX descriptors";
  let description = [{
    Preserve Common execution-tile scheduling, device coordinates, group loops
    and tensor SSA dependencies. Lower each supported computation to descriptor
    sequences in the same pass that places its buffers. There is no intermediate
    compute template, kernel-kind classification, or second lowering pass.

    Pure zero-initialized matmul/matvec reductions keep each output microtile in vACC
    across all K groups of the same execution tile. All Arel panels are prepared
    before MAC begins; C drains once per output microtile. Other loop-carried
    state and intermediate results use physical buffers. Matmul/matvec and
    identity-map parallel add/sub/mul graphs compose through those buffers.
    Common operations without a target implementation fail;
    neither the translator nor the runtime executes Common computations.

    Rectangular valid masks on single-resident-group axes lower to staging
    and drain extents, preserving positive-zero input padding. Varying masks
    across multiple groups and masked read/write aliases need target transfers
    not implemented here. Only integer/index mask expressions are moved to
    staging; tensor computation still lowers to descriptor programs.

    Fixed per-device allocations persist across calls when requested. A complete
    chain is reused when its addresses and control operands are invariant,
    including full-tile equality conditions on dynamic scheduling bounds.
    Common cross-device K reduction is not implemented. AiMX uses BF16 physical
    storage and native arithmetic; this pass does not promise IEEE-equivalent
    source floating-point results.

    Matvec defaults to 4096-element B/C storage rows and one 4096/batch-column
    MAC/drain microtile per output row batch. Storage pitch and compute extent
    are independent; diagnostic options can select the previous layouts.
  }];
  let constructor = "mlir::triton::createConvertPIMToAiMXPass()";
  let options = [
    Option<"batch", "batch", "int64_t", "8",
           "AiMX matmul batch. 1, 2, 4, 8 중 하나.">,
    Option<"matvecRowAlignment", "matvec-row-alignment", "int64_t", "4096",
           "Matvec B/C row alignment in BF16 elements; defaults to 4096. Diagnostic 0 uses 4096/batch.">,
    Option<"matvecFirstNOnly", "matvec-first-n-only", "bool", "true",
           "Matvec vACC reduction computes the first 4096/batch columns; false also computes padded N microtiles for diagnostics.">,
    Option<"matmulRowAlignment", "matmul-row-alignment", "int64_t", "0",
           "GEMM B/C row alignment in BF16 elements; 0 preserves 4096/batch alignment. Used for layout comparisons.">,
    Option<"matmulUsefulNOnly", "matmul-useful-n-only", "bool", "false",
           "GEMM computes useful N rounded to 4096/batch independently of storage pitch; false traverses the padded N extent.">,
    Option<"persistent", "persistent", "bool", "true",
           "Retain fixed per-FPGA buffers and descriptor chains across rounds and calls.">,
  ];
  let dependentDialects = ["mlir::aimx::AiMXDialect", "mlir::pim::PIMDialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::func::FuncDialect", "mlir::memref::MemRefDialect",
                           "mlir::linalg::LinalgDialect", "mlir::tensor::TensorDialect"];
}

#endif
