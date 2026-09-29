// Translate actual single-pass output, including nested construction loops.
// RUN: sed 's/bf16/f32/g' %S/../../Conversion/PIMToAiMX/convert.mlir | triton-shared-opt --mlir-print-op-generic --pim-form-host-plan --convert-pim-to-aimx | triton-shared-translate --aimx-to-descriptors | FileCheck %s --check-prefix=DESC
// RUN: sed 's/bf16/f32/g' %S/../../Conversion/PIMToAiMX/convert.mlir | triton-shared-opt --mlir-print-op-generic --pim-form-host-plan --convert-pim-to-aimx=persistent=false | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=STUB
// RUN: triton-shared-opt --mlir-print-op-generic --pim-form-host-plan --convert-pim-to-aimx --cse --canonicalize %S/../../Conversion/PIMToAiMX/convert.mlir | triton-shared-translate --aimx-to-runtime-stub | FileCheck %s --check-prefix=PERSIST
// DESC: #include "aimx_runtime.h"
// DESC: static uint32_t aimx_prog_matmul_device0_matmul_
// DESC: aimx_require(idx + NUM_CHANNELS <= cap
// DESC-NEXT: PIM_RD_INSTR_MC({{.*}}512u, 0x59000u
// DESC: PIM_RD_INSTR_MC({{.*}}65536u, 0x1a400u
// DESC: PIM_RD_INSTR_MC({{.*}}8192u, 0x19012u
// DESC: PIM_RD_INSTR_MC({{.*}}1024u, 0x59014u
// DESC: PIM_RD_INSTR_MC({{.*}}1024u, 0x58018u
// STUB: #include "aimx_descriptors.h"
// STUB: static const float aimx_data_matmul_device0_constant_
// STUB: extern "C" void matmul(float* a0, float* a1, float* a2) {
// STUB: aimx_runtime_init(AIMX_BASE_FPGA, 2);
// STUB: #pragma omp parallel for collapse(3) num_threads(2)
// STUB: pim_private_malloc(196608, _fpga, 7)
// STUB: aimx_prog_matmul_device0_matmul_
// STUB: descriptor upload failed
// STUB: aimx_to_bf16(
// STUB: pim_exec(&_args, _fpga)
// STUB: + aimx_from_bf16(
// STUB: = aimx_from_bf16(
// PERSIST: std::lock_guard<std::mutex> _host_lock(_host_mutex)
// PERSIST: aimx_runtime_init(AIMX_BASE_FPGA, 2)
// PERSIST: aimx_resident_acquire(AIMX_KERNEL_KEY ":site0", _fpga,
// PERSIST: aimx_resident_alloc(_resident, 0, 196608, 262144, 2)
// PERSIST: if (!_resident->ready) {
// PERSIST: .idx = 0;
// PERSIST: aimx_prog_matmul_device0_matmul_
// PERSIST-NOT: if (!_resident->ready)
// PERSIST: descriptor upload failed
// PERSIST-NEXT: _resident->ready = true;
// PERSIST: input upload failed
// PERSIST: pim_exec
// PERSIST: output download failed
// PERSIST: aimx_resident_release(_resident)
// PERSIST-NOT: pim_free(
