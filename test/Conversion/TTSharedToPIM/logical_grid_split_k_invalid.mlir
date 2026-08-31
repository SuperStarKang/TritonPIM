// source program_id가 m/n 외 별도 K plane을 만들면 scheduling 전에 spatial K가
// 확정된다. Common IR은 이를 logical axis로 만들지 않고 semantic reduction 사용을
// 요구한다.
// RUN: triton-shared-opt --pim-build-logical-grid -verify-diagnostics %s

// expected-error @below {{source-level split-K is not a Common IR logical axis}}
func.func @source_split_k(%npx: i32, %npy: i32, %npz: i32,
                          %pidx: i32, %pidy: i32, %pidz: i32) {
  %mn = arith.addi %pidx, %pidy : i32
  %mnk = arith.addi %mn, %pidz : i32
  return
}
