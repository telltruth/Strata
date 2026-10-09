// Link test for the portable CPU path (no GPU, no models required).
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

int main() {
    using namespace strata::kernels::cpu;
    if (cpu_avx2_ok() || cpu_avx512_ok()) return 1;  // ARM64 must not run x86 intrinsics.
    alignas(64) float input[QKA] = {};
    ActQ a{};
    act_quant_q8_1(input, QKA, a);
    if (a.nchunks != 1 || a.q[0] != 0) return 2;
    if (cpu_name().empty()) return 3;
    return 0;
}
