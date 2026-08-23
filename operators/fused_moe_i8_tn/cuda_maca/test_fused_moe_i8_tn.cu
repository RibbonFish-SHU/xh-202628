#define XH_FUSED_MOE_NO_ENTRYPOINT
#include "submission.cu"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#if defined(__XCORE_WN__)
constexpr const char* kBenchmarkDevice = "c500-local";
#else
constexpr const char* kBenchmarkDevice = "nvidia-proxy";
#endif

#define CUDA_CHECK(expression) check_cuda((expression), #expression, __FILE__, __LINE__)

void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(file) + ":" + std::to_string(line) + ": " + expression
            + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count), pointer_(nullptr) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&pointer_), count * sizeof(T)));
    }

    ~DeviceBuffer() {
        if (pointer_ != nullptr) {
            cudaFree(pointer_);
        }
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return pointer_; }
    size_t count() const { return count_; }

    void copy_from(const std::vector<T>& source) {
        if (source.size() != count_) {
            throw std::runtime_error("host-to-device size mismatch");
        }
        CUDA_CHECK(cudaMemcpy(
            pointer_, source.data(), count_ * sizeof(T), cudaMemcpyHostToDevice));
    }

    std::vector<T> copy_to_host() const {
        std::vector<T> result(count_);
        CUDA_CHECK(cudaMemcpy(
            result.data(), pointer_, count_ * sizeof(T), cudaMemcpyDeviceToHost));
        return result;
    }

private:
    size_t count_;
    T* pointer_;
};

uint32_t xorshift32(uint32_t* state) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

uint16_t float_to_bf16(float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

float bf16_to_float(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

struct SmallCase {
    xh_fused_moe::KernelConfig config;
    int experts;
    uint32_t seed;
    int value_mode;
    std::string name;
};

void fill_small_inputs(
    const SmallCase& test_case,
    std::vector<int8_t>* a,
    std::vector<int8_t>* b,
    std::vector<float>* scale_a,
    std::vector<float>* scale_b,
    std::vector<float>* moe_weights,
    std::vector<int32_t>* expert_ids
) {
    uint32_t state = test_case.seed;
    auto next_int8 = [&]() -> int8_t {
        if (test_case.value_mode == 1) {
            return 0;
        }
        if (test_case.value_mode == 2) {
            return (xorshift32(&state) & 1U) == 0 ? static_cast<int8_t>(-20) : static_cast<int8_t>(20);
        }
        return static_cast<int8_t>(static_cast<int>(xorshift32(&state) % 41U) - 20);
    };
    auto next_scale = [&]() -> float {
        return (static_cast<int>(xorshift32(&state) % 2001U) - 1000) * 0.00025f;
    };

    std::generate(a->begin(), a->end(), next_int8);
    std::generate(b->begin(), b->end(), next_int8);
    std::generate(scale_a->begin(), scale_a->end(), next_scale);
    std::generate(scale_b->begin(), scale_b->end(), next_scale);
    std::generate(moe_weights->begin(), moe_weights->end(), next_scale);
    for (size_t tile = 0; tile < expert_ids->size(); ++tile) {
        (*expert_ids)[tile] = static_cast<int32_t>((tile * 5U + test_case.seed) % test_case.experts);
    }
}

bool vectors_equal_bits(const std::vector<float>& lhs, const std::vector<float>& rhs) {
    return lhs.size() == rhs.size()
        && std::memcmp(lhs.data(), rhs.data(), lhs.size() * sizeof(float)) == 0;
}

void run_small_case(const SmallCase& test_case) {
    const auto& config = test_case.config;
    const size_t a_count = static_cast<size_t>(config.em) * config.k;
    const size_t b_count = static_cast<size_t>(test_case.experts) * config.n * config.k;
    const size_t out_count = static_cast<size_t>(config.em) * config.n;
    std::vector<int8_t> host_a(a_count);
    std::vector<int8_t> host_b(b_count);
    std::vector<float> host_scale_a(config.em);
    std::vector<float> host_scale_b(static_cast<size_t>(test_case.experts) * config.n);
    std::vector<float> host_moe_weights(config.em);
    std::vector<int32_t> host_expert_ids(config.em / 128);
    fill_small_inputs(
        test_case,
        &host_a,
        &host_b,
        &host_scale_a,
        &host_scale_b,
        &host_moe_weights,
        &host_expert_ids
    );

    DeviceBuffer<int8_t> dev_a(a_count);
    DeviceBuffer<int8_t> dev_b(b_count);
    DeviceBuffer<float> dev_scale_a(config.em);
    DeviceBuffer<float> dev_scale_b(host_scale_b.size());
    DeviceBuffer<float> dev_moe_weights(config.em);
    DeviceBuffer<int32_t> dev_expert_ids(host_expert_ids.size());
    DeviceBuffer<uint16_t> dev_out(out_count);
    dev_a.copy_from(host_a);
    dev_b.copy_from(host_b);
    dev_scale_a.copy_from(host_scale_a);
    dev_scale_b.copy_from(host_scale_b);
    dev_moe_weights.copy_from(host_moe_weights);
    dev_expert_ids.copy_from(host_expert_ids);
    CUDA_CHECK(cudaMemset(dev_out.get(), 0x7f, out_count * sizeof(uint16_t)));

    xh_fused_moe::launch(
        dev_a.get(),
        dev_b.get(),
        dev_scale_a.get(),
        dev_scale_b.get(),
        dev_moe_weights.get(),
        dev_expert_ids.get(),
        reinterpret_cast<__nv_bfloat16*>(dev_out.get()),
        config
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<uint16_t> actual = dev_out.copy_to_host();

    size_t matched = 0;
    float max_abs = 0.0f;
    for (int row = 0; row < config.em; ++row) {
        const int expert = host_expert_ids[row / 128];
        for (int col = 0; col < config.n; ++col) {
            int32_t accumulator = 0;
            for (int k = 0; k < config.k; ++k) {
                accumulator += static_cast<int32_t>(host_a[static_cast<size_t>(row) * config.k + k])
                    * static_cast<int32_t>(host_b[
                        (static_cast<size_t>(expert) * config.n + col) * config.k + k]);
            }
            const float reference_value = static_cast<float>(accumulator)
                * (host_scale_a[row] * host_moe_weights[row])
                * host_scale_b[static_cast<size_t>(expert) * config.n + col];
            const float reference = bf16_to_float(float_to_bf16(reference_value));
            const float target = bf16_to_float(actual[static_cast<size_t>(row) * config.n + col]);
            const float difference = std::fabs(target - reference);
            max_abs = std::max(max_abs, difference);
            if (difference <= 5.0e-3f + 2.0e-2f * std::fabs(reference)) {
                ++matched;
            }
        }
    }

    const double matched_ratio = static_cast<double>(matched) / out_count;
    if (matched_ratio < 0.99) {
        throw std::runtime_error(test_case.name + " failed the official matched-ratio gate");
    }
    if (dev_a.copy_to_host() != host_a
        || dev_b.copy_to_host() != host_b
        || !vectors_equal_bits(dev_scale_a.copy_to_host(), host_scale_a)
        || !vectors_equal_bits(dev_scale_b.copy_to_host(), host_scale_b)
        || !vectors_equal_bits(dev_moe_weights.copy_to_host(), host_moe_weights)
        || dev_expert_ids.copy_to_host() != host_expert_ids) {
        throw std::runtime_error(test_case.name + " modified a read-only input");
    }

    std::cout << "CORRECTNESS case=" << test_case.name
              << " matched_ratio=" << std::fixed << std::setprecision(6) << matched_ratio
              << " max_abs=" << max_abs << " PASS\n";
}

__global__ void fill_float_kernel(float* values, size_t count, float value) {
    const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) {
        values[index] = value;
    }
}

void fill_float(float* values, size_t count, float value) {
    const int threads = 256;
    const int blocks = static_cast<int>((count + threads - 1) / threads);
    fill_float_kernel<<<blocks, threads>>>(values, count, value);
    CUDA_CHECK(cudaGetLastError());
}

struct PublicCase {
    xh_fused_moe::KernelConfig config;
    const char* name;
    bool skewed;
};

const PublicCase kPublicCases[] = {
    {{4096, 4096, 7168}, "decode-gate-up", false},
    {{32768, 4096, 7168}, "prefill-gate-up", true},
    {{4096, 7168, 2048}, "decode-down", false},
    {{32768, 7168, 2048}, "prefill-down", true},
};

void verify_public_inference() {
    DeviceBuffer<int8_t> unknown_a(1);
    DeviceBuffer<uint16_t> unknown_out(1);
    for (const PublicCase& public_case : kPublicCases) {
        const auto& expected = public_case.config;
        DeviceBuffer<int8_t> dev_a(static_cast<size_t>(expected.em) * expected.k);
        DeviceBuffer<uint16_t> dev_out(static_cast<size_t>(expected.em) * expected.n);
        xh_fused_moe::KernelConfig actual{};
        if (!xh_fused_moe::infer_public_config(
                dev_a.get(), reinterpret_cast<__nv_bfloat16*>(dev_out.get()), &actual)
            || !xh_fused_moe::same_config(actual, expected)) {
            throw std::runtime_error(std::string("allocation inference failed for ") + public_case.name);
        }
        if (!xh_fused_moe::infer_public_config(
                dev_a.get(), reinterpret_cast<__nv_bfloat16*>(unknown_out.get()), &actual)
            || !xh_fused_moe::same_config(actual, expected)) {
            throw std::runtime_error(std::string("A-primary inference failed for ") + public_case.name);
        }
        if (!xh_fused_moe::infer_public_config(
                unknown_a.get(), reinterpret_cast<__nv_bfloat16*>(dev_out.get()), &actual)
            || !xh_fused_moe::same_config(actual, expected)) {
            throw std::runtime_error(std::string("out-fallback inference failed for ") + public_case.name);
        }
        std::cout << "REGRESSION allocation-inference case=" << public_case.name << " PASS\n";
    }
}

bool infer_public_config_two_query_reference(
    const int8_t* a,
    const __nv_bfloat16* out,
    xh_fused_moe::KernelConfig* config
) {
    xh_fused_moe::KernelConfig a_config{};
    xh_fused_moe::KernelConfig out_config{};
    size_t bytes = 0;
    const bool have_a = xh_fused_moe::allocation_bytes(a, &bytes)
        && xh_fused_moe::config_from_bytes(bytes, true, &a_config);
    const bool have_out = xh_fused_moe::allocation_bytes(out, &bytes)
        && xh_fused_moe::config_from_bytes(bytes, false, &out_config);
    if (have_a && have_out && !xh_fused_moe::same_config(a_config, out_config)) {
        return false;
    }
    if (have_a) {
        *config = a_config;
        return true;
    }
    if (have_out) {
        *config = out_config;
        return true;
    }
    return false;
}

using InferenceFunction = bool (*)(
    const int8_t*, const __nv_bfloat16*, xh_fused_moe::KernelConfig*);

double measure_inference_ns(
    InferenceFunction inference,
    const int8_t* a,
    const __nv_bfloat16* out,
    int iterations,
    volatile uint64_t* checksum
) {
    const auto started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iterations; ++iteration) {
        xh_fused_moe::KernelConfig config{};
        if (!inference(a, out, &config)) {
            throw std::runtime_error("shape-inference benchmark failed");
        }
        *checksum += static_cast<uint64_t>(config.em + config.n + config.k);
    }
    const auto finished = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::nano>(finished - started).count() / iterations;
}

void benchmark_inference_query_path() {
    const auto config = kPublicCases[0].config;
    DeviceBuffer<int8_t> dev_a(static_cast<size_t>(config.em) * config.k);
    DeviceBuffer<uint16_t> dev_out(static_cast<size_t>(config.em) * config.n);
    constexpr int kRounds = 9;
    constexpr int kIterations = 2000;
    std::vector<double> fast_path_ns;
    std::vector<double> two_query_ns;
    volatile uint64_t checksum = 0;
    for (int round = 0; round < kRounds; ++round) {
        if ((round & 1) == 0) {
            fast_path_ns.push_back(measure_inference_ns(
                xh_fused_moe::infer_public_config,
                dev_a.get(),
                reinterpret_cast<__nv_bfloat16*>(dev_out.get()),
                kIterations,
                &checksum));
            two_query_ns.push_back(measure_inference_ns(
                infer_public_config_two_query_reference,
                dev_a.get(),
                reinterpret_cast<__nv_bfloat16*>(dev_out.get()),
                kIterations,
                &checksum));
        } else {
            two_query_ns.push_back(measure_inference_ns(
                infer_public_config_two_query_reference,
                dev_a.get(),
                reinterpret_cast<__nv_bfloat16*>(dev_out.get()),
                kIterations,
                &checksum));
            fast_path_ns.push_back(measure_inference_ns(
                xh_fused_moe::infer_public_config,
                dev_a.get(),
                reinterpret_cast<__nv_bfloat16*>(dev_out.get()),
                kIterations,
                &checksum));
        }
    }
    std::sort(fast_path_ns.begin(), fast_path_ns.end());
    std::sort(two_query_ns.begin(), two_query_ns.end());
    const double fast_median = fast_path_ns[kRounds / 2];
    const double two_query_median = two_query_ns[kRounds / 2];
    std::cout << "BENCHMARK host-inference fast_path_ns=" << std::fixed
              << std::setprecision(1) << fast_median
              << " two_query_ns=" << two_query_median
              << " speedup=" << std::setprecision(3) << two_query_median / fast_median
              << "x checksum=" << checksum << "\n";
}

__global__ void plain_b64_store_proxy_kernel(uint16_t* output) {
    constexpr int kTile = 128;
    const int thread_id = threadIdx.x;
    for (int row_group = 0; row_group < 2; ++row_group) {
        for (int row_in_group = 0; row_in_group < 4; ++row_in_group) {
            const int row = xh_fused_moe::mma_output_row_local(
                thread_id, row_group, row_in_group);
            for (int col_group = 0; col_group < 2; ++col_group) {
                const int col = xh_fused_moe::mma_output_col_local(
                    thread_id, col_group, 0);
                uint64_t packed = 0;
#pragma unroll
                for (int element = 0; element < 4; ++element) {
                    const uint16_t value = static_cast<uint16_t>(
                        row * kTile + col + element + 1);
                    packed |= static_cast<uint64_t>(value) << (16 * element);
                }
                *reinterpret_cast<uint64_t*>(output + row * kTile + col) = packed;
            }
        }
    }
}

void verify_plain_b64_store_mapping() {
    constexpr int kTile = 128;
    constexpr int kThreads = 256;
    constexpr int kElementsPerStore = 4;
    std::vector<int> visits(kTile * kTile, 0);
    int store_count = 0;
    for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
        for (int row_group = 0; row_group < 2; ++row_group) {
            for (int row_in_group = 0; row_in_group < 4; ++row_in_group) {
                const int row = xh_fused_moe::mma_output_row_local(
                    thread_id, row_group, row_in_group);
                for (int col_group = 0; col_group < 2; ++col_group) {
                    const int col = xh_fused_moe::mma_output_col_local(
                        thread_id, col_group, 0);
                    if (row < 0 || row >= kTile || col < 0 || col + 3 >= kTile) {
                        throw std::runtime_error("plain b64 store is outside the output tile");
                    }
                    if (((static_cast<size_t>(row) * kTile + col) * sizeof(uint16_t)) % 8 != 0) {
                        throw std::runtime_error("plain b64 store is not 8-byte aligned");
                    }
                    for (int element = 0; element < kElementsPerStore; ++element) {
                        ++visits[row * kTile + col + element];
                    }
                    ++store_count;
                }
            }
        }
    }
    if (store_count != 4096
        || !std::all_of(visits.begin(), visits.end(), [](int count) { return count == 1; })) {
        throw std::runtime_error("plain b64 stores do not exactly cover the output tile");
    }

    for (const PublicCase& public_case : kPublicCases) {
        const auto& config = public_case.config;
        if ((config.em % kTile) != 0 || (config.n % kTile) != 0
            || (static_cast<size_t>(config.n) * sizeof(uint16_t)) % 8 != 0) {
            throw std::runtime_error(
                std::string("public output is not exact/aligned for ") + public_case.name);
        }
        const int last_tile_m = config.em / kTile - 1;
        const int last_tile_n = config.n / kTile - 1;
        for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
            for (int row_group = 0; row_group < 2; ++row_group) {
                for (int row_in_group = 0; row_in_group < 4; ++row_in_group) {
                    const int row = last_tile_m * kTile
                        + xh_fused_moe::mma_output_row_local(
                            thread_id, row_group, row_in_group);
                    for (int col_group = 0; col_group < 2; ++col_group) {
                        const int col = last_tile_n * kTile
                            + xh_fused_moe::mma_output_col_local(
                                thread_id, col_group, 0);
                        const size_t element_offset = static_cast<size_t>(row) * config.n + col;
                        if (row < 0 || row >= config.em || col < 0 || col + 3 >= config.n
                            || (element_offset * sizeof(uint16_t)) % 8 != 0) {
                            throw std::runtime_error(
                                std::string("last public output tile is unsafe for ")
                                + public_case.name);
                        }
                    }
                }
            }
        }
        std::cout << "REGRESSION maca-plain-b64-store case=" << public_case.name
                  << " exact-bounds=PASS alignment=8 PASS\n";
    }

    DeviceBuffer<uint16_t> device_output(kTile * kTile);
    if ((reinterpret_cast<uintptr_t>(device_output.get()) & 7U) != 0) {
        throw std::runtime_error("CUDA allocation is not 8-byte aligned");
    }
    CUDA_CHECK(cudaMemset(device_output.get(), 0, kTile * kTile * sizeof(uint16_t)));
    plain_b64_store_proxy_kernel<<<1, kThreads>>>(device_output.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    const std::vector<uint16_t> actual = device_output.copy_to_host();
    for (size_t index = 0; index < actual.size(); ++index) {
        if (actual[index] != static_cast<uint16_t>(index + 1)) {
            throw std::runtime_error("plain b64 proxy changed value ownership or order");
        }
    }
    std::cout << "REGRESSION maca-plain-b64-store stores=" << store_count
              << " bytes=" << store_count * sizeof(uint64_t)
              << " elements=" << visits.size()
              << " exact-cover=PASS proxy-values=PASS\n";
}

void verify_mma_output_mapping() {
    constexpr int tile = 128;
    std::vector<int> visits(tile * tile, 0);
    for (int thread_id = 0; thread_id < 256; ++thread_id) {
        for (int row_group = 0; row_group < 2; ++row_group) {
            for (int row_in_group = 0; row_in_group < 4; ++row_in_group) {
                const int row = xh_fused_moe::mma_output_row_local(
                    thread_id, row_group, row_in_group);
                for (int col_group = 0; col_group < 2; ++col_group) {
                    for (int col_in_group = 0; col_in_group < 4; ++col_in_group) {
                        const int col = xh_fused_moe::mma_output_col_local(
                            thread_id, col_group, col_in_group);
                        if (row < 0 || row >= tile || col < 0 || col >= tile) {
                            throw std::runtime_error("MMA output mapping is out of range");
                        }
                        ++visits[row * tile + col];
                    }
                }
            }
        }
    }
    if (!std::all_of(visits.begin(), visits.end(), [](int count) { return count == 1; })) {
        throw std::runtime_error("MMA output mapping does not cover the tile exactly once");
    }
    std::cout << "REGRESSION maca-mma-output-mapping elements=" << visits.size()
              << " exact-cover=PASS\n";
}

void verify_mma_grid_mapping() {
    constexpr int tile_rows = 128;
    for (const PublicCase& public_case : kPublicCases) {
        const int grid_m = (public_case.config.em + tile_rows - 1) / tile_rows;
        const int grid_x = xh_fused_moe::mma_grid_x(public_case.config);
        const int grid_z = (grid_m + grid_x - 1) / grid_x;
        std::vector<int> visits(grid_m, 0);
        for (int z = 0; z < grid_z; ++z) {
            for (int x = 0; x < grid_x; ++x) {
                const int tile_m = x + z * grid_x;
                if (tile_m < grid_m) {
                    ++visits[tile_m];
                }
            }
        }
        if (grid_x != 1
            || !std::all_of(visits.begin(), visits.end(), [](int count) { return count == 1; })) {
            throw std::runtime_error(
                std::string("MMA grid mapping failed for ") + public_case.name);
        }
        std::cout << "REGRESSION maca-mma-grid case=" << public_case.name
                  << " grid_x=" << grid_x << " grid_z=" << grid_z
                  << " exact-cover=PASS\n";
    }
}

std::vector<int32_t> output_scratch_sort_map(
    const std::vector<int32_t>& experts
) {
    if (experts.empty() || experts.size() > xh_fused_moe::kOutputScratchSortTiles) {
        throw std::runtime_error("output scratch sort tile count is invalid");
    }

    std::vector<int32_t> map(experts.size(), -1);
    for (int logical_tile = 0; logical_tile < static_cast<int>(experts.size()); ++logical_tile) {
        const int rank = xh_fused_moe::output_scratch_stable_sort_rank(
            experts.data(), logical_tile, static_cast<int>(experts.size()));
        if (rank < 0 || rank >= static_cast<int>(experts.size()) || map[rank] != -1) {
            throw std::runtime_error("output scratch sort rank is not a bijection");
        }
        map[rank] = logical_tile;
    }
    return map;
}

void verify_output_scratch_sort_cover(
    const std::vector<int32_t>& experts,
    int grid_x,
    int n_tiles
) {
    if (grid_x < 1 || n_tiles < 1) {
        throw std::runtime_error("output scratch sort grid is invalid");
    }
    const std::vector<int32_t> map = output_scratch_sort_map(experts);
    std::vector<int> visits(experts.size() * n_tiles, 0);
    const int grid_z = (static_cast<int>(experts.size()) + grid_x - 1) / grid_x;
    for (int cluster = 0; cluster < grid_z; ++cluster) {
        for (int x = 0; x < grid_x; ++x) {
            const int physical_tile = x + cluster * grid_x;
            if (physical_tile == 0 || physical_tile >= static_cast<int>(experts.size())) {
                continue;
            }
            const int logical_tile = map[physical_tile];
            for (int tile_n = 0; tile_n < n_tiles; ++tile_n) {
                ++visits[static_cast<size_t>(logical_tile) * n_tiles + tile_n];
            }
        }
    }
    for (int tile_n = 0; tile_n < n_tiles; ++tile_n) {
        ++visits[tile_n];
    }
    if (!std::all_of(visits.begin(), visits.end(), [](int count) { return count == 1; })) {
        throw std::runtime_error("output scratch sort does not cover every M/N tile exactly once");
    }
}

struct OutputScratchClusterStats {
    int cluster_count;
    int homogeneous_clusters;
    int mixed_clusters;
    int max_cluster_experts;
};

OutputScratchClusterStats output_scratch_cluster_stats(
    const std::vector<int32_t>& experts,
    const std::vector<int32_t>& map,
    int grid_x
) {
    if (grid_x < 1 || experts.empty() || map.size() != experts.size()) {
        throw std::runtime_error("output scratch cluster model is invalid");
    }
    OutputScratchClusterStats stats{
        (static_cast<int>(experts.size()) + grid_x - 1) / grid_x,
        0,
        0,
        0,
    };
    for (int cluster = 0; cluster < stats.cluster_count; ++cluster) {
        const int first_rank = cluster == 0 ? 1 : cluster * grid_x;
        const int last_rank = std::min(
            static_cast<int>(experts.size()) - 1,
            (cluster + 1) * grid_x - 1);
        if (first_rank > last_rank) {
            continue;
        }
        int cluster_experts = 1;
        for (int rank = first_rank + 1; rank <= last_rank; ++rank) {
            cluster_experts += experts[map[rank]] != experts[map[rank - 1]];
        }
        stats.homogeneous_clusters += cluster_experts == 1;
        stats.mixed_clusters += cluster_experts > 1;
        stats.max_cluster_experts = std::max(
            stats.max_cluster_experts, cluster_experts);
    }
    return stats;
}

void verify_output_scratch_sort_pattern(const std::vector<int32_t>& experts) {
    for (int expert : experts) {
        if (expert < 0 || expert >= 256) {
            throw std::runtime_error("expert id is outside [0,255]");
        }
    }
    const std::vector<int32_t> map = output_scratch_sort_map(experts);
    if (map[0] != 0) {
        throw std::runtime_error("output scratch sort did not reserve logical tile zero");
    }

    std::vector<int32_t> reference;
    for (int logical_tile = 1; logical_tile < static_cast<int>(experts.size()); ++logical_tile) {
        reference.push_back(logical_tile);
    }
    std::sort(reference.begin(), reference.end(), [&](int lhs, int rhs) {
        return experts[lhs] < experts[rhs]
            || (experts[lhs] == experts[rhs] && lhs < rhs);
    });
    for (int physical_tile = 1; physical_tile < static_cast<int>(experts.size()); ++physical_tile) {
        if (map[physical_tile] != reference[physical_tile - 1]) {
            throw std::runtime_error("output scratch sort differs from stable reference order");
        }
    }

    int sorted_position = 1;
    for (int expert = 0; expert < 256; ++expert) {
        int expert_count = 0;
        for (int logical_tile = 1; logical_tile < static_cast<int>(experts.size()); ++logical_tile) {
            expert_count += experts[logical_tile] == expert;
        }
        for (int offset = 0; offset < expert_count; ++offset) {
            if (experts[map[sorted_position + offset]] != expert) {
                throw std::runtime_error("sorted expert interval or count is incorrect");
            }
        }
        sorted_position += expert_count;
    }
    if (sorted_position != static_cast<int>(experts.size())) {
        throw std::runtime_error("sorted expert counts do not cover every nonzero tile");
    }

    verify_output_scratch_sort_cover(
        experts, xh_fused_moe::kOutputScratchSortGridM, 32);
}

void enumerate_output_scratch_sort_patterns(
    std::vector<int32_t>* experts,
    int logical_tile,
    uint64_t* pattern_count
) {
    if (logical_tile == static_cast<int>(experts->size())) {
        verify_output_scratch_sort_pattern(*experts);
        ++*pattern_count;
        return;
    }
    for (int expert = 0; expert < 3; ++expert) {
        (*experts)[logical_tile] = expert;
        enumerate_output_scratch_sort_patterns(experts, logical_tile + 1, pattern_count);
    }
}

std::vector<int32_t> output_scratch_sort_experts(const std::string& pattern) {
    std::vector<int32_t> experts(xh_fused_moe::kOutputScratchSortTiles);
    for (int tile = 0; tile < static_cast<int>(experts.size()); ++tile) {
        if (pattern == "all-same") {
            experts[tile] = 7;
        } else if (pattern == "all-unique") {
            experts[tile] = (tile * 73) % 256;
        } else {
            experts[tile] = (tile * tile + 3 * tile) % 16;
        }
    }
    return experts;
}

void verify_output_scratch_sort_device_map(
    const std::vector<int32_t>& experts,
    const char* pattern
) {
    DeviceBuffer<int32_t> device_experts(experts.size());
    DeviceBuffer<int32_t> device_map(experts.size());
    device_experts.copy_from(experts);
    CUDA_CHECK(cudaMemset(device_map.get(), 0xff, experts.size() * sizeof(int32_t)));
    xh_fused_moe::build_output_scratch_expert_sort_map_kernel
        <<<1, xh_fused_moe::kOutputScratchSortTiles>>>(
            device_experts.get(),
            reinterpret_cast<__nv_bfloat16*>(device_map.get()));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    if (device_map.copy_to_host() != output_scratch_sort_map(experts)) {
        throw std::runtime_error(std::string("device stable sort map mismatch for ") + pattern);
    }
    if (device_experts.copy_to_host() != experts) {
        throw std::runtime_error(std::string("stable sort mutated expert ids for ") + pattern);
    }
}

void verify_output_scratch_sort_structure() {
    uint64_t exhaustive_patterns = 0;
    for (int tile_count = 1; tile_count <= 8; ++tile_count) {
        std::vector<int32_t> experts(tile_count, 0);
        enumerate_output_scratch_sort_patterns(&experts, 1, &exhaustive_patterns);
    }
    if (exhaustive_patterns != 3280) {
        throw std::runtime_error("output scratch exhaustive pattern count changed");
    }

    const std::vector<std::string> patterns = {"all-same", "all-unique", "skewed"};
    for (const std::string& pattern : patterns) {
        const std::vector<int32_t> experts = output_scratch_sort_experts(pattern);
        verify_output_scratch_sort_pattern(experts);
        verify_output_scratch_sort_cover(
            experts, xh_fused_moe::kPrefillDownOutputScratchSortGridM, 56);
        verify_output_scratch_sort_device_map(experts, pattern.c_str());
    }

    const std::vector<int32_t> all_same = output_scratch_sort_experts("all-same");
    const std::vector<int32_t> all_same_map = output_scratch_sort_map(all_same);
    for (int tile = 0; tile < static_cast<int>(all_same_map.size()); ++tile) {
        if (all_same_map[tile] != tile) {
            throw std::runtime_error("all-same stable order changed");
        }
    }
    const std::vector<int32_t> skewed = output_scratch_sort_experts("skewed");
    const std::vector<int32_t> skewed_map = output_scratch_sort_map(skewed);
    for (int expert = 0; expert < 16; ++expert) {
        int count = 0;
        for (int tile = 1; tile < static_cast<int>(skewed.size()); ++tile) {
            count += skewed[tile] == expert;
        }
        const int expected_count = expert == 0 ? 31 : ((expert & 1) == 0 ? 32 : 0);
        if (count != expected_count) {
            throw std::runtime_error("skewed expert count changed");
        }
    }
    const int case2_grid_x = xh_fused_moe::kOutputScratchSortGridM;
    const int case4_grid_x = xh_fused_moe::kPrefillDownOutputScratchSortGridM;
    if (case2_grid_x != 8) {
        throw std::runtime_error("case-2 expert cluster width changed");
    }
    if (case4_grid_x != 8) {
        throw std::runtime_error("case-4 expert cluster width is not eight");
    }
    const OutputScratchClusterStats case2_cluster_stats =
        output_scratch_cluster_stats(skewed, skewed_map, case2_grid_x);
    const OutputScratchClusterStats case4_cluster_stats =
        output_scratch_cluster_stats(skewed, skewed_map, case4_grid_x);
    if (case2_cluster_stats.cluster_count != 32
        || case2_cluster_stats.homogeneous_clusters != 32
        || case2_cluster_stats.mixed_clusters != 0
        || case2_cluster_stats.max_cluster_experts != 1) {
        throw std::runtime_error("case-2 public-skew expert clusters changed");
    }
    if (case4_cluster_stats.cluster_count != 32
        || case4_cluster_stats.homogeneous_clusters != 32
        || case4_cluster_stats.mixed_clusters != 0
        || case4_cluster_stats.max_cluster_experts != 1) {
        throw std::runtime_error("case-4 public-skew expert clusters are not pure");
    }

    int case2_grid_y = 0;
    int case2_grid_z = 0;
    int case2_cleanup_ctas = 0;
    int case2_exact_ctas = 0;
    int case4_grid_y = 0;
    int case4_grid_z = 0;
    int case4_cleanup_ctas = 0;
    int case4_exact_ctas = 0;
    for (const PublicCase& public_case : kPublicCases) {
        const bool expected_case2 = std::string(public_case.name) == "prefill-gate-up";
        const bool expected_case4 = std::string(public_case.name) == "prefill-down";
        if (xh_fused_moe::use_case2_output_scratch_expert_sort(public_case.config)
            != expected_case2) {
            throw std::runtime_error(
                std::string("case-2 output scratch dispatch mismatch for ")
                + public_case.name);
        }
        if (xh_fused_moe::use_prefill_down_output_scratch_expert_sort(
                public_case.config) != expected_case4) {
            throw std::runtime_error(
                std::string("case-4 output scratch dispatch mismatch for ")
                + public_case.name);
        }
        const int expected_legacy_grid_x = expected_case2
            ? xh_fused_moe::kOutputScratchSortGridM
            : xh_fused_moe::mma_grid_x(public_case.config);
        if (xh_fused_moe::mma_launch_grid_x(public_case.config)
            != expected_legacy_grid_x) {
            throw std::runtime_error(
                std::string("legacy expert-cluster grid mismatch for ")
                + public_case.name);
        }
        const int expected_grid_x = expected_case4
            ? xh_fused_moe::kPrefillDownOutputScratchSortGridM
            : expected_legacy_grid_x;
        if (xh_fused_moe::output_scratch_sort_launch_grid_x(public_case.config)
            != expected_grid_x) {
            throw std::runtime_error(
                std::string("combined expert-cluster grid mismatch for ")
                + public_case.name);
        }
        const bool expected_sort = expected_case2 || expected_case4;
        const int grid_y = (public_case.config.n + 127) / 128;
        const int grid_m = (public_case.config.em + 127) / 128;
        const int grid_z = (grid_m + expected_grid_x - 1) / expected_grid_x;
        if (expected_sort) {
            const int main_ctas = expected_grid_x * grid_y * grid_z;
            const int skipped_rank_zero_ctas = grid_y;
            const int cleanup_ctas = grid_y;
            const int exact_ctas = main_ctas - skipped_rank_zero_ctas + cleanup_ctas;
            const int expected_grid_y = expected_case2 ? 32 : 56;
            const int expected_ctas = expected_case2 ? 8192 : 14336;
            if (grid_m != xh_fused_moe::kOutputScratchSortTiles
                || grid_y != expected_grid_y
                || grid_z != 32
                || main_ctas != expected_ctas
                || skipped_rank_zero_ctas != expected_grid_y
                || cleanup_ctas != expected_grid_y
                || exact_ctas != expected_ctas) {
                throw std::runtime_error(
                    std::string("output scratch grid/cleanup cover mismatch for ")
                    + public_case.name);
            }
            if (expected_case2) {
                case2_grid_y = grid_y;
                case2_grid_z = grid_z;
                case2_cleanup_ctas = cleanup_ctas;
                case2_exact_ctas = exact_ctas;
            } else {
                case4_grid_y = grid_y;
                case4_grid_z = grid_z;
                case4_cleanup_ctas = cleanup_ctas;
                case4_exact_ctas = exact_ctas;
            }
        }
    }

    constexpr size_t scratch_bytes =
        xh_fused_moe::kOutputScratchSortTiles * sizeof(int32_t);
    constexpr size_t case2_body_first_write_byte =
        static_cast<size_t>(128) * 4096 * sizeof(uint16_t);
    constexpr size_t case4_body_first_write_byte =
        static_cast<size_t>(128) * 7168 * sizeof(uint16_t);
    constexpr size_t case2_tile_zero_overwrite_bytes = case2_body_first_write_byte;
    constexpr size_t case4_tile_zero_overwrite_bytes = case4_body_first_write_byte;
    constexpr size_t case2_one_a_tile_bytes = static_cast<size_t>(128) * 7168;
    constexpr size_t case2_one_b_tile_bytes = static_cast<size_t>(128) * 7168;
    constexpr size_t case4_one_a_tile_bytes = static_cast<size_t>(128) * 2048;
    constexpr size_t case4_one_b_tile_bytes = static_cast<size_t>(128) * 2048;
    constexpr size_t case2_homogeneous_working_set_bytes =
        xh_fused_moe::kOutputScratchSortGridM * case2_one_a_tile_bytes
        + case2_one_b_tile_bytes;
    constexpr size_t case4_homogeneous_working_set_bytes =
        xh_fused_moe::kPrefillDownOutputScratchSortGridM * case4_one_a_tile_bytes
        + case4_one_b_tile_bytes;
    constexpr size_t modeled_l2_bytes = 8U * 1024U * 1024U;
    const size_t case2_max_cluster_working_set_bytes =
        xh_fused_moe::kOutputScratchSortGridM * case2_one_a_tile_bytes
        + static_cast<size_t>(case2_cluster_stats.max_cluster_experts)
            * case2_one_b_tile_bytes;
    const size_t case4_max_cluster_working_set_bytes =
        xh_fused_moe::kPrefillDownOutputScratchSortGridM * case4_one_a_tile_bytes
        + static_cast<size_t>(case4_cluster_stats.max_cluster_experts)
            * case4_one_b_tile_bytes;
    static_assert(scratch_bytes == 1024, "sort map is not exactly 1 KiB");
    static_assert(
        case2_body_first_write_byte == 1048576,
        "case-2 body write boundary changed");
    static_assert(
        case4_body_first_write_byte == 1835008,
        "case-4 body write boundary changed");
    static_assert(
        case2_homogeneous_working_set_bytes == 8257536,
        "case-2 homogeneous working set changed");
    static_assert(
        case4_homogeneous_working_set_bytes == 2359296,
        "case-4 homogeneous working set changed");
    static_assert(
        scratch_bytes < case2_body_first_write_byte
            && scratch_bytes < case2_tile_zero_overwrite_bytes
            && scratch_bytes < case4_body_first_write_byte
            && scratch_bytes < case4_tile_zero_overwrite_bytes,
        "sort map overlaps a body or escapes tile zero");
    static_assert(
        case2_homogeneous_working_set_bytes <= modeled_l2_bytes
            && case4_homogeneous_working_set_bytes <= modeled_l2_bytes,
        "homogeneous expert cluster exceeds modeled 8 MiB L2");
    if (case2_max_cluster_working_set_bytes > modeled_l2_bytes) {
        throw std::runtime_error("case-2 public-skew cluster exceeds modeled 8 MiB L2");
    }
    if (case4_max_cluster_working_set_bytes > modeled_l2_bytes) {
        throw std::runtime_error("case-4 public-skew cluster exceeds modeled 8 MiB L2");
    }

    std::cout << "REGRESSION maca-output-scratch-sort exhaustive-patterns="
              << exhaustive_patterns
              << " full-distributions=all-same,all-unique,skewed"
              << " stable-order=PASS bijection=PASS"
              << " case2-exact-mn-cover=" << case2_exact_ctas << "/8192"
              << " case4-exact-mn-cover=" << case4_exact_ctas << "/14336"
              << " case2-grid=(" << case2_grid_x << "," << case2_grid_y
              << "," << case2_grid_z << ")"
              << " case4-grid=(" << case4_grid_x << "," << case4_grid_y
              << "," << case4_grid_z << ")"
              << " case2-cleanup-ctas=" << case2_cleanup_ctas
              << " case4-cleanup-ctas=" << case4_cleanup_ctas
              << " expert-count-intervals=PASS"
              << " case2-skewed-homogeneous-clusters="
              << case2_cluster_stats.homogeneous_clusters
              << " case2-skewed-mixed-clusters=" << case2_cluster_stats.mixed_clusters
              << " case4-skewed-homogeneous-clusters="
              << case4_cluster_stats.homogeneous_clusters
              << " case4-skewed-mixed-clusters=" << case4_cluster_stats.mixed_clusters
              << " scratch-bytes=" << scratch_bytes
              << " case2-body-first-write-byte=" << case2_body_first_write_byte
              << " case4-body-first-write-byte=" << case4_body_first_write_byte
              << " case2-tile-zero-overwrite-byte="
              << case2_tile_zero_overwrite_bytes
              << " case4-tile-zero-overwrite-byte="
              << case4_tile_zero_overwrite_bytes
              << " case2-homogeneous-working-set-bytes="
              << case2_homogeneous_working_set_bytes
              << " case4-homogeneous-working-set-bytes="
              << case4_homogeneous_working_set_bytes
              << " case2-max-cluster-working-set-bytes="
              << case2_max_cluster_working_set_bytes
              << " case4-max-cluster-working-set-bytes="
              << case4_max_cluster_working_set_bytes
              << " case2-modeled-l2-headroom-bytes="
              << (modeled_l2_bytes - case2_max_cluster_working_set_bytes)
              << " case4-modeled-l2-headroom-bytes="
              << (modeled_l2_bytes - case4_max_cluster_working_set_bytes)
              << " input-readonly=PASS device-map=PASS\n";
}

void benchmark_public_case(const PublicCase& public_case) {
    const auto& config = public_case.config;
    const size_t a_count = static_cast<size_t>(config.em) * config.k;
    const size_t b_count = static_cast<size_t>(256) * config.n * config.k;
    const size_t out_count = static_cast<size_t>(config.em) * config.n;
    DeviceBuffer<int8_t> dev_a(a_count);
    DeviceBuffer<int8_t> dev_b(b_count);
    DeviceBuffer<float> dev_scale_a(config.em);
    DeviceBuffer<float> dev_scale_b(static_cast<size_t>(256) * config.n);
    DeviceBuffer<float> dev_moe_weights(config.em);
    DeviceBuffer<int32_t> dev_expert_ids(config.em / 128);
    DeviceBuffer<uint16_t> dev_out(out_count);
    CUDA_CHECK(cudaMemset(dev_a.get(), 1, a_count));
    CUDA_CHECK(cudaMemset(dev_b.get(), 1, b_count));
    fill_float(dev_scale_a.get(), config.em, 1.0f);
    fill_float(dev_scale_b.get(), static_cast<size_t>(256) * config.n, 1.0f);
    fill_float(dev_moe_weights.get(), config.em, 1.0f);
    std::vector<int32_t> experts(config.em / 128);
    for (size_t tile = 0; tile < experts.size(); ++tile) {
        experts[tile] = public_case.skewed
            ? static_cast<int32_t>((tile * tile + 3 * tile) % 16)
            : static_cast<int32_t>((tile * 73) % 256);
    }
    dev_expert_ids.copy_from(experts);
    CUDA_CHECK(cudaDeviceSynchronize());

    for (int warmup = 0; warmup < 1; ++warmup) {
        xh_fused_moe::launch(
            dev_a.get(), dev_b.get(), dev_scale_a.get(), dev_scale_b.get(),
            dev_moe_weights.get(), dev_expert_ids.get(),
            reinterpret_cast<__nv_bfloat16*>(dev_out.get()), config);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    std::vector<float> milliseconds;
    for (int iteration = 0; iteration < 5; ++iteration) {
        CUDA_CHECK(cudaEventRecord(start));
        xh_fused_moe::launch(
            dev_a.get(), dev_b.get(), dev_scale_a.get(), dev_scale_b.get(),
            dev_moe_weights.get(), dev_expert_ids.get(),
            reinterpret_cast<__nv_bfloat16*>(dev_out.get()), config);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float elapsed = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));
        milliseconds.push_back(elapsed);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaGetLastError());

    const uint16_t expected = float_to_bf16(static_cast<float>(config.k));
    const size_t sample_indices[] = {
        0,
        static_cast<size_t>(127) * config.n + (config.n - 1),
        static_cast<size_t>(128) * config.n,
        out_count - 1,
    };
    for (size_t index : sample_indices) {
        uint16_t actual = 0;
        CUDA_CHECK(cudaMemcpy(
            &actual, dev_out.get() + index, sizeof(actual), cudaMemcpyDeviceToHost));
        if (actual != expected) {
            throw std::runtime_error(std::string("public-shape sample mismatch for ") + public_case.name);
        }
    }

    std::vector<float> sorted = milliseconds;
    std::sort(sorted.begin(), sorted.end());
    const double median_ms = sorted[sorted.size() / 2];
    const double operations = 2.0 * config.em * config.n * config.k;
    const double tops = operations / (median_ms * 1.0e9);
    std::cout << "BENCHMARK device=" << kBenchmarkDevice
              << " case=" << public_case.name << " samples_ms=";
    for (size_t i = 0; i < milliseconds.size(); ++i) {
        std::cout << (i == 0 ? "" : ",") << std::fixed << std::setprecision(3) << milliseconds[i];
    }
    std::cout << " median_ms=" << median_ms << " effective_TOPS=" << tops
              << " sampled_correctness=PASS\n";
}

void run_correctness() {
    run_small_case({{128, 32, 256}, 5, 0x12345678U, 0, "random-boundary-128"});
    run_small_case({{256, 64, 512}, 7, 0x9abcdef0U, 0, "random-two-expert-tiles"});
    run_small_case({{128, 64, 256}, 3, 0x31415926U, 2, "int8-extremes"});
}

void run_benchmark() {
    benchmark_inference_query_path();
    for (const PublicCase& public_case : kPublicCases) {
        benchmark_public_case(public_case);
    }
}

void verify_mma_a_load_bounds() {
    constexpr int tile_rows = 128;
    constexpr int rows_per_load = 32;
    constexpr int loads_per_thread = 4;
    for (const PublicCase& public_case : kPublicCases) {
        if ((public_case.config.em % tile_rows) != 0
            || (public_case.config.k % 128) != 0) {
            throw std::runtime_error(
                std::string("public shape is not exact-tile aligned for ") + public_case.name);
        }
        for (int thread_id = 0; thread_id < 256; ++thread_id) {
            for (int load = 0; load < loads_per_thread; ++load) {
                const int local_row = thread_id / 8 + rows_per_load * load;
                if (local_row < 0 || local_row >= tile_rows) {
                    throw std::runtime_error("MMA A load row is outside its exact tile");
                }
            }
        }
        const int predicates_removed_per_thread =
            loads_per_thread * (public_case.config.k / 128);
        std::cout << "REGRESSION maca-a-load-bounds case=" << public_case.name
                  << " exact-rows=PASS predicates-removed-per-thread="
                  << predicates_removed_per_thread << "\n";
    }
}

void verify_mma_row_metadata_unpredicated() {
    constexpr int tile_rows = 128;
    constexpr int threads = 256;
    constexpr int row_groups = 2;
    constexpr int rows_per_group = 4;
    constexpr int metadata_arrays = 2;
    constexpr int bytes_per_scalar = sizeof(float);
    constexpr int loads_per_thread = row_groups * rows_per_group * metadata_arrays;
    constexpr int loads_per_cta = threads * loads_per_thread;
    constexpr int dynamic_bytes_per_cta = loads_per_cta * bytes_per_scalar;
    constexpr int unique_bytes_per_cta = tile_rows * metadata_arrays * bytes_per_scalar;
    constexpr int predicate_compares_removed_per_cta = loads_per_cta;
    constexpr int expected_row_hits =
        threads * row_groups * rows_per_group / tile_rows;
    static_assert(loads_per_thread == 16, "row metadata load count changed");
    static_assert(loads_per_cta == 4096, "CTA row metadata load count changed");
    static_assert(dynamic_bytes_per_cta == 16384, "CTA row metadata bytes changed");
    static_assert(unique_bytes_per_cta == 1024, "CTA row metadata footprint changed");
    static_assert(expected_row_hits == 16, "row metadata reuse factor changed");

    for (const PublicCase& public_case : kPublicCases) {
        const int em = public_case.config.em;
        if ((em % tile_rows) != 0) {
            throw std::runtime_error(
                std::string("row metadata shape is not exact-tile aligned for ")
                + public_case.name);
        }
        std::vector<int> row_hits(em, 0);
        std::vector<uint32_t> weight_bits(em);
        std::vector<uint32_t> scale_bits(em);
        for (int row = 0; row < em; ++row) {
            weight_bits[row] = 0x3f000000U ^ static_cast<uint32_t>(row * 0x9e3779b9U);
            scale_bits[row] = 0x40000000U ^ static_cast<uint32_t>(row * 0x85ebca6bU);
        }
        for (int row_base = 0; row_base < em; row_base += tile_rows) {
            for (int thread_id = 0; thread_id < threads; ++thread_id) {
                for (int row_group = 0; row_group < row_groups; ++row_group) {
                    for (int row_in_group = 0; row_in_group < rows_per_group;
                         ++row_in_group) {
                        const int local_row = xh_fused_moe::mma_output_row_local(
                            thread_id, row_group, row_in_group);
                        const int global_row = row_base + local_row;
                        if (local_row < 0 || local_row >= tile_rows
                            || global_row < row_base || global_row >= row_base + tile_rows
                            || global_row >= em) {
                            throw std::runtime_error(
                                std::string("unpredicated row metadata is out of bounds for ")
                                + public_case.name);
                        }
                        const bool baseline_active = global_row < em;
                        const uint32_t baseline_weight =
                            baseline_active ? weight_bits[global_row] : 0U;
                        const uint32_t baseline_scale =
                            baseline_active ? scale_bits[global_row] : 0U;
                        if (baseline_weight != weight_bits[global_row]
                            || baseline_scale != scale_bits[global_row]) {
                            throw std::runtime_error(
                                "unpredicated row metadata changed a loaded value");
                        }
                        ++row_hits[global_row];
                    }
                }
            }
        }
        if (!std::all_of(
                row_hits.begin(), row_hits.end(),
                [](int count) { return count == expected_row_hits; })) {
            throw std::runtime_error(
                std::string("row metadata coverage changed for ") + public_case.name);
        }
        std::cout << "REGRESSION maca-row-metadata-unpredicated case="
                  << public_case.name
                  << " exact-bounds=PASS exact-address-values=PASS row-hits="
                  << expected_row_hits
                  << " loads-per-thread=" << loads_per_thread
                  << " loads-per-cta=" << loads_per_cta
                  << " dynamic-bytes-per-cta=" << dynamic_bytes_per_cta
                  << " unique-bytes-per-cta=" << unique_bytes_per_cta
                  << " predicate-compares-removed-per-cta="
                  << predicate_compares_removed_per_cta << "\n";
    }
}

void run_regression() {
    verify_public_inference();
    verify_plain_b64_store_mapping();
    verify_mma_output_mapping();
    verify_mma_grid_mapping();
    verify_output_scratch_sort_structure();
    verify_mma_a_load_bounds();
    verify_mma_row_metadata_unpredicated();
    run_small_case({{128, 32, 256}, 2, 0x27182818U, 1, "all-zero-readonly"});
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 2) {
            std::cerr << "usage: test_fused_moe_i8_tn --correctness|--benchmark|--regression\n";
            return 64;
        }
        const std::string mode(argv[1]);
        if (mode == "--correctness") {
            run_correctness();
        } else if (mode == "--benchmark") {
            run_benchmark();
        } else if (mode == "--regression") {
            run_regression();
        } else {
            std::cerr << "unknown mode: " << mode << "\n";
            return 64;
        }
        std::cout << "SUITE " << mode << " PASS\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << "\n";
        return 1;
    }
}
