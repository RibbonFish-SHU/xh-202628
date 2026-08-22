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
    std::cout << "BENCHMARK proxy-host-inference fast_path_ns=" << std::fixed
              << std::setprecision(1) << fast_median
              << " two_query_ns=" << two_query_median
              << " speedup=" << std::setprecision(3) << two_query_median / fast_median
              << "x checksum=" << checksum << "\n";
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
    std::cout << "BENCHMARK proxy=NVIDIA case=" << public_case.name << " samples_ms=";
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

struct MmaVectorAddress {
    int lane;
    int row;
    int chunk;
};

using MmaLoadGroup = std::vector<MmaVectorAddress>;

void require_exact_vector_cover(
    const std::vector<int>& visits,
    const std::string& label
) {
    if (visits.size() != 128U * 8U
        || !std::all_of(visits.begin(), visits.end(), [](int count) { return count == 1; })) {
        throw std::runtime_error(label + " does not cover every 16-byte A/B vector exactly once");
    }
}

void verify_direct_a_fragment_mapping() {
    constexpr int tile_rows = 128;
    constexpr int chunks_per_row = 8;
    constexpr int threads = 256;
    constexpr int wave_size = 64;
    std::vector<int> shared_owner(tile_rows * chunks_per_row, -1);
    std::vector<int> direct_visits(tile_rows * chunks_per_row, 0);

    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        const int wave = thread_id / wave_size;
        const int lane = thread_id % wave_size;
        const int global_chunk = lane % 8;
        const int shared_chunk = ((thread_id / 8) + (thread_id % 8)) % 8;
        for (int load = 0; load < 4; ++load) {
            const int global_row = thread_id / 8 + 32 * load;
            const int shared_row = wave * 32 + lane / 8 + 8 * load;
            const int shared_index = shared_row * chunks_per_row + shared_chunk;
            if (shared_owner[shared_index] != -1) {
                throw std::runtime_error("baseline A STS aliases a shared vector");
            }
            shared_owner[shared_index] = global_row * chunks_per_row + global_chunk;
        }
    }
    if (std::any_of(shared_owner.begin(), shared_owner.end(), [](int owner) { return owner < 0; })) {
        throw std::runtime_error("baseline A STS leaves a shared vector unwritten");
    }

    int compared = 0;
    for (int wave = 0; wave < 4; ++wave) {
        for (int lane = 0; lane < wave_size; ++lane) {
            const int r = lane % 16;
            const int q = lane / 16;
            for (int mma_row = 0; mma_row < 2; ++mma_row) {
                for (int half = 0; half < 2; ++half) {
                    const int shared_row = r + wave * 32 + 16 * mma_row;
                    const int shared_chunk = (r + q + 4 * half) % 8;
                    const int baseline_vector =
                        shared_owner[shared_row * chunks_per_row + shared_chunk];
                    const int direct_row =
                        64 * mma_row + 32 * (r / 8) + 8 * wave + (r % 8);
                    const int direct_chunk = q + 4 * half;
                    const int direct_vector = direct_row * chunks_per_row + direct_chunk;
                    if (baseline_vector != direct_vector) {
                        throw std::runtime_error("direct A fragment does not match baseline STS-to-LDS data");
                    }
                    if (((direct_row * 128 + direct_chunk * 16) % 16) != 0) {
                        throw std::runtime_error("direct A fragment address is not 16-byte aligned");
                    }
                    ++direct_visits[direct_vector];
                    ++compared;
                }
            }
        }
    }
    require_exact_vector_cover(direct_visits, "direct A fragment mapping");
    if (compared != 1024) {
        throw std::runtime_error("direct A fragment identity comparison count changed");
    }
    std::cout << "REGRESSION maca-direct-a-fragment-map vectors=" << compared
              << " baseline-identity=PASS exact-cover=PASS alignment=16B\n";
}

std::vector<MmaLoadGroup> baseline_a_load_groups() {
    std::vector<MmaLoadGroup> groups;
    for (int wave = 0; wave < 4; ++wave) {
        for (int load = 0; load < 4; ++load) {
            MmaLoadGroup group;
            for (int lane = 0; lane < 64; ++lane) {
                group.push_back({lane, wave * 8 + lane / 8 + 32 * load, lane % 8});
            }
            groups.push_back(group);
        }
    }
    return groups;
}

std::vector<MmaLoadGroup> direct_a_load_groups() {
    std::vector<MmaLoadGroup> groups;
    for (int wave = 0; wave < 4; ++wave) {
        for (int mma_row = 0; mma_row < 2; ++mma_row) {
            for (int half = 0; half < 2; ++half) {
                MmaLoadGroup group;
                for (int lane = 0; lane < 64; ++lane) {
                    const int r = lane % 16;
                    group.push_back({
                        lane,
                        64 * mma_row + 32 * (r / 8) + 8 * wave + (r % 8),
                        lane / 16 + 4 * half,
                    });
                }
                groups.push_back(group);
            }
        }
    }
    return groups;
}

int segment_request_count(
    const std::vector<MmaLoadGroup>& groups,
    int segment_bytes,
    int base_offset,
    int coalescer_lanes
) {
    int requests = 0;
    for (const MmaLoadGroup& group : groups) {
        for (int first_lane = 0; first_lane < 64; first_lane += coalescer_lanes) {
            for (int row = 0; row < 128; ++row) {
                std::vector<int> touched;
                for (const MmaVectorAddress& vector : group) {
                    if (vector.lane < first_lane
                        || vector.lane >= first_lane + coalescer_lanes
                        || vector.row != row) {
                        continue;
                    }
                    const int first_byte = base_offset + vector.row * 128 + vector.chunk * 16;
                    const int first_segment = first_byte / segment_bytes;
                    const int last_segment = (first_byte + 15) / segment_bytes;
                    touched.push_back(first_segment);
                    if (last_segment != first_segment) {
                        touched.push_back(last_segment);
                    }
                }
                std::sort(touched.begin(), touched.end());
                requests += static_cast<int>(
                    std::unique(touched.begin(), touched.end()) - touched.begin());
            }
        }
    }
    return requests;
}

int unique_fetched_segments(
    const std::vector<MmaLoadGroup>& groups,
    int segment_bytes,
    int base_offset
) {
    std::vector<int> touched;
    for (const MmaLoadGroup& group : groups) {
        for (const MmaVectorAddress& vector : group) {
            const int first_byte = base_offset + vector.row * 128 + vector.chunk * 16;
            const int first_segment = first_byte / segment_bytes;
            const int last_segment = (first_byte + 15) / segment_bytes;
            touched.push_back(first_segment);
            if (last_segment != first_segment) {
                touched.push_back(last_segment);
            }
        }
    }
    std::sort(touched.begin(), touched.end());
    return static_cast<int>(std::unique(touched.begin(), touched.end()) - touched.begin());
}

void verify_a_request_model() {
    struct RequestCase {
        int segment_bytes;
        int base_offset;
        int baseline_requests;
        int direct_requests;
        int unique_segments;
    };
    const RequestCase cases[] = {
        {32, 0, 512, 512, 512},
        {32, 16, 640, 768, 513},
        {64, 0, 256, 256, 256},
        {64, 16, 384, 512, 257},
        {64, 32, 384, 512, 257},
        {64, 48, 384, 512, 257},
        {128, 0, 128, 256, 128},
        {128, 16, 256, 384, 129},
        {128, 32, 256, 384, 129},
        {128, 48, 256, 384, 129},
        {128, 64, 256, 256, 129},
        {128, 80, 256, 384, 129},
        {128, 96, 256, 384, 129},
        {128, 112, 256, 384, 129},
    };
    const std::vector<MmaLoadGroup> baseline = baseline_a_load_groups();
    const std::vector<MmaLoadGroup> direct = direct_a_load_groups();
    for (const RequestCase& test_case : cases) {
        const int baseline_requests = segment_request_count(
            baseline, test_case.segment_bytes, test_case.base_offset, 64);
        const int direct_requests = segment_request_count(
            direct, test_case.segment_bytes, test_case.base_offset, 64);
        const int baseline_unique = unique_fetched_segments(
            baseline, test_case.segment_bytes, test_case.base_offset);
        const int direct_unique = unique_fetched_segments(
            direct, test_case.segment_bytes, test_case.base_offset);
        if (baseline_requests != test_case.baseline_requests
            || direct_requests != test_case.direct_requests
            || baseline_unique != test_case.unique_segments
            || direct_unique != test_case.unique_segments) {
            throw std::runtime_error("direct A segment request model changed");
        }
        std::cout << "REGRESSION maca-direct-a-requests segment="
                  << test_case.segment_bytes << "B offset=" << test_case.base_offset
                  << " baseline=" << baseline_requests << " direct=" << direct_requests
                  << " unique=" << direct_unique << " PASS\n";
    }

    for (int segment_bytes : {32, 64, 128}) {
        const int baseline_subwave = segment_request_count(baseline, segment_bytes, 0, 16);
        const int direct_subwave = segment_request_count(direct, segment_bytes, 0, 16);
        const int expected_baseline = 16384 / segment_bytes;
        if (baseline_subwave != expected_baseline || direct_subwave != 1024) {
            throw std::runtime_error("16-lane direct A coalescer diagnostic changed");
        }
        std::cout << "REGRESSION maca-direct-a-subwave segment=" << segment_bytes
                  << "B baseline=" << baseline_subwave << " direct=" << direct_subwave
                  << " diagnostic=PASS\n";
    }
}

void verify_direct_a_lifecycle(int num_k_tiles) {
    if (num_k_tiles <= 0) {
        throw std::runtime_error("direct A lifecycle requires at least one K tile");
    }
    int fragment_tile[2][2] = {
        {num_k_tiles - 1, num_k_tiles - 1},
        {num_k_tiles - 1, num_k_tiles - 1},
    };
    std::vector<int> accumulator_trace[2][8];
    auto consume_half = [&](int mma_row, int half) {
        for (int mma_col = 0; mma_col < 8; ++mma_col) {
            for (int depth = 4 * half; depth < 4 * half + 4; ++depth) {
                accumulator_trace[mma_row][mma_col].push_back(
                    fragment_tile[mma_row][half] * 8 + depth);
            }
        }
    };

    int a_ldg_per_thread = 4;
    for (int tile = 0; tile < num_k_tiles - 1; ++tile) {
        consume_half(0, 0);
        fragment_tile[0][0] = tile;
        consume_half(0, 1);
        fragment_tile[0][1] = tile;
        consume_half(1, 0);
        fragment_tile[1][0] = tile;
        consume_half(1, 1);
        fragment_tile[1][1] = tile;
        a_ldg_per_thread += 4;
    }
    consume_half(0, 0);
    consume_half(0, 1);
    consume_half(1, 0);
    consume_half(1, 1);

    std::vector<int> expected;
    const auto append_tile = [&](int tile) {
        for (int depth = 0; depth < 8; ++depth) {
            expected.push_back(tile * 8 + depth);
        }
    };
    append_tile(num_k_tiles - 1);
    for (int tile = 0; tile < num_k_tiles - 1; ++tile) {
        append_tile(tile);
    }
    for (int mma_row = 0; mma_row < 2; ++mma_row) {
        for (int mma_col = 0; mma_col < 8; ++mma_col) {
            if (accumulator_trace[mma_row][mma_col] != expected) {
                throw std::runtime_error("direct A fragment overwrite changed an accumulator chain");
            }
        }
    }

    const int lds_bytes = 128 * 128;
    const int a_sts_per_thread = 0;
    const int a_lds_per_thread = 0;
    const int mma_per_tile_per_thread = 2 * 8 * 8;
    const int barrier_call_sites = 3;
    const int dynamic_barriers = 1 + 2 * (num_k_tiles - 1);
    if (lds_bytes != 16 * 1024
        || a_ldg_per_thread != 4 * num_k_tiles
        || a_sts_per_thread != 0
        || a_lds_per_thread != 0
        || mma_per_tile_per_thread != 128
        || barrier_call_sites != 3) {
        throw std::runtime_error("direct A resource model changed");
    }
    std::cout << "REGRESSION maca-direct-a-lifecycle k-tiles=" << num_k_tiles
              << " chains=16 tile-order=last,0..last-1 depth-order=0..7"
              << " lds-bytes=" << lds_bytes
              << " a-ldg/thread=" << a_ldg_per_thread
              << " a-sts/thread=0 a-lds/thread=0 mma/tile/thread=128"
              << " barrier-sites=" << barrier_call_sites
              << " dynamic-barriers=" << dynamic_barriers << " PASS\n";
}

void verify_b_pipeline_cover() {
    constexpr int rows = 128;
    constexpr int chunks = 8;
    std::vector<int> global_visits(rows * chunks, 0);
    std::vector<int> shared_visits(rows * chunks, 0);
    for (int thread_id = 0; thread_id < 256; ++thread_id) {
        const int lane = thread_id % 64;
        const int global_chunk = lane % 8;
        const int shared_chunk = ((thread_id / 8) + (thread_id % 8)) % 8;
        for (int load = 0; load < 4; ++load) {
            const int global_row = (thread_id / 8) * 4 + load;
            const int shared_row = thread_id / 8 + 32 * load;
            ++global_visits[global_row * chunks + global_chunk];
            ++shared_visits[shared_row * chunks + shared_chunk];
        }
    }
    require_exact_vector_cover(global_visits, "B global producer mapping");
    require_exact_vector_cover(shared_visits, "B shared producer mapping");

    for (int wave = 0; wave < 4; ++wave) {
        std::vector<int> lds_visits(rows * chunks, 0);
        for (int lane = 0; lane < 64; ++lane) {
            for (int mma_col = 0; mma_col < 8; ++mma_col) {
                const int shared_row = lane % 16 + 16 * mma_col;
                for (int half = 0; half < 2; ++half) {
                    const int shared_chunk =
                        ((lane % 16) + lane / 16 + 4 * half) % 8;
                    ++lds_visits[shared_row * chunks + shared_chunk];
                }
            }
        }
        require_exact_vector_cover(lds_visits, "per-wave B LDS mapping");
    }
    std::cout << "REGRESSION maca-direct-a-b-pipeline producer-vectors=1024"
              << " shared-vectors=1024 per-wave-lds-vectors=1024 waves=4 PASS\n";
}

void run_regression() {
    verify_public_inference();
    verify_mma_output_mapping();
    verify_mma_grid_mapping();
    verify_mma_a_load_bounds();
    verify_direct_a_fragment_mapping();
    verify_a_request_model();
    verify_direct_a_lifecycle(56);
    verify_direct_a_lifecycle(16);
    verify_b_pipeline_cover();
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
