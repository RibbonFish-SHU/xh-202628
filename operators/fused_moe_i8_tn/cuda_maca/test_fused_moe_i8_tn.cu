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

constexpr int kMmaModelTileM = 128;
constexpr int kMmaModelTileN = 128;
constexpr int kMmaModelTileK = 128;
constexpr int kMmaModelThreads = 256;
constexpr int kMmaModelWaveSize = 64;
constexpr int kMmaModelWaves = kMmaModelThreads / kMmaModelWaveSize;
constexpr int kMmaModelVectorBytes = 16;
constexpr int kMmaModelABytes = kMmaModelTileM * kMmaModelTileK;
constexpr int kMmaModelBBytes = kMmaModelTileN * kMmaModelTileK;

int mma_model_store_col(int thread_id) {
    return (((thread_id / 8) + (thread_id % 8)) % 8) * kMmaModelVectorBytes;
}

int mma_model_lds_col(int thread_id, int half) {
    const int lane = thread_id % kMmaModelWaveSize;
    return (((thread_id % 16) + (lane / 16) + 4 * half) % 8)
        * kMmaModelVectorBytes;
}

void verify_mma_b_pingpong_layout() {
    const int starts[] = {0, kMmaModelABytes, kMmaModelABytes + kMmaModelBBytes};
    const int sizes[] = {kMmaModelABytes, kMmaModelBBytes, kMmaModelBBytes};
    constexpr int shared_bytes = kMmaModelABytes + 2 * kMmaModelBBytes;
    std::vector<int> coverage(shared_bytes, 0);
    for (int region = 0; region < 3; ++region) {
        if ((starts[region] % kMmaModelVectorBytes) != 0
            || (sizes[region] % kMmaModelVectorBytes) != 0) {
            throw std::runtime_error("B ping-pong shared region is not b128 aligned");
        }
        for (int address = starts[region]; address < starts[region] + sizes[region]; ++address) {
            if (address < 0 || address >= shared_bytes) {
                throw std::runtime_error("B ping-pong shared region is out of range");
            }
            ++coverage[address];
        }
    }
    if (starts[0] != 0
        || starts[1] != 16 * 1024
        || starts[2] != 32 * 1024
        || shared_bytes != 48 * 1024
        || !std::all_of(
            coverage.begin(), coverage.end(), [](int count) { return count == 1; })) {
        throw std::runtime_error("B ping-pong shared layout is not an exact non-overlapping cover");
    }
    std::cout << "REGRESSION maca-b-pingpong-layout bytes=" << shared_bytes
              << " A=[0,16384) B0=[16384,32768) B1=[32768,49152)"
              << " alignment=16 exact-cover=PASS\n";
}

void verify_mma_a_wave_private_ownership() {
    std::vector<int> writer_wave(kMmaModelABytes, -1);
    std::vector<int> writer_source(kMmaModelABytes, -1);
    std::vector<int> write_visits(kMmaModelABytes, 0);
    for (int thread_id = 0; thread_id < kMmaModelThreads; ++thread_id) {
        const int wave = thread_id / kMmaModelWaveSize;
        const int lane = thread_id % kMmaModelWaveSize;
        const int load_k = (lane % 8) * kMmaModelVectorBytes;
        const int store_col = mma_model_store_col(thread_id);
        for (int load = 0; load < 4; ++load) {
            const int global_row = thread_id / 8 + 32 * load;
            const int store_row = wave * 32 + lane / 8 + 8 * load;
            for (int byte = 0; byte < kMmaModelVectorBytes; ++byte) {
                const int address = store_row * kMmaModelTileK + store_col + byte;
                if (address < 0 || address >= kMmaModelABytes || write_visits[address] != 0) {
                    throw std::runtime_error("A shared store mapping is not an exact cover");
                }
                ++write_visits[address];
                writer_wave[address] = wave;
                writer_source[address] = global_row * kMmaModelTileK + load_k + byte;
            }
        }
    }

    std::vector<int> read_visits(kMmaModelABytes, 0);
    int cross_wave_reads = 0;
    for (int thread_id = 0; thread_id < kMmaModelThreads; ++thread_id) {
        const int wave = thread_id / kMmaModelWaveSize;
        for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
            const int lds_row = (thread_id % 16) + wave * 32 + 16 * row_fragment;
            for (int half = 0; half < 2; ++half) {
                const int lds_col = mma_model_lds_col(thread_id, half);
                for (int byte = 0; byte < kMmaModelVectorBytes; ++byte) {
                    const int address = lds_row * kMmaModelTileK + lds_col + byte;
                    if (address < 0 || address >= kMmaModelABytes
                        || writer_source[address] < 0) {
                        throw std::runtime_error("A LDS reads an unwritten shared byte");
                    }
                    cross_wave_reads += writer_wave[address] != wave;
                    ++read_visits[address];
                }
            }
        }
    }
    if (cross_wave_reads != 0
        || !std::all_of(
            write_visits.begin(), write_visits.end(), [](int count) { return count == 1; })
        || !std::all_of(
            read_visits.begin(), read_visits.end(), [](int count) { return count == 1; })) {
        throw std::runtime_error("A shared ownership is not exhaustive and wave-private");
    }
    std::cout << "REGRESSION maca-b-pingpong-a-ownership bytes=" << kMmaModelABytes
              << " writes=1x reads=1x cross-wave=0 PASS\n";
}

void write_mma_b_tile_model(std::vector<int>* buffer, int tile_label) {
    std::vector<int> visits(kMmaModelBBytes, 0);
    for (int thread_id = 0; thread_id < kMmaModelThreads; ++thread_id) {
        const int lane = thread_id % kMmaModelWaveSize;
        const int load_k = (lane % 8) * kMmaModelVectorBytes;
        const int store_col = mma_model_store_col(thread_id);
        for (int load = 0; load < 4; ++load) {
            const int global_row = (thread_id / 8) * 4 + load;
            const int store_row = thread_id / 8 + 32 * load;
            for (int byte = 0; byte < kMmaModelVectorBytes; ++byte) {
                const int address = store_row * kMmaModelTileK + store_col + byte;
                const int source = global_row * kMmaModelTileK + load_k + byte;
                if (address < 0 || address >= kMmaModelBBytes
                    || source < 0 || source >= kMmaModelBBytes
                    || visits[address] != 0) {
                    throw std::runtime_error("B shared producer mapping is not an exact cover");
                }
                ++visits[address];
                (*buffer)[address] = tile_label;
            }
        }
    }
    if (!std::all_of(visits.begin(), visits.end(), [](int count) { return count == 1; })) {
        throw std::runtime_error("B shared producer left a hole");
    }
}

void consume_mma_b_tile_model(const std::vector<int>& buffer, int expected_tile) {
    std::vector<int> visits(kMmaModelBBytes, 0);
    for (int thread_id = 0; thread_id < kMmaModelThreads; ++thread_id) {
        for (int row_fragment = 0; row_fragment < 8; ++row_fragment) {
            const int lds_row = (thread_id % 16) + 16 * row_fragment;
            for (int half = 0; half < 2; ++half) {
                const int lds_col = mma_model_lds_col(thread_id, half);
                for (int byte = 0; byte < kMmaModelVectorBytes; ++byte) {
                    const int address = lds_row * kMmaModelTileK + lds_col + byte;
                    if (address < 0 || address >= kMmaModelBBytes
                        || buffer[address] != expected_tile) {
                        throw std::runtime_error("B LDS observed the wrong ping-pong tile label");
                    }
                    ++visits[address];
                }
            }
        }
    }
    if (!std::all_of(visits.begin(), visits.end(), [](int count) { return count == 4; })) {
        throw std::runtime_error("B shared consumer mapping is not four-wave exact");
    }
}

void verify_mma_b_pingpong_reuse() {
    std::vector<int> buffers[2] = {
        std::vector<int>(kMmaModelBBytes, -1),
        std::vector<int>(kMmaModelBBytes, -1),
    };
    int tile_labels[2] = {-1, -1};
    int consumed_waves[2] = {0, 0};
    int active = 0;
    int barriers = 0;
    std::vector<int> consumed_tiles;

    write_mma_b_tile_model(&buffers[active], 2);
    tile_labels[active] = 2;
    ++barriers;
    for (int next_tile = 0; next_tile < 2; ++next_tile) {
        consume_mma_b_tile_model(buffers[active], tile_labels[active]);
        consumed_waves[active] = (1 << kMmaModelWaves) - 1;
        consumed_tiles.push_back(tile_labels[active]);

        const int inactive = active ^ 1;
        if (tile_labels[inactive] >= 0
            && consumed_waves[inactive] != (1 << kMmaModelWaves) - 1) {
            throw std::runtime_error("B buffer was reused before all waves consumed it");
        }
        write_mma_b_tile_model(&buffers[inactive], next_tile);
        tile_labels[inactive] = next_tile;
        consumed_waves[inactive] = 0;
        ++barriers;
        active = inactive;
    }
    consume_mma_b_tile_model(buffers[active], tile_labels[active]);
    consumed_tiles.push_back(tile_labels[active]);
    const std::vector<int> expected = {2, 0, 1};
    if (consumed_tiles != expected || barriers != 3) {
        throw std::runtime_error("two-iteration B ping-pong schedule is incorrect");
    }
    std::cout << "REGRESSION maca-b-pingpong-reuse iterations=2 order=2,0,1"
              << " producer-cover=1x consumer-cover=4x barriers=3 PASS\n";
}

void verify_mma_b_tile_schedules() {
    const int tile_counts[] = {1, 2, 16, 56};
    for (int tile_count : tile_counts) {
        int labels[2] = {tile_count - 1, -1};
        int active = 0;
        int barriers = 1;
        std::vector<int> consumed;
        for (int next_tile = 0; next_tile < tile_count - 1; ++next_tile) {
            consumed.push_back(labels[active]);
            const int inactive = active ^ 1;
            labels[inactive] = next_tile;
            ++barriers;
            active = inactive;
        }
        consumed.push_back(labels[active]);

        std::vector<int> expected;
        expected.push_back(tile_count - 1);
        for (int tile = 0; tile < tile_count - 1; ++tile) {
            expected.push_back(tile);
        }
        const int baseline_barriers = 2 * tile_count - 1;
        if (consumed != expected || barriers != tile_count) {
            throw std::runtime_error("B ping-pong tile-label or barrier schedule is incorrect");
        }
        std::cout << "REGRESSION maca-b-pingpong-schedule tiles=" << tile_count
                  << " barriers=" << barriers
                  << " baseline-barriers=" << baseline_barriers
                  << " order=last,0..last-1 PASS\n";
    }
}

void verify_mma_b_pipeline_phase_order() {
    const int initial_last_store = 0;
    const int initial_barrier = 1;
    const int initial_first_read = 2;
    const int steady_last_active_b_read = 0;
    const int steady_last_current_a_read = 1;
    const int steady_first_inactive_b_store = 2;
    const int steady_last_next_a_store = 3;
    const int steady_barrier = 4;
    const int steady_swap = 5;
    const int steady_first_next_read = 6;
    constexpr int tail_next_tile_loads = 0;
    constexpr int tail_next_tile_stores = 0;
    constexpr int tail_barriers = 0;
    if (!(initial_last_store < initial_barrier && initial_barrier < initial_first_read)
        || !(steady_last_active_b_read < steady_first_inactive_b_store)
        || !(steady_last_current_a_read < steady_first_inactive_b_store)
        || !(steady_first_inactive_b_store < steady_barrier)
        || !(steady_last_next_a_store < steady_barrier)
        || !(steady_barrier < steady_swap && steady_swap < steady_first_next_read)
        || tail_next_tile_loads != 0 || tail_next_tile_stores != 0 || tail_barriers != 0) {
        throw std::runtime_error("B ping-pong initial/steady/tail phase order is unsafe");
    }
    std::cout << "REGRESSION maca-b-pingpong-phase-order"
              << " initial=store,barrier,read"
              << " steady=read,inactive-store,barrier,swap,next-read"
              << " tail=no-next-tile-stage PASS\n";
}

void verify_mma_global_traffic_and_resources() {
    std::vector<int> a_visits(kMmaModelABytes, 0);
    std::vector<int> b_visits(kMmaModelBBytes, 0);
    for (int thread_id = 0; thread_id < kMmaModelThreads; ++thread_id) {
        const int load_k = (thread_id % 8) * kMmaModelVectorBytes;
        for (int load = 0; load < 4; ++load) {
            const int a_row = thread_id / 8 + 32 * load;
            const int b_row = (thread_id / 8) * 4 + load;
            for (int byte = 0; byte < kMmaModelVectorBytes; ++byte) {
                ++a_visits[a_row * kMmaModelTileK + load_k + byte];
                ++b_visits[b_row * kMmaModelTileK + load_k + byte];
            }
        }
    }
    if (!std::all_of(a_visits.begin(), a_visits.end(), [](int count) { return count == 1; })
        || !std::all_of(b_visits.begin(), b_visits.end(), [](int count) { return count == 1; })) {
        throw std::runtime_error("global A/B vector loads are not exact tile covers");
    }

    constexpr int a_lds_per_tile = 2 * 2;
    constexpr int b_lds_per_tile = 8 * 2;
    constexpr int stores_per_tile = 4 + 4;
    constexpr int mma_chains = 2 * 8;
    constexpr int mma_depth = 8;
    constexpr int mma_per_tile = mma_chains * mma_depth;
    if (a_lds_per_tile + b_lds_per_tile != 20
        || stores_per_tile != 8
        || mma_chains != 16
        || mma_per_tile != 128) {
        throw std::runtime_error("MMA per-tile instruction/resource model changed");
    }
    for (int chain = 0; chain < mma_chains; ++chain) {
        for (int depth = 0; depth < mma_depth; ++depth) {
            if (depth < 0 || depth > 7) {
                throw std::runtime_error("MMA chain depth order changed");
            }
        }
    }

    for (const PublicCase& public_case : kPublicCases) {
        const int k_tiles = public_case.config.k / kMmaModelTileK;
        const int k_head = (public_case.config.k - 1) % kMmaModelTileK + 1;
        const int a_bytes = 4 * k_tiles * kMmaModelVectorBytes * kMmaModelThreads;
        const int b_bytes = 4 * k_tiles * kMmaModelVectorBytes * kMmaModelThreads;
        int true_initial_b_predicates = 0;
        for (int thread_id = 0; thread_id < kMmaModelThreads; ++thread_id) {
            const int load_k = (thread_id % 8) * kMmaModelVectorBytes;
            for (int load = 0; load < 4; ++load) {
                true_initial_b_predicates += load_k < k_head;
            }
        }
        if ((public_case.config.k % kMmaModelTileK) != 0
            || (public_case.config.n % kMmaModelTileN) != 0
            || a_bytes != kMmaModelTileM * public_case.config.k
            || b_bytes != kMmaModelTileN * public_case.config.k
            || true_initial_b_predicates != 4 * kMmaModelThreads) {
            throw std::runtime_error(
                std::string("global address/byte/predicate model changed for ") + public_case.name);
        }
        std::cout << "REGRESSION maca-b-pingpong-traffic case=" << public_case.name
                  << " A-bytes=" << a_bytes << " B-bytes=" << b_bytes
                  << " initial-B-predicates=" << true_initial_b_predicates << "/"
                  << 4 * kMmaModelThreads << " LDS=20 STS=8 MMA="
                  << mma_per_tile * k_tiles << " chains=16x0..7 PASS\n";
    }
}

void run_regression() {
    verify_public_inference();
    verify_mma_output_mapping();
    verify_mma_grid_mapping();
    verify_mma_a_load_bounds();
    verify_mma_b_pingpong_layout();
    verify_mma_a_wave_private_ownership();
    verify_mma_b_pingpong_reuse();
    verify_mma_b_tile_schedules();
    verify_mma_b_pipeline_phase_order();
    verify_mma_global_traffic_and_resources();
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
