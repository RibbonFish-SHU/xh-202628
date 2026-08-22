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
        for (int mma_row = 0; mma_row < 4; ++mma_row) {
            for (int row_in_vector = 0; row_in_vector < 4; ++row_in_vector) {
                const int row = xh_fused_moe::mma_output_row_local(
                    thread_id, mma_row, row_in_vector);
                for (int col_in_vector = 0; col_in_vector < 4; ++col_in_vector) {
                    const int col = xh_fused_moe::mma_output_col_local(
                        thread_id, col_in_vector);
                    if (row < 0 || row >= tile || col < 0 || col >= tile) {
                        throw std::runtime_error("MMA output mapping is out of range");
                    }
                    ++visits[row * tile + col];
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

void verify_mma_wave_fragment_mapping() {
    constexpr int threads = 256;
    constexpr int wave_size = 64;
    constexpr int tile = 128;
    constexpr int k_chunks = 8;
    constexpr int fragments = 4;
    constexpr int halves = 2;
    const int shared_cells = tile * k_chunks;

    std::vector<int> a_source_row(shared_cells, -1);
    std::vector<int> a_source_k(shared_cells, -1);
    std::vector<int> a_owner(shared_cells, -1);
    std::vector<int> b_source_col(shared_cells, -1);
    std::vector<int> b_source_k(shared_cells, -1);
    std::vector<int> b_owner(shared_cells, -1);

    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        const int wave = thread_id / wave_size;
        const int lane = thread_id % wave_size;
        const int store_k = ((thread_id / 8) + (thread_id % 8)) % k_chunks;
        for (int load = 0; load < 4; ++load) {
            const int a_row = wave * 32 + lane / 8 + 8 * load;
            const int a_index = a_row * k_chunks + store_k;
            const int b_row = thread_id / 8 + 32 * load;
            const int b_index = b_row * k_chunks + store_k;
            if (a_row < 0 || a_row >= tile || b_row < 0 || b_row >= tile
                || a_source_row[a_index] != -1 || b_source_col[b_index] != -1) {
                throw std::runtime_error("MMA shared store mapping is not one-to-one");
            }
            a_source_row[a_index] = thread_id / 8 + 32 * load;
            a_source_k[a_index] = lane % 8;
            a_owner[a_index] = wave;
            b_source_col[b_index] = (thread_id / 8) * 4 + load;
            b_source_k[b_index] = lane % 8;
            b_owner[b_index] = wave;
        }
    }
    if (!std::all_of(a_source_row.begin(), a_source_row.end(), [](int value) {
            return value >= 0;
        })
        || !std::all_of(b_source_col.begin(), b_source_col.end(), [](int value) {
            return value >= 0;
        })) {
        throw std::runtime_error("MMA shared store mapping leaves a hole");
    }

    const int operand_visits_size = 4 * fragments * tile * k_chunks;
    std::vector<int> a_visits(operand_visits_size, 0);
    std::vector<int> b_visits(operand_visits_size, 0);
    int cross_wave_a = 0;
    int cross_wave_b = 0;
    for (int thread_id = 0; thread_id < threads; ++thread_id) {
        const int wave = thread_id / wave_size;
        const int lane = thread_id % wave_size;
        const int wave_m = wave / 2;
        const int wave_n = wave % 2;
        for (int half = 0; half < halves; ++half) {
            const int lds_k =
                ((thread_id % 16) + lane / 16 + 4 * half) % k_chunks;
            for (int mma_row = 0; mma_row < fragments; ++mma_row) {
                const int shared_row =
                    (thread_id % 16) + wave_m * 64 + 16 * mma_row;
                const int index = shared_row * k_chunks + lds_k;
                const int source_row = a_source_row[index];
                const int source_k = a_source_k[index];
                if (shared_row < 0 || shared_row >= tile
                    || (shared_row * 128 + lds_k * 16) % 16 != 0
                    || source_row < 0 || source_row >= tile
                    || source_k < 4 * half || source_k >= 4 * half + 4) {
                    throw std::runtime_error("MMA A LDS mapping is out of range or misaligned");
                }
                const int visit =
                    ((wave * fragments + mma_row) * tile + source_row) * k_chunks
                    + source_k;
                ++a_visits[visit];
                cross_wave_a += a_owner[index] != wave ? 1 : 0;
            }
            for (int mma_col = 0; mma_col < fragments; ++mma_col) {
                const int shared_row =
                    (thread_id % 16) + wave_n * 16 + 32 * mma_col;
                const int index = shared_row * k_chunks + lds_k;
                const int source_col = b_source_col[index];
                const int source_k = b_source_k[index];
                const int expected_col =
                    xh_fused_moe::mma_output_col_local(thread_id, mma_col);
                if (shared_row < 0 || shared_row >= tile
                    || (shared_row * 128 + lds_k * 16) % 16 != 0
                    || source_col != expected_col
                    || source_k < 4 * half || source_k >= 4 * half + 4) {
                    throw std::runtime_error("MMA B LDS fragment identity is inconsistent");
                }
                const int visit =
                    ((wave * fragments + mma_col) * tile + source_col) * k_chunks
                    + source_k;
                ++b_visits[visit];
                cross_wave_b += b_owner[index] != wave ? 1 : 0;
            }
        }
    }

    for (int wave = 0; wave < 4; ++wave) {
        for (int fragment = 0; fragment < fragments; ++fragment) {
            std::vector<int> expected_rows(tile, 0);
            std::vector<int> expected_cols(tile, 0);
            for (int lane = 0; lane < wave_size; ++lane) {
                const int thread_id = wave * wave_size + lane;
                for (int row_in_vector = 0; row_in_vector < 4; ++row_in_vector) {
                    expected_rows[xh_fused_moe::mma_output_row_local(
                        thread_id, fragment, row_in_vector)] = 1;
                }
                expected_cols[xh_fused_moe::mma_output_col_local(
                    thread_id, fragment)] = 1;
            }
            for (int coordinate = 0; coordinate < tile; ++coordinate) {
                for (int k_chunk = 0; k_chunk < k_chunks; ++k_chunk) {
                    const int a_visit =
                        ((wave * fragments + fragment) * tile + coordinate)
                        * k_chunks + k_chunk;
                    const int b_visit = a_visit;
                    if (a_visits[a_visit] != expected_rows[coordinate]
                        || b_visits[b_visit] != expected_cols[coordinate]) {
                        throw std::runtime_error(
                            "MMA operand fragments do not match their output coordinates");
                    }
                }
            }
        }
    }
    if (cross_wave_a != 1024 || cross_wave_b != 1536) {
        throw std::runtime_error("MMA cross-wave LDS ownership model changed");
    }
    std::cout << "REGRESSION maca-wave-2x2-fragments A/B-exact=PASS"
              << " aligned=PASS cross-wave-A=" << cross_wave_a
              << " cross-wave-B=" << cross_wave_b << "\n";
}

void verify_mma_wave_resource_and_sync_model() {
    constexpr int tile_m = 128;
    constexpr int tile_n = 128;
    constexpr int tile_k = 128;
    constexpr int threads = 256;
    constexpr int mma_rows = 4;
    constexpr int mma_cols = 4;
    constexpr int mma_depth = 8;
    constexpr int accumulator_vectors = mma_rows * mma_cols;
    constexpr int accumulator_ints = accumulator_vectors * 4;
    constexpr int a_fragment_ints = mma_rows * mma_depth;
    constexpr int b_fragment_ints = mma_cols * mma_depth;
    constexpr int lds_b128_per_tile = 2 * (mma_rows + mma_cols);
    constexpr int shared_bytes = tile_m * tile_k + tile_n * tile_k;
    constexpr int a_bytes_per_tile = 4 * 16 * threads;
    constexpr int b_bytes_per_tile = 4 * 16 * threads;
    constexpr int mma_per_tile = mma_rows * mma_cols * mma_depth;
    constexpr int output_bytes_per_tile = threads * accumulator_vectors * 8;

    if (accumulator_vectors != 16 || accumulator_ints != 64
        || a_fragment_ints != 32 || b_fragment_ints != 32
        || a_fragment_ints + b_fragment_ints != 64
        || lds_b128_per_tile != 16 || shared_bytes != 32 * 1024
        || a_bytes_per_tile != tile_m * tile_k
        || b_bytes_per_tile != tile_n * tile_k
        || mma_per_tile != 128
        || output_bytes_per_tile != tile_m * tile_n * 2) {
        throw std::runtime_error("MMA 2x2 resource or traffic model changed");
    }

    for (const PublicCase& public_case : kPublicCases) {
        const int k_tiles = public_case.config.k / tile_k;
        const int barriers = 1 + 2 * (k_tiles - 1);
        const int a_loads_per_thread = 4 * k_tiles;
        const int b_loads_per_thread = 4 * k_tiles;
        const int lds_per_thread = lds_b128_per_tile * k_tiles;
        const int mma_instructions = mma_per_tile * k_tiles;
        if ((public_case.config.k % tile_k) != 0
            || a_loads_per_thread * 16 * threads != tile_m * public_case.config.k
            || b_loads_per_thread * 16 * threads != tile_n * public_case.config.k
            || barriers != 1 + 2 * (public_case.config.k / tile_k - 1)
            || lds_per_thread != 16 * k_tiles
            || mma_instructions != 128 * k_tiles) {
            throw std::runtime_error(
                std::string("MMA 2x2 dynamic model failed for ") + public_case.name);
        }
        std::cout << "REGRESSION maca-wave-2x2-model case=" << public_case.name
                  << " LDS-b128/thread=" << lds_per_thread
                  << " MMA=" << mma_instructions << " barriers=" << barriers
                  << " PASS\n";
    }
}

void run_regression() {
    verify_public_inference();
    verify_mma_output_mapping();
    verify_mma_grid_mapping();
    verify_mma_a_load_bounds();
    verify_mma_wave_fragment_mapping();
    verify_mma_wave_resource_and_sync_model();
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
