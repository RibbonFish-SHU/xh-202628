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

struct K64VectorIdentity {
    int row;
    int k_chunk;
};

void verify_mma_k64_fragment_mapping() {
    constexpr int threads = 256;
    constexpr int wave_size = 64;
    constexpr int tile_rows = 128;
    constexpr int k_chunks = 4;
    constexpr int loads_per_thread = 2;
    constexpr int vectors = tile_rows * k_chunks;
    const K64VectorIdentity unset{-1, -1};
    std::vector<K64VectorIdentity> shared_a(vectors, unset);
    std::vector<K64VectorIdentity> shared_b(vectors, unset);
    std::vector<int> global_a_visits(vectors, 0);
    std::vector<int> global_b_visits(vectors, 0);
    std::vector<int> shared_a_visits(vectors, 0);
    std::vector<int> shared_b_visits(vectors, 0);

    for (int tid = 0; tid < threads; ++tid) {
        const int wave = tid / wave_size;
        const int lane = tid % wave_size;
        const int global_k_chunk = lane % k_chunks;
        const int shared_k_chunk = ((tid / 4) + (tid % 4)) % k_chunks;
        for (int load = 0; load < loads_per_thread; ++load) {
            const int a_slot = lane / 4 + 16 * load;
            const int global_a_row =
                wave * 8 + (a_slot % 8) + 32 * (a_slot / 8);
            const int shared_a_row = wave * 32 + a_slot;
            const int global_a_index = global_a_row * k_chunks + global_k_chunk;
            const int shared_a_index = shared_a_row * k_chunks + shared_k_chunk;
            shared_a[shared_a_index] = {global_a_row, global_k_chunk};
            ++global_a_visits[global_a_index];
            ++shared_a_visits[shared_a_index];

            const int b_slot = tid / 4;
            const int global_b_row =
                4 * (b_slot % 32) + b_slot / 32 + 2 * load;
            const int shared_b_row = b_slot + 64 * load;
            const int global_b_index = global_b_row * k_chunks + global_k_chunk;
            const int shared_b_index = shared_b_row * k_chunks + shared_k_chunk;
            shared_b[shared_b_index] = {global_b_row, global_k_chunk};
            ++global_b_visits[global_b_index];
            ++shared_b_visits[shared_b_index];
        }
    }

    auto exact_cover = [](const std::vector<int>& visits) {
        return std::all_of(visits.begin(), visits.end(), [](int count) { return count == 1; });
    };
    if (!exact_cover(global_a_visits) || !exact_cover(global_b_visits)
        || !exact_cover(shared_a_visits) || !exact_cover(shared_b_visits)) {
        throw std::runtime_error("K64 producer mapping is not a vector bijection");
    }

    int checked_a_words = 0;
    int checked_b_words = 0;
    for (int tid = 0; tid < threads; ++tid) {
        const int wave = tid / wave_size;
        const int lane = tid % wave_size;
        const int lds_k_chunk = ((tid % 16) + lane / 16) % k_chunks;
        for (int row_fragment = 0; row_fragment < 2; ++row_fragment) {
            const int shared_row = (tid % 16) + wave * 32 + 16 * row_fragment;
            const K64VectorIdentity actual =
                shared_a[shared_row * k_chunks + lds_k_chunk];
            const int expected_row =
                wave * 8 + (shared_row % 8) + 32 * ((shared_row % 32) / 8);
            if (actual.row != expected_row || actual.k_chunk != lane / 16) {
                throw std::runtime_error("K64 A fragment identity mismatch");
            }
            for (int depth = 0; depth < 4; ++depth) {
                const int actual_k = actual.k_chunk * 16 + depth * 4;
                const int expected_k = (lane / 16) * 16 + depth * 4;
                if (actual_k != expected_k) {
                    throw std::runtime_error("K64 A fragment K-word mismatch");
                }
                ++checked_a_words;
            }
        }
        for (int col_fragment = 0; col_fragment < 8; ++col_fragment) {
            const int shared_row = (tid % 16) + 16 * col_fragment;
            const K64VectorIdentity actual =
                shared_b[shared_row * k_chunks + lds_k_chunk];
            const int expected_row = 4 * (shared_row % 32) + shared_row / 32;
            if (actual.row != expected_row || actual.k_chunk != lane / 16) {
                throw std::runtime_error("K64 B fragment identity mismatch");
            }
            for (int depth = 0; depth < 4; ++depth) {
                const int actual_k = actual.k_chunk * 16 + depth * 4;
                const int expected_k = (lane / 16) * 16 + depth * 4;
                if (actual_k != expected_k) {
                    throw std::runtime_error("K64 B fragment K-word mismatch");
                }
                ++checked_b_words;
            }
        }
    }
    if (checked_a_words != 2048 || checked_b_words != 8192) {
        throw std::runtime_error("K64 fragment cardinality mismatch");
    }
    std::cout << "REGRESSION maca-k64-fragment-mapping producer-vectors=1024"
              << " A-words=" << checked_a_words << " B-words=" << checked_b_words
              << " exact-cover=PASS\n";
}

void verify_mma_k64_pipeline_lifecycle() {
    constexpr int threads = 256;
    constexpr int a_rows = 2;
    constexpr int b_cols = 8;
    constexpr int depth = 4;
    const int tile_counts[] = {1, 2, 32, 112};

    for (int tiles : tile_counts) {
        std::vector<int> a_frag_tile(threads * a_rows * depth, tiles - 1);
        std::vector<int> b_frag_tile(threads * b_cols * depth, tiles - 1);
        std::vector<int> a_uses(tiles * threads * a_rows * depth, 0);
        std::vector<int> b_uses(tiles * threads * b_cols * depth, 0);
        std::vector<int> tile_order;
        int shared_tile = tiles - 1;
        int barriers = 1;
        int ldg_a_vectors = 2;
        int ldg_b_vectors = 2;
        int sts_a_vectors = 2;
        int sts_b_vectors = 2;
        int lds_a_vectors = 2;
        int lds_b_vectors = 8;
        int mma_instructions = 0;

        auto consume = [&](int tile) {
            for (int tid = 0; tid < threads; ++tid) {
                for (int row = 0; row < a_rows; ++row) {
                    for (int col = 0; col < b_cols; ++col) {
                        for (int k_depth = 0; k_depth < depth; ++k_depth) {
                            const int a_fragment = (tid * a_rows + row) * depth + k_depth;
                            const int b_fragment = (tid * b_cols + col) * depth + k_depth;
                            if (a_frag_tile[a_fragment] != tile
                                || b_frag_tile[b_fragment] != tile) {
                                throw std::runtime_error("K64 fragment used after overwrite");
                            }
                            const int a_key =
                                ((tile * threads + tid) * a_rows + row) * depth + k_depth;
                            const int b_key =
                                ((tile * threads + tid) * b_cols + col) * depth + k_depth;
                            ++a_uses[a_key];
                            ++b_uses[b_key];
                            ++mma_instructions;
                        }
                    }
                }
            }
            tile_order.push_back(tile);
        };

        for (int next_tile = 0; next_tile < tiles - 1; ++next_tile) {
            ++barriers;
            ldg_a_vectors += 2;
            ldg_b_vectors += 2;
            consume(shared_tile);
            shared_tile = next_tile;
            sts_a_vectors += 2;
            sts_b_vectors += 2;
            ++barriers;
            std::fill(a_frag_tile.begin(), a_frag_tile.end(), shared_tile);
            std::fill(b_frag_tile.begin(), b_frag_tile.end(), shared_tile);
            lds_a_vectors += 2;
            lds_b_vectors += 8;
        }
        consume(shared_tile);

        std::vector<int> expected_order;
        expected_order.push_back(tiles - 1);
        for (int tile = 0; tile < tiles - 1; ++tile) {
            expected_order.push_back(tile);
        }
        if (tile_order != expected_order || barriers != 2 * tiles - 1
            || ldg_a_vectors != 2 * tiles || ldg_b_vectors != 2 * tiles
            || sts_a_vectors != 2 * tiles || sts_b_vectors != 2 * tiles
            || lds_a_vectors != 2 * tiles || lds_b_vectors != 8 * tiles
            || mma_instructions != threads * 64 * tiles
            || !std::all_of(a_uses.begin(), a_uses.end(), [](int count) { return count == 8; })
            || !std::all_of(b_uses.begin(), b_uses.end(), [](int count) { return count == 2; })) {
            throw std::runtime_error("K64 pipeline lifecycle or traffic mismatch");
        }
        std::cout << "REGRESSION maca-k64-pipeline tiles=" << tiles
                  << " barriers=" << barriers << " A-LDG/STS/LDS="
                  << ldg_a_vectors << "/" << sts_a_vectors << "/" << lds_a_vectors
                  << " B-LDG/STS/LDS=" << ldg_b_vectors << "/" << sts_b_vectors
                  << "/" << lds_b_vectors << " MMA/thread="
                  << mma_instructions / threads
                  << " lifecycle=PASS\n";
    }
}

void verify_mma_k64_public_traffic() {
    constexpr int threads = 256;
    constexpr int vector_bytes = 16;
    for (const PublicCase& public_case : kPublicCases) {
        const int k64_tiles = public_case.config.k / 64;
        const int k128_tiles = public_case.config.k / 128;
        const int k64_a_bytes = k64_tiles * threads * 2 * vector_bytes;
        const int k64_b_bytes = k64_tiles * threads * 2 * vector_bytes;
        const int baseline_a_bytes = k128_tiles * threads * 4 * vector_bytes;
        const int baseline_b_bytes = k128_tiles * threads * 4 * vector_bytes;
        const int k64_lds_bytes =
            k64_tiles * threads * (2 + 8) * vector_bytes;
        const int baseline_lds_bytes =
            k128_tiles * threads * (4 + 16) * vector_bytes;
        if (k64_a_bytes != baseline_a_bytes || k64_b_bytes != baseline_b_bytes
            || k64_lds_bytes != baseline_lds_bytes
            || 64 * k64_tiles != 128 * k128_tiles) {
            throw std::runtime_error("K64 public traffic does not match K128 baseline");
        }
        std::cout << "REGRESSION maca-k64-traffic case=" << public_case.name
                  << " shared-bytes=16384 K64-tiles=" << k64_tiles
                  << " barriers=" << 2 * k64_tiles - 1
                  << " baseline-barriers=" << 2 * k128_tiles - 1
                  << " traffic-and-mma=UNCHANGED\n";
    }
}

void verify_mma_a_load_bounds() {
    constexpr int tile_rows = 128;
    constexpr int loads_per_thread = 2;
    for (const PublicCase& public_case : kPublicCases) {
        if ((public_case.config.em % tile_rows) != 0
            || (public_case.config.k % 64) != 0) {
            throw std::runtime_error(
                std::string("public shape is not exact-tile aligned for ") + public_case.name);
        }
        for (int thread_id = 0; thread_id < 256; ++thread_id) {
            const int wave = thread_id / 64;
            const int lane = thread_id % 64;
            for (int load = 0; load < loads_per_thread; ++load) {
                const int slot = lane / 4 + 16 * load;
                const int local_row = wave * 8 + (slot % 8) + 32 * (slot / 8);
                if (local_row < 0 || local_row >= tile_rows) {
                    throw std::runtime_error("MMA A load row is outside its exact tile");
                }
            }
        }
        const int predicates_removed_per_thread =
            loads_per_thread * (public_case.config.k / 64);
        std::cout << "REGRESSION maca-a-load-bounds case=" << public_case.name
                  << " exact-rows=PASS predicates-removed-per-thread="
                  << predicates_removed_per_thread << "\n";
    }
}

void run_regression() {
    verify_public_inference();
    verify_mma_output_mapping();
    verify_mma_grid_mapping();
    verify_mma_k64_fragment_mapping();
    verify_mma_k64_pipeline_lifecycle();
    verify_mma_k64_public_traffic();
    verify_mma_a_load_bounds();
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
