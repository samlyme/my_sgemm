#include "sgemm.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
  do {                                                                          \
    const cudaError_t error = (call);                                            \
    if (error != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string(#call) + ": " +                      \
                               cudaGetErrorString(error));                       \
    }                                                                           \
  } while (false)

namespace {

struct Options {
  int m = 1024;
  int n = 1024;
  int k = 1024;
  int iterations = 100;
  SgemmKernel kernel = SgemmKernel::Naive;
};

void print_usage(const char* program) {
  std::cout << "Usage: " << program
            << " [--kernel naive] [--m M] [--n N] [--k K] [--iters N]\n";
}

int positive_int(const char* value, const char* option) {
  const long parsed = std::stol(value);
  if (parsed <= 0 || parsed > 2147483647L) {
    throw std::invalid_argument(std::string(option) + " must be a positive integer");
  }
  return static_cast<int>(parsed);
}

Options parse_options(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string arg(argv[index]);
    if (arg == "--help" || arg == "-h") {
      print_usage(argv[0]);
      std::exit(0);
    }
    if (index + 1 >= argc) {
      throw std::invalid_argument("missing value for " + arg);
    }
    const char* value = argv[++index];
    if (arg == "--kernel") {
      options.kernel = parse_kernel(value);
    } else if (arg == "--m") {
      options.m = positive_int(value, "--m");
    } else if (arg == "--n") {
      options.n = positive_int(value, "--n");
    } else if (arg == "--k") {
      options.k = positive_int(value, "--k");
    } else if (arg == "--iters") {
      options.iterations = positive_int(value, "--iters");
    } else {
      throw std::invalid_argument("unknown option " + arg);
    }
  }
  return options;
}

void check_result(const std::vector<float>& a,
                  const std::vector<float>& b,
                  const std::vector<float>& actual,
                  int m,
                  int n,
                  int k) {
  constexpr std::size_t max_samples = 256;
  const std::size_t output_elements = static_cast<std::size_t>(m) * n;
  const std::size_t samples = std::min(max_samples, output_elements);
  float max_absolute_error = 0.0f;
  float max_relative_error = 0.0f;

  // Sample across C so correctness checking remains quick for large benchmarks.
  for (std::size_t sample = 0; sample < samples; ++sample) {
    const std::size_t index = output_elements <= max_samples
                                  ? sample
                                  : ((sample * 104729) % m) * static_cast<std::size_t>(n) +
                                        (sample * 130363) % n;
    const int row = static_cast<int>(index / n);
    const int col = static_cast<int>(index % n);
    double reference = 0.0;
    for (int inner = 0; inner < k; ++inner) {
      reference += static_cast<double>(a[row * k + inner]) * b[inner * n + col];
    }
    const float expected = static_cast<float>(reference);
    const float absolute_error = std::abs(expected - actual[index]);
    const float relative_error = absolute_error / std::max(std::abs(expected), 1.0e-6f);
    max_absolute_error = std::max(max_absolute_error, absolute_error);
    max_relative_error = std::max(max_relative_error, relative_error);

    const float tolerance = 1.0e-3f + 2.0e-5f * k * std::abs(expected);
    if (absolute_error > tolerance) {
      throw std::runtime_error("result mismatch at element " + std::to_string(index) +
                               ": expected " + std::to_string(expected) +
                               ", got " + std::to_string(actual[index]));
    }
  }
  std::cout << "check: PASS (" << samples << " samples, max abs error "
            << max_absolute_error << ", max rel error " << max_relative_error << ")\n";
}

}  // namespace

int main(int argc, char** argv) {
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  try {
    const Options options = parse_options(argc, argv);
    const std::size_t a_elements = static_cast<std::size_t>(options.m) * options.k;
    const std::size_t b_elements = static_cast<std::size_t>(options.k) * options.n;
    const std::size_t c_elements = static_cast<std::size_t>(options.m) * options.n;

    std::vector<float> a(a_elements);
    std::vector<float> b(b_elements);
    std::vector<float> actual(c_elements, 0.0f);

    std::mt19937 generator(0);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    std::generate(a.begin(), a.end(), [&] { return distribution(generator); });
    std::generate(b.begin(), b.end(), [&] { return distribution(generator); });

    std::cout << "kernel: " << kernel_name(options.kernel) << '\n'
              << "shape:  " << options.m << " x " << options.n << " x " << options.k << '\n';

    CUDA_CHECK(cudaMalloc(&device_a, a_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_b, b_elements * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_c, c_elements * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(device_a, a.data(), a_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(device_b, b.data(), b_elements * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_c, 0, c_elements * sizeof(float)));

    launch_sgemm(options.kernel, options.m, options.n, options.k, 1.0f,
                 device_a, device_b, 0.0f, device_c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(actual.data(), device_c, c_elements * sizeof(float), cudaMemcpyDeviceToHost));
    check_result(a, b, actual, options.m, options.n, options.k);

    for (int warmup = 0; warmup < 5; ++warmup) {
      launch_sgemm(options.kernel, options.m, options.n, options.k, 1.0f,
                   device_a, device_b, 0.0f, device_c);
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      launch_sgemm(options.kernel, options.m, options.n, options.k, 1.0f,
                   device_a, device_b, 0.0f, device_c);
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    const double average_ms = elapsed_ms / options.iterations;
    const double operations = 2.0 * options.m * options.n * options.k;
    const double gflops = operations / (average_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(3)
              << "time:   " << average_ms << " ms\n"
              << "perf:   " << gflops << " GFLOP/s\n";

    CUDA_CHECK(cudaEventDestroy(start));
    start = nullptr;
    CUDA_CHECK(cudaEventDestroy(stop));
    stop = nullptr;
    CUDA_CHECK(cudaFree(device_a));
    device_a = nullptr;
    CUDA_CHECK(cudaFree(device_b));
    device_b = nullptr;
    CUDA_CHECK(cudaFree(device_c));
    device_c = nullptr;
    return 0;
  } catch (const std::exception& error) {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
    if (device_a != nullptr) cudaFree(device_a);
    if (device_b != nullptr) cudaFree(device_b);
    if (device_c != nullptr) cudaFree(device_c);
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
