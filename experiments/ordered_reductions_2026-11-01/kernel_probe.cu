#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <vector>

#include "cuda_kernels.h"
#include "semantic_check.h"
using json = nlohmann::json;
#ifndef NDEBUG
#error Release guard proof requires NDEBUG
#endif
extern __global__ void rmsnorm_optimized_det_kernel(const float *,
                                                    const float *, float *, int,
                                                    float);
#ifdef ORDERED_CANDIDATE
extern __global__ void softmax_optimized_det_kernel(const float *, float *,
                                                    int);
#endif
void ck(cudaError_t e) {
  if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
int main(int argc, char **argv) {
  try {
    setenv("EMBER_DECODE_ATTN", "legacy", 1);
    int count = 0;
    ck(cudaGetDeviceCount(&count));
    if (count < 1) throw std::runtime_error("actual CUDA required");
    cudaDeviceProp prop{};
    ck(cudaGetDeviceProperties(&prop, 0));
    if (std::string(prop.name).find("L4") == std::string::npos)
      throw std::runtime_error("L4 required");
    const int cap = 16384;
    float *x, *w, *y;
    ck(cudaMalloc(&x, cap * 4));
    ck(cudaMalloc(&w, cap * 4));
    ck(cudaMalloc(&y, cap * 4));
    if (argc > 1 && std::string(argv[1]) == "trap") {
#ifdef ORDERED_CANDIDATE
      if (argc != 4) throw std::runtime_error("trap kernel axis");
      std::vector<float> ones(cap, 1);
      ck(cudaMemcpy(x, ones.data(), cap * 4, cudaMemcpyHostToDevice));
      ck(cudaMemcpy(w, ones.data(), cap * 4, cudaMemcpyHostToDevice));
      dim3 grid(1, 1, 1);
      int axis = std::stoi(argv[3]);
      if (axis == 0)
        grid.x = 2;
      else if (axis == 1)
        grid.y = 2;
      else if (axis == 2)
        grid.z = 2;
      else
        throw std::runtime_error("axis");
      std::string kernel = argv[2];
      if (kernel == "rms")
        rmsnorm_optimized_det_kernel<<<grid, 256>>>(x, w, y, 320, 1e-5f);
      else if (kernel == "softmax")
        softmax_optimized_det_kernel<<<grid, 256>>>(x, y, 320);
      else
        throw std::runtime_error("kernel");
      auto launch = cudaGetLastError();
      auto result = cudaDeviceSynchronize();
      bool pass =
          launch == cudaSuccess && (result == cudaErrorIllegalInstruction ||
                                    result == cudaErrorLaunchFailure);
      std::cout << json{{"stage", "actual_guarded_kernel"},
                        {"kernel", kernel},
                        {"axis", axis},
                        {"allocation_valid", true},
                        {"launch", int(launch)},
                        {"sync", int(result)},
                        {"pass", pass}}
                << std::endl;
      return pass ? 0 : 2;
#else
      throw std::runtime_error("baseline trap not registered");
#endif
    }
    cudaStream_t stream;
    ck(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    json records = json::array();
    bool all = true;
    for (int n : {1, 31, 257, 320, 1024, 16384})
      for (int pattern = 0; pattern < 3; ++pattern)
        for (int kind = 0; kind < 2; ++kind) {
          std::vector<float> a(n), b(n), out(n), first(n);
          std::vector<double> reference(n);
          for (int i = 0; i < n; ++i) {
            a[i] = pattern == 0   ? float(std::sin(i * .731) * 3)
                   : pattern == 1 ? 2.0f
                                  : float((i % 17) - 8) * 10;
            b[i] = float(.5 + (i % 11) * .071);
          }
          ck(cudaMemcpy(x, a.data(), n * 4, cudaMemcpyHostToDevice));
          ck(cudaMemcpy(w, b.data(), n * 4, cudaMemcpyHostToDevice));
          auto forward = [&]() {
            if (kind == 0)
              rmsnorm_vector_cuda_optimized(x, w, y, n, 1e-5f, stream);
            else
              softmax_vector_cuda_optimized(x, y, n, stream);
          };
          if (kind == 0) {
            double sum = 0;
            for (float v : a) sum += double(v) * v;
            for (int i = 0; i < n; ++i)
              reference[i] =
                  double(a[i]) * b[i] / std::sqrt(sum / n + double(1e-5f));
          } else {
            double m = *std::max_element(a.begin(), a.end()), sum = 0;
            for (float v : a) sum += std::exp(double(v) - m);
            for (int i = 0; i < n; ++i)
              reference[i] = std::exp(double(a[i]) - m) / sum;
          }
          bool repeat = true;
          double error = 0;
          for (int r = 0; r < 20; ++r) {
            forward();
            ck(cudaDeviceSynchronize());
            ck(cudaMemcpy(out.data(), y, n * 4, cudaMemcpyDeviceToHost));
            if (r == 0)
              first = out;
            else
              repeat &= std::memcmp(first.data(), out.data(), n * 4) == 0;
            for (int i = 0; i < n; ++i)
              error = std::max(error, checked_error(out[i], reference[i]));
          }
          cudaGraph_t graph;
          cudaGraphExec_t exec;
          ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
          for (int r = 0; r < 100; ++r) forward();
          ck(cudaStreamEndCapture(stream, &graph));
          ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
          for (int r = 0; r < 5; ++r) ck(cudaGraphLaunch(exec, stream));
          ck(cudaDeviceSynchronize());
          cudaEvent_t start, stop;
          ck(cudaEventCreate(&start));
          ck(cudaEventCreate(&stop));
          std::vector<float> ms;
          for (int r = 0; r < 30; ++r) {
            ck(cudaEventRecord(start, stream));
            ck(cudaGraphLaunch(exec, stream));
            ck(cudaEventRecord(stop, stream));
            ck(cudaEventSynchronize(stop));
            float t;
            ck(cudaEventElapsedTime(&t, start, stop));
            ms.push_back(t / 100);
          }
          bool pass = std::isfinite(error) && error <= 1e-5;
#ifdef ORDERED_CANDIDATE
          pass &= repeat;
#endif
          all &= pass;
          records.push_back(json{{"kernel", kind == 0 ? "rms" : "softmax"},
                                 {"n", n},
                                 {"pattern", pattern},
                                 {"input", a},
                                 {"weights", b},
                                 {"first_output", first},
                                 {"reference", reference},
                                 {"error", error},
                                 {"repeat_equal", repeat},
                                 {"milliseconds", ms},
                                 {"pass", pass}});
          ck(cudaEventDestroy(start));
          ck(cudaEventDestroy(stop));
          ck(cudaGraphExecDestroy(exec));
          ck(cudaGraphDestroy(graph));
        }

    // Exercise both actual fallback wrappers against a binary64 causal GQA
    // reference.
    float *q, *k, *v, *att;
    int *step;
    ck(cudaMalloc(&q, 128 * 4));
    ck(cudaMalloc(&k, 512 * 64 * 4));
    ck(cudaMalloc(&v, 512 * 64 * 4));
    ck(cudaMalloc(&att, 128 * 4));
    ck(cudaMalloc(&step, 2 * sizeof(int)));
    for (int length : {1, 31, 257, 512})
      for (int devpos = 0; devpos < 2; ++devpos) {
        std::vector<float> Q(128), K(length * 64), V(length * 64), output(128),
            first(128);
        std::vector<double> ref(128);
        for (int i = 0; i < 128; ++i) Q[i] = float(std::sin(i * .17));
        for (int i = 0; i < length * 64; ++i) {
          K[i] = float(std::cos(i * .031));
          V[i] = float(std::sin(i * .071));
        }
        ck(cudaMemcpy(q, Q.data(), 128 * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(k, K.data(), K.size() * 4, cudaMemcpyHostToDevice));
        ck(cudaMemcpy(v, V.data(), V.size() * 4, cudaMemcpyHostToDevice));
        int st[2] = {0, length - 1};
        ck(cudaMemcpy(step, st, sizeof(st), cudaMemcpyHostToDevice));
        float scale = 1 / std::sqrt(32.0f);
        for (int head = 0; head < 4; ++head) {
          std::vector<double> scores(length);
          double maximum = -INFINITY, total = 0;
          for (int t = 0; t < length; ++t) {
            double dot = 0;
            for (int d = 0; d < 32; ++d)
              dot += double(Q[head * 32 + d]) * K[t * 64 + (head / 2) * 32 + d];
            scores[t] = dot * double(scale);
            maximum = std::max(maximum, scores[t]);
          }
          for (double score : scores) total += std::exp(score - maximum);
          for (int d = 0; d < 32; ++d) {
            double sum = 0;
            for (int t = 0; t < length; ++t)
              sum += std::exp(scores[t] - maximum) / total *
                     V[t * 64 + (head / 2) * 32 + d];
            ref[head * 32 + d] = sum;
          }
        }
        double error = 0;
        bool repeat = true;
        for (int r = 0; r < 20; ++r) {
          if (devpos)
            attention_devpos_cuda(q, k, v, att, 4, 32, scale, 512, 2, step,
                                  stream);
          else
            attention_cuda_optimized(q, k, v, att, 4, length, 32, scale, 512, 2,
                                     stream);
          ck(cudaDeviceSynchronize());
          ck(cudaMemcpy(output.data(), att, 128 * 4, cudaMemcpyDeviceToHost));
          if (r == 0)
            first = output;
          else
            repeat &= std::memcmp(first.data(), output.data(), 128 * 4) == 0;
          for (int i = 0; i < 128; ++i)
            error = std::max(error, checked_error(output[i], ref[i]));
        }
        bool pass = error <= 1e-5;
#ifdef ORDERED_CANDIDATE
        pass &= repeat;
#endif
        cudaEvent_t start, stop;
        ck(cudaEventCreate(&start));
        ck(cudaEventCreate(&stop));
        std::vector<float> ms;
        for (int batch = 0; batch < 30; ++batch) {
          ck(cudaEventRecord(start, stream));
          for (int repeat = 0; repeat < 100; ++repeat) {
            if (devpos)
              attention_devpos_cuda(q, k, v, att, 4, 32, scale, 512, 2, step,
                                    stream);
            else
              attention_cuda_optimized(q, k, v, att, 4, length, 32, scale, 512,
                                       2, stream);
          }
          ck(cudaEventRecord(stop, stream));
          ck(cudaEventSynchronize(stop));
          float elapsed;
          ck(cudaEventElapsedTime(&elapsed, start, stop));
          ms.push_back(elapsed / 100);
        }
        ck(cudaEventDestroy(start));
        ck(cudaEventDestroy(stop));
        all &= pass;
        records.push_back(json{
            {"kernel", devpos ? "attention_devpos_legacy" : "attention_legacy"},
            {"n", length},
            {"milliseconds", ms},
            {"input_q", Q},
            {"input_k", K},
            {"input_v", V},
            {"scale", scale},
            {"first_output", first},
            {"reference", ref},
            {"error", error},
            {"repeat_equal", repeat},
            {"pass", pass}});
      }
    ck(cudaFree(q));
    ck(cudaFree(k));
    ck(cudaFree(v));
    ck(cudaFree(att));
    ck(cudaFree(step));
    std::cout << json{{"device", prop.name},
                      {"records", records},
                      {"pass", all}}
              << std::endl;
    ck(cudaFree(x));
    ck(cudaFree(w));
    ck(cudaFree(y));
    return all ? 0 : 1;
  } catch (const std::exception &e) {
    std::cerr << "PROBE_ERROR: " << e.what() << std::endl;
    return 2;
  }
}
