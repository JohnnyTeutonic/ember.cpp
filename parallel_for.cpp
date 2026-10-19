#include "parallel_for.h"

#if defined(__EMSCRIPTEN_PTHREADS__)
#include <emscripten/threading.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#ifndef EMBER_WASM_THREADS
#define EMBER_WASM_THREADS 4
#endif

namespace ember {
namespace detail {
namespace {

// Workers wait for a job generation, take chunks from a shared counter, and
// report back; the caller takes chunks too. One job at a time (the caller
// holds run_mu for the job's duration).
class Pool {
 public:
  Pool() {
    const int cores = std::max(1, emscripten_num_logical_cores());
    const int n = std::max(0, std::min(EMBER_WASM_THREADS, cores) - 1);  // the caller is a worker too
    for (int i = 0; i < n; ++i) threads_.emplace_back([this] { loop(); });
  }

  void run(int64_t n, const std::function<void(int64_t, int64_t)>& body) {
    std::lock_guard<std::mutex> job(run_mu_);
    const int64_t parts = static_cast<int64_t>(threads_.size()) + 1;
    const int64_t chunk = (n + parts - 1) / parts;
    {
      std::lock_guard<std::mutex> lk(mu_);
      body_ = &body;
      n_ = n;
      chunk_ = chunk;
      next_.store(0);
      busy_ = static_cast<int>(threads_.size());
      ++generation_;
    }
    cv_.notify_all();
    work();
    std::unique_lock<std::mutex> lk(mu_);
    done_cv_.wait(lk, [this] { return busy_ == 0; });
    body_ = nullptr;
  }

 private:
  void work() {
    for (;;) {
      const int64_t lo = next_.fetch_add(chunk_);
      if (lo >= n_) return;
      (*body_)(lo, std::min(n_, lo + chunk_));
    }
  }

  void loop() {
    uint64_t seen = 0;
    for (;;) {
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [&] { return generation_ != seen; });
        seen = generation_;
      }
      work();
      std::lock_guard<std::mutex> lk(mu_);
      if (--busy_ == 0) done_cv_.notify_one();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex run_mu_, mu_;
  std::condition_variable cv_, done_cv_;
  const std::function<void(int64_t, int64_t)>* body_ = nullptr;
  int64_t n_ = 0, chunk_ = 1;
  std::atomic<int64_t> next_{0};
  int busy_ = 0;
  uint64_t generation_ = 0;
};

}  // namespace

void pool_run(int64_t n, const std::function<void(int64_t, int64_t)>& body) {
  static Pool pool;  // threads live for the module's lifetime
  pool.run(n, body);
}

}  // namespace detail
}  // namespace ember
#endif
