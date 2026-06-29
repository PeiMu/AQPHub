#pragma once

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace middleware {

// Global count of active background tasks across all ThreadPool instances.
// Read by OpenMP regions to reduce thread count and avoid oversubscription.
extern std::atomic<int> g_bg_active_threads;

class ThreadPool {
public:
  explicit ThreadPool(size_t num_threads = 1) : stop_(false) {
    workers_.reserve(num_threads);
    for (size_t i = 0; i < num_threads; i++)
      workers_.emplace_back([this] { WorkerLoop(); });
  }

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lk(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &w : workers_)
      w.join();
  }

  ThreadPool(const ThreadPool &) = delete;
  ThreadPool &operator=(const ThreadPool &) = delete;

  template <typename F, typename... Args>
  auto Submit(F &&f, Args &&...args)
      -> std::future<typename std::result_of<F(Args...)>::type> {
    using R = typename std::result_of<F(Args...)>::type;
    auto task = std::make_shared<std::packaged_task<R()>>(
        std::bind(std::forward<F>(f), std::forward<Args>(args)...));
    auto fut = task->get_future();
    {
      std::lock_guard<std::mutex> lk(mu_);
      if (stop_)
        throw std::runtime_error("Submit on stopped ThreadPool");
      tasks_.push([task]() { (*task)(); });
    }
    cv_.notify_one();
    return fut;
  }

  void DrainAll() {
    std::unique_lock<std::mutex> lk(mu_);
    drain_cv_.wait(lk, [this] { return tasks_.empty() && active_ == 0; });
  }

  size_t NumThreads() const { return workers_.size(); }

private:
  void WorkerLoop() {
    for (;;) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lk(mu_);
        cv_.wait(lk, [this] { return stop_ || !tasks_.empty(); });
        if (stop_ && tasks_.empty())
          return;
        task = std::move(tasks_.front());
        tasks_.pop();
        active_++;
      }
      g_bg_active_threads.fetch_add(1, std::memory_order_relaxed);
      task();
      g_bg_active_threads.fetch_sub(1, std::memory_order_relaxed);
      {
        std::lock_guard<std::mutex> lk(mu_);
        active_--;
      }
      drain_cv_.notify_all();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mu_;
  std::condition_variable cv_;
  std::condition_variable drain_cv_;
  std::queue<std::function<void()>> tasks_;
  bool stop_;
  int active_ = 0;
};

} // namespace middleware
