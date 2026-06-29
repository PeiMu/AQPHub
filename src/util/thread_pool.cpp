#include "util/thread_pool.h"

namespace middleware {

std::atomic<int> g_bg_active_threads{0};

} // namespace middleware
