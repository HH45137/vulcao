#include "vulcao/query_pool.h"

#include <utility>

namespace vulcao {

QueryPool::~QueryPool() {
    destroy();
}

QueryPool::QueryPool(QueryPool&& other) noexcept
    : device_(std::exchange(other.device_, vk::Device{})),
      pool_(std::exchange(other.pool_, vk::QueryPool{})),
      query_count_(std::exchange(other.query_count_, 0)),
      type_(std::exchange(other.type_, vk::QueryType::eOcclusion)) {}

QueryPool& QueryPool::operator=(QueryPool&& other) noexcept {
    if (this != &other) {
        destroy();
        device_ = std::exchange(other.device_, vk::Device{});
        pool_ = std::exchange(other.pool_, vk::QueryPool{});
        query_count_ = std::exchange(other.query_count_, 0);
        type_ = std::exchange(other.type_, vk::QueryType::eOcclusion);
    }
    return *this;
}

QueryPool QueryPool::create(vk::Device device,
                            vk::QueryType type,
                            uint32_t query_count,
                            vk::QueryPoolCreateFlags flags) {
    QueryPool query_pool;
    query_pool.device_ = device;
    query_pool.pool_ = device.createQueryPool(vk::QueryPoolCreateInfo{
        .flags = flags,
        .queryType = type,
        .queryCount = query_count,
    });
    query_pool.query_count_ = query_count;
    query_pool.type_ = type;
    return query_pool;
}

void QueryPool::destroy() {
    if (pool_)
        device_.destroyQueryPool(pool_);

    device_ = nullptr;
    pool_ = nullptr;
    query_count_ = 0;
    type_ = vk::QueryType::eOcclusion;
}

}
