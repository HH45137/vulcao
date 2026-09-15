#pragma once

#include <cstdint>

#include <vulkan/vulkan.hpp>

namespace vulcao {

/// @brief RAII wrapper around a Vulkan query pool.
class QueryPool {
public:
    /// @brief Creates an empty query pool.
    QueryPool() = default;

    /// @brief Destroys the query pool.
    ~QueryPool();

    /// @brief Not copyable.
    QueryPool(const QueryPool&) = delete;
    QueryPool& operator=(const QueryPool&) = delete;

    /// @brief Moves the query pool, leaving the source empty.
    QueryPool(QueryPool&& other) noexcept;

    /// @brief Move assignment. Destroys the current query pool first.
    QueryPool& operator=(QueryPool&& other) noexcept;

    /// @brief Creates a query pool.
    /// @param device Device that creates the pool.
    /// @param type Type of the queries.
    /// @param query_count Number of queries in the pool.
    /// @param flags Query pool creation flags.
    /// @return The created query pool.
    static QueryPool create(vk::Device device,
                            vk::QueryType type,
                            uint32_t query_count,
                            vk::QueryPoolCreateFlags flags = {});

    /// @brief Returns true if the query pool holds a valid handle.
    bool valid() const { return static_cast<bool>(pool_); }

    /// @brief Returns true if the query pool holds a valid handle.
    explicit operator bool() const { return valid(); }

    /// @brief Returns the raw Vulkan query pool handle.
    vk::QueryPool handle() const { return pool_; }

    /// @brief Returns the number of queries in the pool.
    uint32_t query_count() const { return query_count_; }

    /// @brief Returns the type of the queries.
    vk::QueryType type() const { return type_; }

    /// @brief Destroys the query pool and resets the wrapper.
    void destroy();

private:
    vk::Device device_;
    vk::QueryPool pool_;
    uint32_t query_count_ = 0;
    vk::QueryType type_ = vk::QueryType::eOcclusion;
};

}
