#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace vulcao {

/// @brief Defers destruction of move-only resources until flush() is called.
///
/// Owns type-erased deleters for resources that must outlive the GPU work
/// referencing them. FrameManager keeps one queue per frame in flight and
/// flushes it once the slot's fence signals; used standalone it simply delays
/// destruction until the queue is flushed or destroyed.
class DeletionQueue {
public:
    /// @brief Flushes remaining deleters on destruction.
    ~DeletionQueue() { flush(); }

    DeletionQueue() = default;

    /// @brief Not copyable.
    DeletionQueue(const DeletionQueue&) = delete;
    DeletionQueue& operator=(const DeletionQueue&) = delete;

    /// @brief Movable; the moved-from queue is left empty.
    DeletionQueue(DeletionQueue&&) = default;
    DeletionQueue& operator=(DeletionQueue&&) = default;

    /// @brief Queues a resource for destruction at the next flush().
    ///
    /// The resource is moved into the queue and assigned an empty object when
    /// flushed, which destroys whatever it owned. Works with every vulcao
    /// resource type. Non-invocable types only: pass callables through the
    /// std::function overload so they are run rather than destroyed.
    /// @tparam T Move-only resource type.
    /// @param resource Resource to destroy later.
    template <typename T>
        requires(!std::invocable<T>)
    void push(T&& resource) {
        entries_.push_back(
            std::make_unique<ResourceEntry<std::decay_t<T>>>(std::forward<T>(resource)));
    }

    /// @brief Queues a raw callable, invoked once at the next flush().
    /// @param fn Callable to run.
    void push(std::function<void()> fn) {
        entries_.push_back(std::make_unique<CallableEntry>(std::move(fn)));
    }

    /// @brief Runs every queued deleter and clears the queue. Safe when empty.
    void flush() {
        for (const std::unique_ptr<Entry>& entry : entries_)
            entry->run();
        entries_.clear();
    }

    /// @brief Returns true if nothing is queued.
    bool empty() const { return entries_.empty(); }

    /// @brief Returns the number of queued deleters.
    size_t size() const { return entries_.size(); }

private:
    struct Entry {
        virtual ~Entry() = default;
        virtual void run() = 0;
    };

    template <typename T>
    struct ResourceEntry final : Entry {
        explicit ResourceEntry(T&& resource) : held(std::move(resource)) {}
        void run() override { held = T{}; }

        T held;
    };

    struct CallableEntry final : Entry {
        explicit CallableEntry(std::function<void()> fn) : fn(std::move(fn)) {}
        void run() override { fn(); }

        std::function<void()> fn;
    };

    std::vector<std::unique_ptr<Entry>> entries_;
};

}
