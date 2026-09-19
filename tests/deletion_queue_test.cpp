#include <utility>

#include <doctest/doctest.h>

#include <vulcao/deletion_queue.h>

namespace {

/// @brief Counts how often its payload is destroyed, through moves.
class Probe {
public:
    Probe() = default;
    explicit Probe(int* counter) : counter_(counter) {}

    Probe(const Probe&) = delete;
    Probe& operator=(const Probe&) = delete;

    Probe(Probe&& other) noexcept : counter_(std::exchange(other.counter_, nullptr)) {}

    Probe& operator=(Probe&& other) noexcept {
        if (this != &other) {
            release();
            counter_ = std::exchange(other.counter_, nullptr);
        }
        return *this;
    }

    ~Probe() { release(); }

private:
    void release() {
        if (counter_ != nullptr)
            ++*counter_;
        counter_ = nullptr;
    }

    int* counter_ = nullptr;
};

}

TEST_CASE("deletion queue destroys pushed resources at flush") {
    vulcao::DeletionQueue queue;

    int destroyed = 0;
    queue.push(Probe{&destroyed});
    CHECK(destroyed == 0);
    CHECK(queue.size() == 1);

    queue.flush();
    CHECK(destroyed == 1);
    CHECK(queue.empty());

    // Flushing twice is a no-op.
    queue.flush();
    CHECK(destroyed == 1);
}

TEST_CASE("deletion queue flushes on destruction") {
    int destroyed = 0;
    {
        vulcao::DeletionQueue queue;
        queue.push(Probe{&destroyed});
        queue.push(Probe{&destroyed});
        CHECK(destroyed == 0);
    }
    CHECK(destroyed == 2);
}

TEST_CASE("deletion queue runs raw callables once") {
    vulcao::DeletionQueue queue;

    int calls = 0;
    queue.push([&calls] { ++calls; });
    queue.flush();
    queue.flush();
    CHECK(calls == 1);
}

TEST_CASE("deletion queue moves pending deleters") {
    int destroyed = 0;

    vulcao::DeletionQueue first;
    first.push(Probe{&destroyed});

    vulcao::DeletionQueue second = std::move(first);
    CHECK(first.empty());
    CHECK(destroyed == 0);

    second.flush();
    CHECK(destroyed == 1);
}
