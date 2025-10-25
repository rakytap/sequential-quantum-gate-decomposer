#include <atomic>
#include <memory>
#include <utility>

template<typename T>
class SmartAtomicPtr {
public:
    SmartAtomicPtr() = default;

    explicit SmartAtomicPtr(std::shared_ptr<T> p) {
        std::atomic_store(&ptr_, std::move(p));
    }

    // Copy / assign load the current shared_ptr atomically
    SmartAtomicPtr(const SmartAtomicPtr& other) {
        std::atomic_store(&ptr_, std::atomic_load(&other.ptr_));
    }
    SmartAtomicPtr& operator=(const SmartAtomicPtr& other) {
        if (this != &other) {
            std::atomic_store(&ptr_, std::atomic_load(&other.ptr_));
        }
        return *this;
    }

    // Move ops
    SmartAtomicPtr(SmartAtomicPtr&& other) noexcept {
        auto p = std::atomic_load(&other.ptr_);
        std::atomic_store(&ptr_, std::move(p));
        std::atomic_store(&other.ptr_, std::shared_ptr<T>{});
    }
    SmartAtomicPtr& operator=(SmartAtomicPtr&& other) noexcept {
        if (this != &other) {
            std::atomic_store(&ptr_, std::atomic_load(&other.ptr_));
            std::atomic_store(&other.ptr_, std::shared_ptr<T>{});
        }
        return *this;
    }

    ~SmartAtomicPtr() = default; // shared_ptr cleans up when last ref dies

    // Replace the pointer; old value is released when 'old' goes out of scope
    void update(std::shared_ptr<T> p) {
        auto old = std::atomic_exchange(&ptr_, std::move(p));
        (void)old; // dropping 'old' decrements refcount
    }

    // Convenience for raw pointer input (creates sole owner control block)
    void update(T* p) { update(std::shared_ptr<T>(p)); }

    // Readers get a shared_ptr snapshot, safe against concurrent updates
    std::shared_ptr<T> get() const {
        return std::atomic_load(&ptr_);
    }

private:
    std::shared_ptr<T> ptr_;
};
