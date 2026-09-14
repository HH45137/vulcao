#pragma once

#include <string>
#include <vulkan/vulkan_raii.hpp>

namespace vulcao::graphics {
    class Context {
        public:
            explicit Context(const std::string& appName = "vulcao",
                             uint32_t appVersion = VK_MAKE_VERSION(1, 0, 0));
            ~Context() = default;

            Context(const Context&) = delete;
            Context& operator=(const Context&) = delete;
            Context(Context&&) = delete;
            Context& operator=(Context&&) = delete;

            vk::raii::Instance& instance() { return instance_; }
            vk::raii::PhysicalDevices& physical_device() {return physical_devices_;}
            
            void inquiry_physical_devices_info();
        

        private:
            vk::raii::Instance make_instance(const vk::raii::Context& context,
                                                   const std::string& app_name,
                                                   uint32_t app_version);
            vk::raii::Device make_device();
            vk::raii::Context context_;
            vk::raii::Instance instance_;
            vk::raii::PhysicalDevices physical_devices_;
            vk::raii::PhysicalDevice physical_device_;
            vk::raii::Device device_;

    };
}
