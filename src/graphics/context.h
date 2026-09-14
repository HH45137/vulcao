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
            vk::raii::PhysicalDevice& physical_device() {return physical_device_;}
            
            void inquery_physical_devices_info();
        

        private:
            vk::raii::Instance make_instance(const vk::raii::Context& context,
                                                   const std::string& app_name,
                                                   uint32_t app_version);
            vk::raii::Device make_device();
            vk::raii::CommandPool make_command_pool();
            vk::raii::CommandBuffer make_command_buffer();


            uint32_t graphics_queue_family_index_ = 0;

            vk::raii::Context context_;
            vk::raii::Instance instance_;
            vk::raii::PhysicalDevices physical_devices_;
            vk::raii::PhysicalDevice physical_device_;
            vk::raii::Device device_;
            vk::raii::Queue graphics_queue_;
            vk::raii::CommandPool command_pool_;
            vk::raii::CommandBuffer immediate_command_buffer_;
            
           

    };
}
