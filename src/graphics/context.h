#pragma once

#include <string>
#include <vector>
#include <vulkan/vulkan.hpp>

namespace vulcao::graphics {
    class Context {
        public:
            explicit Context(const std::string& appName = "vulcao",
                             uint32_t appVersion = VK_MAKE_VERSION(1, 0, 0));
            ~Context();

            Context(const Context&) = delete;
            Context& operator=(const Context&) = delete;
            Context(Context&&) = delete;
            Context& operator=(Context&&) = delete;

            vk::Instance& instance() { return instance_; }
            vk::PhysicalDevice& physical_device() {return physical_device_;}
            vk::Device& device() { return device_; }
            vk::Queue& graphics_queue() { return graphics_queue_; }
            vk::CommandPool& command_pool() { return command_pool_; }
            vk::CommandBuffer& immediate_command_buffer() { return immediate_command_buffer_; }

            void inquery_physical_devices_info();


        private:
            vk::Instance make_instance(const std::string& app_name,
                                       uint32_t app_version);
            vk::Device make_device();
            vk::CommandPool make_command_pool();
            vk::CommandBuffer make_command_buffer();


            uint32_t graphics_queue_family_index_ = 0;

            vk::Instance instance_;
            std::vector<vk::PhysicalDevice> physical_devices_;
            vk::PhysicalDevice physical_device_;
            vk::Device device_;
            vk::Queue graphics_queue_;
            vk::CommandPool command_pool_;
            vk::CommandBuffer immediate_command_buffer_;



    };
}
