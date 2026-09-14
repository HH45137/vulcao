#include "context.h"
#include <string>
#include <iostream>

namespace vulcao::graphics {
    vk::raii::Instance Context::make_instance(const vk::raii::Context& context,
                                             const std::string& appName,
                                             uint32_t appVersion) {
        vk::ApplicationInfo appInfo{
            .pApplicationName = appName.c_str(),
            .applicationVersion = appVersion,
            .apiVersion = VK_API_VERSION_1_3,
        };

        vk::InstanceCreateInfo createInfo{
            .pApplicationInfo = &appInfo,
        };

        return vk::raii::Instance(context, createInfo);
    }
    vk::raii::Device Context::make_device(){
        inquery_physical_devices_info();
        std::cout<<"------------------- devices info -------------------"<<std::endl;
        std::vector<vk::QueueFamilyProperties> queue_family_properties = physical_device_.getQueueFamilyProperties();
        
        std::cout << "number of queue families: " << queue_family_properties.size() << std::endl;
        for (uint32_t i = 0; i < queue_family_properties.size(); i++) {
            std::cout << "Queue family " << i << ": " << queue_family_properties[i].queueCount <<
                         " queues, flags: " << vk::to_string(queue_family_properties[i].queueFlags) << std::endl;
        }
        for (uint32_t i = 0; i < queue_family_properties.size(); i++) {
             if (queue_family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphics_queue_family_index_ = i;
                std::cout << "use graphics queue family index: " << graphics_queue_family_index_ << std::endl;
                break;
            }
        }

        float queuePriority = 1.0f;

        vk::DeviceQueueCreateInfo device_queue_create_info{};
        device_queue_create_info.queueFamilyIndex = graphics_queue_family_index_;
        device_queue_create_info.queueCount       = 1;
        device_queue_create_info.pQueuePriorities = &queuePriority;

        vk::DeviceCreateInfo deviceCreateInfo{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos    = &device_queue_create_info,
        };
        return vk::raii::Device(physical_device_, deviceCreateInfo);
    }
    vk::raii::CommandPool Context::make_command_pool(){
        vk::CommandPoolCreateInfo command_pool_create_info{};
        command_pool_create_info.flags = {};
        command_pool_create_info.queueFamilyIndex = graphics_queue_family_index_;

        return vk::raii::CommandPool(device_, command_pool_create_info);
    }
    vk::raii::CommandBuffer Context::make_command_buffer(){
        vk::CommandBufferAllocateInfo commandBufferAllocateInfo{
            .commandPool        = command_pool_,
            .level              = vk::CommandBufferLevel::ePrimary,
            .commandBufferCount = 1,
        };
        vk::raii::CommandBuffer       commandBuffer = std::move( vk::raii::CommandBuffers( device_, commandBufferAllocateInfo ).front() );
        return commandBuffer;
    }
    void Context::inquery_physical_devices_info(){
        std::cout<<"------------------- devices info -------------------"<<std::endl;
        if(physical_devices_.empty()){
            std::cerr << "not find physical device!!!!!!!\n";
        }
        for (const auto& physical_device : physical_devices_){
            auto const properties = physical_device.getProperties();

            std::string vendor_name;
            std::cout << "find gpu vendor: ";
            switch (properties.vendorID) {
                    case 0x10DE:
                        vendor_name = "NVIDIA";
                        break;
                    case 0x1002:
                        vendor_name = "AMD";
                        break;
                    case 0x8086:
                        vendor_name = "Intel";
                        break;
                    case 0x13B5:
                        vendor_name = "ARM";
                        break;
                    default:
                        vendor_name = "Unknown";
                        break; 
            }
            std::cout <<vendor_name <<". \n";

            char const * device_name = properties.deviceName;
            std::cout <<"device name: " << device_name<<". \n";

            std::string device_type;
            switch (properties.deviceType) {
                    case vk::PhysicalDeviceType::eCpu:
                        device_type = "CPU";
                        break;
                    case vk::PhysicalDeviceType::eOther:
                        device_type = "Other";
                        break;
                    case vk::PhysicalDeviceType::eVirtualGpu:
                        device_type = "Virtual GPU";
                        break;
                    case vk::PhysicalDeviceType::eDiscreteGpu:
                        device_type = "Discrete GPU";
                        break;
                    case vk::PhysicalDeviceType::eIntegratedGpu:
                        device_type = "Integrated GPU";
                        break;
                    default:
                        device_type = "Unknown";
                        break;

            }
            std::cout <<"device type: " << device_type << ". \n";

            std::cout<<"driver version: "<< properties.driverVersion <<". \n";
        }
    }

    Context::Context(const std::string& appName, uint32_t appVersion)
        : context_{},
          instance_{make_instance(context_, appName, appVersion)},
          physical_devices_(instance_),
          physical_device_(physical_devices_.front()),
          device_{make_device()},
          graphics_queue_{device_.getQueue(graphics_queue_family_index_, 0) },
          command_pool_{make_command_pool()},
          immediate_command_buffer_{ make_command_buffer() }
    {}
          
}
