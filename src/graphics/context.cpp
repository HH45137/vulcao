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
        inquiry_physical_devices_info();
        std::cout<<"------------------- devices info -------------------"<<std::endl;
        uint32_t graphics_queue_family_index = 1.0;
        std::vector<vk::QueueFamilyProperties> queue_family_properties = physical_device_.getQueueFamilyProperties();
        
        std::cout << "number of queue families: " << queue_family_properties.size() << std::endl;
        for (uint32_t i = 0; i < queue_family_properties.size(); i++) {
            std::cout << "Queue family " << i << ": " << queue_family_properties[i].queueCount <<
                         " queues, flags: " << vk::to_string(queue_family_properties[i].queueFlags) << std::endl;
        }
        for (uint32_t i = 0; i < queue_family_properties.size(); i++) {
             if (queue_family_properties[i].queueFlags & vk::QueueFlagBits::eGraphics) {
                graphics_queue_family_index = i;
                std::cout << "use graphics queue family index: " << graphics_queue_family_index << std::endl;
                break;
            }
        }

        float queuePriority = 1.0f;

        vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
        deviceQueueCreateInfo.queueFamilyIndex = graphics_queue_family_index;
        deviceQueueCreateInfo.queueCount       = 1;
        deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

        vk::DeviceCreateInfo      deviceCreateInfo( {}, deviceQueueCreateInfo );
        return vk::raii::Device(physical_device_, deviceCreateInfo);
    }
    void Context::inquiry_physical_devices_info(){
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
          device_{make_device()}
    {}
          
}
