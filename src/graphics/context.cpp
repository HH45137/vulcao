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

    void Context::inquiry_physical_devices_info(){
           if(physical_devices_.empty()){
               std::cerr << "not find physical device!!!!!!!\n";
           }
           for (const auto& physical_device : physical_devices_){
               auto properties = physical_device.getProperties();
               std::string vendor_name;
               char* device_name = properties.deviceName;
               std::cout << "find gpu vendor :";
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
                       vendor_name = "Unknow";
                       break; 
               }

                std::cout <<vendor_name<<". \ndevice name: " << device_name <<". \ndriver version："<< properties.driverVersion <<std::endl;
                
           }
    }

    Context::Context(const std::string& appName, uint32_t appVersion)
        : context_{},
          instance_{make_instance(context_, appName, appVersion)},
          physical_devices_(instance_)
    {}
          
}
