#include <vulkan/vulkan_raii.hpp>
#include <iostream>

#include "graphics/context.h"

int main(){
    std::cout<<"hello world"<<std::endl;
    vulcao::graphics::Context ctx{"vulcao-game"};
    std::cout<<"instance created"<<std::endl;
    return 0;
}
