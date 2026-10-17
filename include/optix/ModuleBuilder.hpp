# pragma once 

#include <iostream>
#include <vector>
#include <array>
#include <string>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include "core/Triangle.hpp"
#include "optix/PipelineBuilder.hpp"


using namespace std ;

class ModuleBuilder{

    private:
    OptixModule module_ = nullptr;
    OptixModuleCompileOptions moduleCompileOptions_ ;
    char log_[4096];
    size_t logSize_ = sizeof(log_);

    public:
    ModuleBuilder(){
        this->setModuleCompileOptions() ;
    } ;

    OptixModule& module(){
        return this->module_ ;
    }

    OptixModuleCompileOptions& optixModuleCompileOptions(){
        return this->moduleCompileOptions_ ;
    }

    char* moduleLog(){
        return this->log_ ;
    }

    size_t& moduleLogSize(){
        return this->logSize_ ;
    }

    void setModuleCompileOptions() ;
    void createModule(OptixDeviceContext context, OptixPipelineCompileOptions pipelineCompileOptions, string ptxCode) ;

} ;