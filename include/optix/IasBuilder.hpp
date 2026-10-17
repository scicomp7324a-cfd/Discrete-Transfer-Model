# pragma once 

#include <iostream>
#include <vector>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include "core/Triangle.hpp"
#include "optix/GasBuilder.hpp"

class IasBuilder{

    private:

        CUdeviceptr d_instances_  = 0 ;
        CUdeviceptr d_tempBuffer_ = 0 ;
        CUdeviceptr d_iasBuffer_  = 0 ;
        OptixTraversableHandle handle_ = 0 ;
    
    public:

        IasBuilder(){}

        ~IasBuilder(){
            if(d_instances_) cudaFree(reinterpret_cast<void*>(d_instances_));
            if(d_tempBuffer_) cudaFree(reinterpret_cast<void*>(d_tempBuffer_));
            if(d_iasBuffer_) cudaFree(reinterpret_cast<void*>(d_iasBuffer_));
        }

        CUdeviceptr& d_instances(){
            return this->d_instances_ ;
        }

        CUdeviceptr& d_tempBuffer(){
            return this->d_tempBuffer_ ;
        }

        CUdeviceptr& d_iasBuffer(){
            return this->d_iasBuffer_ ;
        }

        OptixTraversableHandle& handle(){
            return this->handle_ ;
        }

        static void setIdentityTransform(float m[12]){
            m[0]  = 1.0f; m[1]  = 0.0f; m[2]  = 0.0f; m[3]  = 0.0f;
            m[4]  = 0.0f; m[5]  = 1.0f; m[6]  = 0.0f; m[7]  = 0.0f;
            m[8]  = 0.0f; m[9]  = 0.0f; m[10] = 1.0f; m[11] = 0.0f;
        }

        void buildIas(OptixDeviceContext context, cudaStream_t stream, GasBuilder& gasA, GasBuilder& gasB) ;

} ;