# pragma once 

#include <iostream>
#include <vector>
#include <array>

#include <cuda.h>
#include <cuda_runtime.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_device.h>

#include "radiation/DataBuilder.hpp"
#include "core/RayGeometry.hpp"
#include "core/TriangleTraceHistory.hpp"
#include "core/RayParams.hpp"
#include "optix/IasBuilder.hpp"
#include "optix/PipelineBuilder.hpp"
#include "optix/SbtBuilder.hpp"


class LaunchManager{

    private:

        // Host side code 

        vector<float3> h_rayOrigins_ ;
        vector<float3> h_rayDirections_ ;
        vector<int> h_originTriangleIds_ ;
        vector<int> h_originSurfaceIds_ ;
        vector<int> h_originGlobalTriangleIds_ ;
        vector<int> h_surfaceTriangleOffsets_ ;

        const unsigned int numRays_ ;
        const unsigned int numTotalTriangles_ ;

        vector<RayGeometry> h_info_ ;
        vector<TriangleTraceHistory> h_triangleInfo_ ;
        vector<HitRecord> h_hitRecords_ ;

        Params h_params_ = {} ;

        // Device side code

        CUdeviceptr d_rayOrigins_ = 0 ;
        CUdeviceptr d_rayDirections_ = 0 ;
        CUdeviceptr d_originTriangleIds_ = 0 ;
        CUdeviceptr d_originSurfaceIds_ = 0;
        CUdeviceptr d_hitRecords_ = 0;
        CUdeviceptr d_originGlobalTriangleIds_ = 0 ;
        CUdeviceptr d_surfaceTriangleOffsets_ = 0 ;

        CUdeviceptr d_info_ = 0;
        CUdeviceptr d_triangleInfo_ = 0 ;

        CUdeviceptr d_params_ = 0;

    public:

        LaunchManager(DataBuilder& dataBuilder, IasBuilder& ias):h_rayOrigins_(dataBuilder.rayOrigins()),
                                                h_rayDirections_(dataBuilder.rayDirections()),
                                                h_originTriangleIds_(dataBuilder.originTriangleIds()),
                                                h_originSurfaceIds_(dataBuilder.originSurfaceIds()),
                                                h_originGlobalTriangleIds_(dataBuilder.originGlobalTriangleIds()),
                                                numRays_(dataBuilder.numTotalRays()),
                                                numTotalTriangles_(dataBuilder.indicesA().size() + dataBuilder.indicesB().size()),
                                                h_info_(this->numRays_),
                                                h_triangleInfo_(this->numTotalTriangles_),
                                                h_hitRecords_(this->numRays_),
                                                h_surfaceTriangleOffsets_{0, static_cast<int>(dataBuilder.indicesA().size())}
        {

            for(int i=0; i<dataBuilder.indicesA().size(); ++i){
                this->h_triangleInfo_[i].setTriangleId(i) ;
                this->h_triangleInfo_[i].setSurfaceId(0) ;
            }

            for(int i = dataBuilder.indicesA().size(); i < dataBuilder.indicesA().size()+ dataBuilder.indicesB().size(); ++i){
                this->h_triangleInfo_[i].setTriangleId(i-dataBuilder.indicesA().size()) ;
                this->h_triangleInfo_[i].setSurfaceId(1) ;
            }

            if (this->h_rayDirections().size() != this->numRays() || 
                this->h_originTriangleIds().size() != this->numRays() || 
                this->h_rayOrigins().size() != this->numRays() || 
                this->h_originSurfaceIds().size() != this->numRays() ||
                this->h_originGlobalTriangleIds().size() != this->numRays())
            {
                cerr << "Input arrays do not have the same length\n";
                cerr << "numRays = " << this->numRays() << endl;
                cerr << "rayOrigins = " << this->h_rayOrigins().size() << endl;
                cerr << "rayDirections = " << this->h_rayDirections().size() << endl;
                cerr << "originTriangleIds = " << this->h_originTriangleIds().size() << endl;
                cerr << "originSurfaceIds = " << this->h_originSurfaceIds().size() << endl;
                cerr << "originGlobalTriangleIds = " << this->h_originGlobalTriangleIds().size() << endl;
                throw std::runtime_error("Input arrays do not have the same length\n");
            }
            
            this->initialiseDeviceVariables() ;
            this->initialiseHostParameters(dataBuilder, ias) ;
            this->initialiseDeviceParameters() ;

        }

        ~LaunchManager(){
            if(d_rayOrigins_) cudaFree(reinterpret_cast<void*>(d_rayOrigins_));
            if(d_rayDirections_) cudaFree(reinterpret_cast<void*>(d_rayDirections_));
            if(d_originTriangleIds_) cudaFree(reinterpret_cast<void*>(d_originTriangleIds_));
            if(d_originSurfaceIds_) cudaFree(reinterpret_cast<void*>(d_originSurfaceIds_));
            if(d_originGlobalTriangleIds_) cudaFree(reinterpret_cast<void*>(d_originGlobalTriangleIds_));
            if(d_info_) cudaFree(reinterpret_cast<void*>(d_info_));
            if(d_triangleInfo_) cudaFree(reinterpret_cast<void*>(d_triangleInfo_));
            if(d_hitRecords_) cudaFree(reinterpret_cast<void*>(d_hitRecords_));
            if(d_surfaceTriangleOffsets_) cudaFree(reinterpret_cast<void*>(d_surfaceTriangleOffsets_));
            if(d_params_) cudaFree(reinterpret_cast<void*>(d_params_));
        }

        vector<float3>& h_rayOrigins(){ return this->h_rayOrigins_ ; }
        vector<float3>& h_rayDirections(){ return this->h_rayDirections_ ;} 
        vector<int>& h_originTriangleIds(){ return this->h_originTriangleIds_ ;}
        vector<int>& h_originSurfaceIds(){ return this->h_originSurfaceIds_ ;}
        vector<HitRecord>& h_hitRecords(){ return this->h_hitRecords_ ;}
        vector<int>& h_originGlobalTriangleIds(){ return this->h_originGlobalTriangleIds_ ;}

        unsigned int numRays() const{ return this->numRays_ ; }
        unsigned int numTotalTriangles() const{ return this->numTotalTriangles_ ;}

        vector<RayGeometry>& h_info(){ return this->h_info_ ;}
        vector<TriangleTraceHistory>& h_triangleInfo(){ return this->h_triangleInfo_ ;}
        
        Params& h_params(){ return this->h_params_ ;}

        CUdeviceptr& d_rayOrigins(){ return this->d_rayOrigins_ ;}
        CUdeviceptr& d_rayDirections(){ return this->d_rayDirections_ ;}
        CUdeviceptr& d_originTriangleIds(){ return this->d_originTriangleIds_ ;}
        CUdeviceptr& d_originSurfaceIds(){ return this->d_originSurfaceIds_ ;}
        CUdeviceptr& d_hitRecords(){ return this->d_hitRecords_ ;}
        CUdeviceptr& d_originGlobalTriangleIds(){ return d_originGlobalTriangleIds_ ;}
        CUdeviceptr& d_surfaceTriangleOffsets(){ return d_surfaceTriangleOffsets_;}

        CUdeviceptr& d_info(){ return this->d_info_ ;}
        CUdeviceptr& d_triangleInfo(){ return this->d_triangleInfo_ ;}

        CUdeviceptr& d_params(){ return this->d_params_ ;}

        void initialiseDeviceVariables() ;
        void initialiseHostParameters(DataBuilder& dataBuilder, IasBuilder& ias) ;
        void initialiseDeviceParameters() ;
        OptixResult LaunchOptixSimulation(PipelineBuilder& pipelineBuilder, SbtBuilder& sbtBuilder, cudaStream_t& stream) ;

} ;
