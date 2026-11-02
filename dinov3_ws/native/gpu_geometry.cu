// Small CUDA kernels for ROI preparation and plane RANSAC. No DeepStream/TRT ABI.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <cstdint>
#include <new>

static thread_local char error_text[512] = "";
static bool check(cudaError_t e) {
    if (e == cudaSuccess) return true;
    snprintf(error_text, sizeof(error_text), "%s", cudaGetErrorString(e)); return false;
}
extern "C" const char* geometry_error() { return error_text; }
struct PlaneContext {
    cudaStream_t stream{};
    float *points=nullptr, *planes=nullptr;
    int *counts=nullptr, *winner=nullptr;
    unsigned char* mask=nullptr;
    size_t capacity=0;
};
__device__ unsigned hash32(unsigned x) {
    x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; return x ^ (x >> 16);
}
__global__ void plane_candidates(const float* points, int n, float threshold,
                                  unsigned seed, float* planes, int* counts) {
    __shared__ float model[4]; __shared__ int sums[256];
    int hypothesis=blockIdx.x;
    if (!threadIdx.x) {
        unsigned s=hash32(seed+hypothesis*747796405U);
        int a=s%n, b=hash32(s+1)%n, c=hash32(s+2)%n;
        float3 p=make_float3(points[a*3],points[a*3+1],points[a*3+2]);
        float3 u=make_float3(points[b*3]-p.x,points[b*3+1]-p.y,points[b*3+2]-p.z);
        float3 v=make_float3(points[c*3]-p.x,points[c*3+1]-p.y,points[c*3+2]-p.z);
        float3 normal=make_float3(u.y*v.z-u.z*v.y,u.z*v.x-u.x*v.z,u.x*v.y-u.y*v.x);
        float length=sqrtf(normal.x*normal.x+normal.y*normal.y+normal.z*normal.z);
        float inv=length>1e-10f?1.f/length:0.f;
        model[0]=normal.x*inv;model[1]=normal.y*inv;model[2]=normal.z*inv;
        model[3]=-(model[0]*p.x+model[1]*p.y+model[2]*p.z);
        if (!inv) model[3]=1e20f;
        for(int i=0;i<4;++i)planes[hypothesis*4+i]=model[i];
    }
    __syncthreads(); int count=0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)
        count+=fabsf(model[0]*points[i*3]+model[1]*points[i*3+1]+model[2]*points[i*3+2]+model[3])<=threshold;
    sums[threadIdx.x]=count;__syncthreads();
    for(int stride=128;stride;stride>>=1){if(threadIdx.x<stride)sums[threadIdx.x]+=sums[threadIdx.x+stride];__syncthreads();}
    if(!threadIdx.x)counts[hypothesis]=sums[0];
}
__global__ void select_plane(const int* counts,int iterations,int* winner) {
    if(threadIdx.x||blockIdx.x)return;
    int best=0;for(int i=1;i<iterations;++i)if(counts[i]>counts[best])best=i;
    *winner=best;
}
__global__ void plane_mask(const float* p,int n,const float* planes,const int* winner,
                           float threshold,unsigned char* mask) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const float* m=planes+*winner*4;
    mask[i]=fabsf(m[0]*p[i*3]+m[1]*p[i*3+1]+m[2]*p[i*3+2]+m[3])<=threshold;
}
extern "C" void* plane_create() {
    auto* c=new(std::nothrow) PlaneContext;if(!c)return nullptr;
    int least=0,greatest=0;cudaDeviceGetStreamPriorityRange(&least,&greatest);
    if(!check(cudaStreamCreateWithPriority(&c->stream,cudaStreamNonBlocking,least))){delete c;return nullptr;}
    if(!check(cudaMalloc(&c->planes,1024*4*sizeof(float))) || !check(cudaMalloc(&c->counts,1024*sizeof(int))) || !check(cudaMalloc(&c->winner,sizeof(int)))) {
        cudaFree(c->planes);cudaFree(c->counts);cudaFree(c->winner);cudaStreamDestroy(c->stream);delete c;return nullptr;
    }
    return c;
}
extern "C" void plane_destroy(void* ptr) {
    auto* c=static_cast<PlaneContext*>(ptr);if(!c)return;
    cudaStreamSynchronize(c->stream);cudaFree(c->points);cudaFree(c->mask);cudaFree(c->planes);
    cudaFree(c->counts);cudaFree(c->winner);cudaStreamDestroy(c->stream);delete c;
}
extern "C" int plane_fit(void* ptr,const float* points,int n,float threshold,int iterations,
                          unsigned seed,unsigned char* mask) {
    auto* c=static_cast<PlaneContext*>(ptr);
    if(!c||n<3||iterations<1||iterations>1024||!(threshold>0))return -1;
    if(size_t(n)>c->capacity) {
        cudaStreamSynchronize(c->stream);cudaFree(c->points);cudaFree(c->mask);c->points=nullptr;c->mask=nullptr;c->capacity=0;
        size_t capacity=std::max(size_t(n),size_t(4096));
        if(!check(cudaMalloc(&c->points,capacity*3*sizeof(float)))||!check(cudaMalloc(&c->mask,capacity)))return -1;
        c->capacity=capacity;
    }
    if(!check(cudaMemcpyAsync(c->points,points,n*3*sizeof(float),cudaMemcpyHostToDevice,c->stream)))return -1;
    plane_candidates<<<iterations,256,0,c->stream>>>(c->points,n,threshold,seed,c->planes,c->counts);
    select_plane<<<1,1,0,c->stream>>>(c->counts,iterations,c->winner);
    plane_mask<<<(n+255)/256,256,0,c->stream>>>(c->points,n,c->planes,c->winner,threshold,c->mask);
    if(!check(cudaGetLastError())||!check(cudaMemcpyAsync(mask,c->mask,n,cudaMemcpyDeviceToHost,c->stream))||!check(cudaStreamSynchronize(c->stream)))return -1;
    return 0;
}

// Integer-coefficient bilinear interpolation with centered white padding.
// CUDA/OpenCV rounding may differ by one uint8 intensity level.
__device__ int sample_rgb(const unsigned char* rgb,int width,int height,int pitch,
                          const int* box,int px,int py,int side,int left,int top,int channel) {
    px-=left;py-=top;
    if(px<0||py<0||px>=box[2]-box[0]||py>=box[3]-box[1])return 255;
    return rgb[(box[1]+py)*pitch+(box[0]+px)*3+channel];
}
__global__ void prepare_rois(const unsigned char* rgb,int width,int height,int pitch,
                             const int* boxes,int count,int output_size,unsigned char* output) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;
    int plane=output_size*output_size*3,roi=index/plane;if(roi>=count)return;
    int pixel=index%plane,c=pixel%3,x=(pixel/3)%output_size,y=pixel/(3*output_size);
    const int* box=boxes+roi*4;int w=box[2]-box[0],h=box[3]-box[1],side=max(w,h);
    int left=(side-w)/2,top=(side-h)/2;
    float fx=(x+.5f)*side/output_size-.5f,fy=(y+.5f)*side/output_size-.5f;
    int x0=floorf(fx),y0=floorf(fy);fx-=x0;fy-=y0;
    if(x0<0){x0=0;fx=0;}if(x0>=side-1){x0=side-1;fx=0;}
    if(y0<0){y0=0;fy=0;}if(y0>=side-1){y0=side-1;fy=0;}
    int ax=__float2int_rn(fx*2048),ay=__float2int_rn(fy*2048);
    int a=sample_rgb(rgb,width,height,pitch,box,x0,y0,side,left,top,c);
    int b=sample_rgb(rgb,width,height,pitch,box,min(x0+1,side-1),y0,side,left,top,c);
    int d=sample_rgb(rgb,width,height,pitch,box,x0,min(y0+1,side-1),side,left,top,c);
    int e=sample_rgb(rgb,width,height,pitch,box,min(x0+1,side-1),min(y0+1,side-1),side,left,top,c);
    output[index]=((a*(2048-ax)+b*ax)*(2048-ay)+(d*(2048-ax)+e*ax)*ay+(1<<21))>>22;
}
extern "C" int prepare_rgb_rois(const void* rgb,int width,int height,int pitch,const void* boxes,
                                 int count,int size,void* output,void* stream) {
    if(!rgb||!boxes||!output||width<1||height<1||count<1||size<1)return -1;
    int total=count*size*size*3;
    prepare_rois<<<(total+255)/256,256,0,static_cast<cudaStream_t>(stream)>>>(
        static_cast<const unsigned char*>(rgb),width,height,pitch,static_cast<const int*>(boxes),count,size,static_cast<unsigned char*>(output));
    return check(cudaGetLastError())?0:-1;
}

__global__ void rgb_rgba_resize(const unsigned char* input,int width,int height,int pitch,
                                unsigned char* output,int out_width,int out_height,int out_pitch) {
    int index=blockIdx.x*blockDim.x+threadIdx.x;if(index>=out_width*out_height)return;
    int x=index%out_width,y=index/out_width;
    float fx=(x+.5f)*width/out_width-.5f,fy=(y+.5f)*height/out_height-.5f;
    int x0=floorf(fx),y0=floorf(fy);fx-=x0;fy-=y0;
    if(x0<0){x0=0;fx=0;}if(x0>=width-1){x0=width-1;fx=0;}
    if(y0<0){y0=0;fy=0;}if(y0>=height-1){y0=height-1;fy=0;}
    int x1=min(x0+1,width-1),y1=min(y0+1,height-1);
    int ax=__float2int_rn(fx*2048),ay=__float2int_rn(fy*2048);
    for(int c=0;c<3;++c){
        int a=input[y0*pitch+x0*3+c],b=input[y0*pitch+x1*3+c];
        int d=input[y1*pitch+x0*3+c],e=input[y1*pitch+x1*3+c];
        output[y*out_pitch+x*4+c]=((a*(2048-ax)+b*ax)*(2048-ay)+(d*(2048-ax)+e*ax)*ay+(1<<21))>>22;
    }
    output[y*out_pitch+x*4+3]=255;
}
extern "C" int resize_rgb_rgba(const void* input,int width,int height,int pitch,
                                void* output,int out_width,int out_height,int out_pitch) {
    int total=out_width*out_height;
    rgb_rgba_resize<<<(total+255)/256,256>>>(static_cast<const unsigned char*>(input),width,height,pitch,
                                           static_cast<unsigned char*>(output),out_width,out_height,out_pitch);
    return check(cudaGetLastError())&&check(cudaStreamSynchronize(nullptr))?0:-1;
}
