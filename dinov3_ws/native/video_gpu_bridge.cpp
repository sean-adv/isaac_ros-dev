// Isolated DeepStream GPU OSD -> NVENC pipeline. stdout is a binary protocol.
#include <nvbufsurface.h>
#include <nvbufsurftransform.h>
#include <nvll_osd_api.h>
#include <cuda_runtime_api.h>
extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/opt.h>
}
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
#include <cstdint>
#include <vector>
#include <string>
#include <stdexcept>

extern "C" int resize_rgb_rgba(const void*,int,int,int,void*,int,int,int);

static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static bool read_all(void* p,size_t n){return fread(p,1,n,stdin)==n;}
struct Resources {
    NvBufSurface *input=nullptr,*rgba=nullptr,*nv12=nullptr;
    NvOSDCtxHandle osd=nullptr;
    AVBufferRef *device=nullptr,*frames=nullptr;
    AVCodecContext* encoder=nullptr;
    AVFrame* frame=nullptr; AVPacket* packet=nullptr;
    void* shared=MAP_FAILED;size_t bytes=0;int fd=-1;
    ~Resources(){
        av_packet_free(&packet);av_frame_free(&frame);avcodec_free_context(&encoder);
        av_buffer_unref(&frames);av_buffer_unref(&device);
        if(osd)nvll_osd_destroy_context(osd);
        for(auto* surface:{input,rgba,nv12})if(surface)NvBufSurfaceDestroy(surface);
        if(shared!=MAP_FAILED)munmap(shared,bytes);
        if(fd>=0)close(fd);
    }
};
static NvBufSurface* allocate(unsigned width,unsigned height,NvBufSurfaceColorFormat format){
    NvBufSurfaceCreateParams p{};p.gpuId=0;p.width=width;p.height=height;
    p.layout=NVBUF_LAYOUT_PITCH;p.colorFormat=format;p.memType=NVBUF_MEM_CUDA_DEVICE;
    NvBufSurface* surface=nullptr;require(NvBufSurfaceCreate(&surface,1,&p)==0,"video surface allocation failed");
    surface->numFilled=1;return surface;
}
int main(int argc,char** argv){
    FILE* protocol=fdopen(dup(STDOUT_FILENO),"wb");dup2(STDERR_FILENO,STDOUT_FILENO);
    try{
        require(argc==4,"usage: video_gpu_bridge RGB_SHM FPS BITRATE");Resources r;
        r.fd=shm_open(argv[1],O_RDONLY,0);require(r.fd>=0,"video shm open failed");
        struct stat st{};require(fstat(r.fd,&st)==0,"video shm stat failed");r.bytes=st.st_size;
        r.shared=mmap(nullptr,r.bytes,PROT_READ,MAP_SHARED,r.fd,0);require(r.shared!=MAP_FAILED,"video shm mmap failed");
        AVDictionary* options=nullptr;av_dict_set(&options,"primary_ctx","1",0);
        require(av_hwdevice_ctx_create(&r.device,AV_HWDEVICE_TYPE_CUDA,"0",options,0)>=0,"CUDA video device failed");av_dict_free(&options);
        r.frames=av_hwframe_ctx_alloc(r.device);require(r.frames,"CUDA frame context failed");
        auto* frames=reinterpret_cast<AVHWFramesContext*>(r.frames->data);
        frames->format=AV_PIX_FMT_CUDA;frames->sw_format=AV_PIX_FMT_NV12;
        frames->width=960;frames->height=540;frames->initial_pool_size=2;
        require(av_hwframe_ctx_init(r.frames)>=0,"CUDA video frames failed");
        const AVCodec* codec=avcodec_find_encoder_by_name("h264_nvenc");require(codec,"NVENC not available");
        r.encoder=avcodec_alloc_context3(codec);r.encoder->width=960;r.encoder->height=540;
        r.encoder->pix_fmt=AV_PIX_FMT_CUDA;r.encoder->time_base={1,90000};r.encoder->framerate={std::stoi(argv[2]),1};
        r.encoder->bit_rate=std::stoll(argv[3]);r.encoder->gop_size=std::stoi(argv[2]);r.encoder->max_b_frames=0;
        r.encoder->hw_frames_ctx=av_buffer_ref(r.frames);
        av_dict_set(&options,"preset","p1",0);av_dict_set(&options,"tune","ull",0);
        av_dict_set(&options,"profile","baseline",0);av_dict_set(&options,"forced-idr","1",0);
        av_dict_set(&options,"delay","0",0);av_dict_set(&options,"rc-lookahead","0",0);av_dict_set(&options,"zerolatency","1",0);
        require(avcodec_open2(r.encoder,codec,&options)>=0,"NVENC initialization failed");av_dict_free(&options);
        r.frame=av_frame_alloc();r.packet=av_packet_alloc();
        r.rgba=allocate(960,540,NVBUF_COLOR_FORMAT_RGBA);r.nv12=allocate(960,540,NVBUF_COLOR_FORMAT_NV12);
        r.osd=nvll_osd_create_context();require(r.osd,"GPU OSD context failed");nvll_osd_set_params(r.osd,960,540);
        NvBufSurfTransformConfigParams config{};config.compute_mode=NvBufSurfTransformCompute_GPU;config.gpu_id=0;
        require(NvBufSurfTransformSetSessionParams(&config)==NvBufSurfTransformError_Success,"video transform setup failed");
        uint32_t magic=0x47564944;fwrite(&magic,4,1,protocol);fflush(protocol);
        while(true){
            uint32_t h[7];if(!read_all(h,sizeof(h)))break;
            require(h[0]&&h[1]&&size_t(h[0])*h[1]*3<=r.bytes&&h[2]<=2400&&h[3]<=200,"invalid video request");
            std::vector<NvOSD_LineParams> lines(h[2]);
            for(auto& line:lines){
                float p[7];require(read_all(p,sizeof(p)),"truncated lines");
                line.x1=p[0];line.y1=p[1];line.x2=p[2];line.y2=p[3];line.line_width=2;
                line.line_color={p[4],p[5],p[6],1.};
            }
            std::vector<NvOSD_TextParams> texts(h[3]);std::vector<std::string> labels(h[3]);
            for(unsigned i=0;i<h[3];++i){
                uint32_t p[6];require(read_all(p,sizeof(p))&&p[2]<=4096,"invalid text");labels[i].resize(p[2]);
                require(read_all(labels[i].data(),p[2]),"truncated text");
                auto& text=texts[i];text.display_text=labels[i].data();text.x_offset=p[0];text.y_offset=p[1];
                text.font_params.font_name=const_cast<char*>("Sans");text.font_params.font_size=13;
                text.font_params.font_color={p[3]/255.,p[4]/255.,p[5]/255.,1.};
                text.set_bg_clr=1;text.text_bg_clr={0.,0.,0.,.65};
            }
            if(!r.input||r.input->surfaceList[0].width!=h[0]||r.input->surfaceList[0].height!=h[1]){
                if(r.input)NvBufSurfaceDestroy(r.input);
                r.input=allocate(h[0],h[1],NVBUF_COLOR_FORMAT_RGB);
            }
            auto& input=r.input->surfaceList[0];
            require(cudaMemcpy2D(input.dataPtr,input.pitch,r.shared,h[0]*3,h[0]*3,h[1],cudaMemcpyHostToDevice)==cudaSuccess,"video RGB upload failed");
            NvBufSurfTransformParams transform{};transform.transform_flag=NVBUFSURF_TRANSFORM_FILTER;
            transform.transform_filter=NvBufSurfTransformInter_Bilinear;
            auto& rgba_surface=r.rgba->surfaceList[0];
            require(resize_rgb_rgba(input.dataPtr,h[0],h[1],input.pitch,rgba_surface.dataPtr,960,540,rgba_surface.pitch)==0,"video CUDA resize failed");
            if(!lines.empty()){
                NvOSD_FrameLineParams params{};params.mode=MODE_GPU;params.num_lines=lines.size();params.line_params_list=lines.data();params.surf=r.rgba;
                require(nvll_osd_draw_lines(r.osd,&params)==0,"GPU lines failed");
            }
            if(!texts.empty()){
                NvOSD_FrameTextParams params{};params.mode=MODE_GPU;params.num_strings=texts.size();params.text_params_list=texts.data();params.surf=r.rgba;
                require(nvll_osd_put_text(r.osd,&params)==0,"GPU text failed");
            }
            require(nvll_osd_apply(r.osd,nullptr,r.rgba)==0,"GPU OSD apply failed");
            require(NvBufSurfTransform(r.rgba,r.nv12,&transform)==NvBufSurfTransformError_Success,"video GPU NV12 conversion failed");
            av_frame_unref(r.frame);require(av_hwframe_get_buffer(r.frames,r.frame,0)>=0,"encoder CUDA frame failed");
            auto& nv12=r.nv12->surfaceList[0];
            for(int plane=0;plane<2;++plane){
                auto* src=static_cast<unsigned char*>(nv12.dataPtr)+nv12.planeParams.offset[plane];
                require(cudaMemcpy2D(r.frame->data[plane],r.frame->linesize[plane],src,nv12.planeParams.pitch[plane],960,plane?270:540,cudaMemcpyDeviceToDevice)==cudaSuccess,"encoder CUDA copy failed");
            }
            r.frame->pts=(int64_t(h[5])<<32)|h[6];r.frame->pict_type=h[4]?AV_PICTURE_TYPE_I:AV_PICTURE_TYPE_NONE;
            require(avcodec_send_frame(r.encoder,r.frame)>=0,"NVENC send failed");
            require(avcodec_receive_packet(r.encoder,r.packet)>=0,"NVENC buffered frame instead of immediate output");
            uint32_t response[2]={uint32_t(r.packet->size),uint32_t(bool(r.packet->flags&AV_PKT_FLAG_KEY))};
            require(fwrite(response,sizeof(response),1,protocol)==1&&fwrite(r.packet->data,1,r.packet->size,protocol)==size_t(r.packet->size),"video protocol write failed");
            fflush(protocol);av_packet_unref(r.packet);
        }
        return 0;
    }catch(const std::exception& e){fprintf(stderr,"GPU video bridge: %s\n",e.what());return 1;}
}
