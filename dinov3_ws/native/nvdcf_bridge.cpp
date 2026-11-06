// Isolated DeepStream NvMOT bridge. stdout is exclusively the binary protocol.
#include <nvdstracker.h>
#include <nvbufsurftransform.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

extern "C" int resize_rgb_rgba(const void*,int,int,int,void*,int,int,int);

static void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}
static bool read_all(void* ptr, size_t bytes) {
    auto p = static_cast<char*>(ptr);
    while (bytes) {
        size_t n = fread(p, 1, bytes, stdin);
        if (!n) return false;
        p += n; bytes -= n;
    }
    return true;
}
template<class T> void write_value(const T& value, FILE* out) {
    require(fwrite(&value, sizeof(T), 1, out) == 1, "protocol write failed");
}
int main(int argc, char** argv) {
    // NVIDIA libraries may print to stdout: redirect their output to stderr.
    FILE* protocol = fdopen(dup(STDOUT_FILENO), "wb");
    dup2(STDERR_FILENO, STDOUT_FILENO);
    NvMOTContextHandle context = nullptr;
    NvBufSurface* surface = nullptr;
    NvBufSurface* input_surface = nullptr;
    NvBufSurface* rgba_surface = nullptr;
    void* shared_pixels = MAP_FAILED;
    size_t shared_bytes = 0;
    int shared_fd = -1;
    void* library = nullptr;
    decltype(&NvMOT_DeInit) deinit = nullptr;
    try {
        require(argc == 5 || argc == 6, "usage: nvdcf_bridge LIB CONFIG WIDTH HEIGHT [RGB_SHM]");
        const bool shared_input = argc == 6;
        if (shared_input) {
            shared_fd = shm_open(argv[5], O_RDONLY, 0);
            require(shared_fd >= 0, "RGB shm open failed");
            struct stat st{}; require(fstat(shared_fd, &st) == 0, "RGB shm stat failed");
            shared_bytes = st.st_size;
            shared_pixels = mmap(nullptr, shared_bytes, PROT_READ, MAP_SHARED, shared_fd, 0);
            require(shared_pixels != MAP_FAILED, "RGB shm map failed");
            NvBufSurfTransformConfigParams transform_config{};
            transform_config.compute_mode = NvBufSurfTransformCompute_GPU;
            transform_config.gpu_id = 0;
            require(NvBufSurfTransformSetSessionParams(&transform_config) == NvBufSurfTransformError_Success, "transform session failed");
        }
        const unsigned width = std::stoul(argv[3]), height = std::stoul(argv[4]);
        require(width && height && width <= 4096 && height <= 4096, "invalid tracker resolution");
        library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
        if (!library) throw std::runtime_error(dlerror());
        auto query = reinterpret_cast<decltype(&NvMOT_Query)>(dlsym(library, "NvMOT_Query"));
        auto init = reinterpret_cast<decltype(&NvMOT_Init)>(dlsym(library, "NvMOT_Init"));
        auto process = reinterpret_cast<decltype(&NvMOT_Process)>(dlsym(library, "NvMOT_Process"));
        deinit = reinterpret_cast<decltype(&NvMOT_DeInit)>(dlsym(library, "NvMOT_DeInit"));
        require(query && init && process && deinit, "missing NvMOT symbols");
        // DeepStream 9.1 queries a live context (unlike older NvMOT releases).
        NvMOTPerTransformBatchConfig transform{};
        transform.maxWidth = width; transform.maxHeight = height;
        transform.maxPitch = width * 8; transform.maxSize = transform.maxPitch * height;
        NvMOTConfig config{};
        config.computeConfig = NVMOTCOMP_GPU; config.maxStreams = 1; config.maxBufSurfAddrSize = 1;
        config.numTransforms = 1; config.perTransformBatchConfig = &transform;
        config.customConfigFilePath = argv[2]; config.customConfigFilePathSize = strlen(argv[2]);
        NvMOTConfigResponse response{};
        require(init(&config, &context, &response) == NvMOTStatus_OK, "NvMOT_Init failed");
        NvMOTQuery capabilities{};
        capabilities.contextHandle = context;
        capabilities.colorFormats[0] = NVBUF_COLOR_FORMAT_RGBA;
        require(query(strlen(argv[2]), argv[2], &capabilities) == NvMOTStatus_OK, "NvMOT_Query failed");
        require(capabilities.numTransforms == 1 && capabilities.colorFormats[0] == NVBUF_COLOR_FORMAT_RGBA,
                "NvDCF must request one RGBA transform");
        NvBufSurfaceCreateParams allocation{};
        allocation.gpuId = 0; allocation.width = width; allocation.height = height;
        allocation.layout = NVBUF_LAYOUT_PITCH; allocation.colorFormat = NVBUF_COLOR_FORMAT_RGBA;
        allocation.memType = capabilities.memType;
        require(NvBufSurfaceCreate(&surface, 1, &allocation) == 0, "NvBufSurfaceCreate failed");
        surface->numFilled = 1;
        if (shared_input) {
            NvBufSurfaceCreateParams prep{};
            prep.gpuId=0;prep.width=width;prep.height=height;
            prep.layout=NVBUF_LAYOUT_PITCH;prep.colorFormat=NVBUF_COLOR_FORMAT_RGBA;
            prep.memType=NVBUF_MEM_CUDA_DEVICE;
            require(NvBufSurfaceCreate(&rgba_surface,1,&prep)==0,"GPU RGBA allocation failed");
            rgba_surface->numFilled=1;
        }
        auto& buffer = surface->surfaceList[0];
        uint32_t magic = 0x4e564443;
        write_value(magic, protocol); fflush(protocol);
        std::vector<unsigned char> pixels(width * height * 4);
        const unsigned capacity = capabilities.maxTargetsPerStream ? capabilities.maxTargetsPerStream : 150;
        require(capacity <= 10000, "unexpected tracker capacity");
        std::vector<NvMOTTrackedObj> output(capacity);
        while (true) {
            // frame number, detectionDone, count; then RGBA and count*(xywh,score).
            uint32_t header[5]{};
            if (!read_all(header, (shared_input ? 5 : 3) * sizeof(uint32_t))) break;
            require(header[2] <= capacity, "too many detections");
            if (!shared_input) require(read_all(pixels.data(), pixels.size()), "truncated image");
            else require(header[3] && header[4] && size_t(header[3])*header[4]*3 <= shared_bytes,
                         "RGB shm image exceeds capacity");
            std::vector<NvMOTObjToTrack> detections(header[2]);
            for (unsigned i = 0; i < detections.size(); ++i) {
                float values[5];
                require(read_all(values, sizeof(values)), "truncated detections");
                auto& object = detections[i];
                object.classId = 0; object.bbox = {values[0],values[1],values[2],values[3]};
                object.confidence = values[4]; object.doTracking = true;
                object.pPreservedData = reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
            }
            if (shared_input) {
                if (!input_surface || input_surface->surfaceList[0].width != header[3]
                                   || input_surface->surfaceList[0].height != header[4]) {
                    if (input_surface) NvBufSurfaceDestroy(input_surface);
                    // RGBA stays at tracker resolution and lives until shutdown.
                    input_surface = nullptr;
                    NvBufSurfaceCreateParams input_params{};
                    input_params.gpuId = 0; input_params.width = header[3]; input_params.height = header[4];
                    input_params.layout = NVBUF_LAYOUT_PITCH; input_params.colorFormat = NVBUF_COLOR_FORMAT_RGB;
                    input_params.memType = NVBUF_MEM_CUDA_DEVICE;
                    require(NvBufSurfaceCreate(&input_surface, 1, &input_params) == 0, "RGB surface allocation failed");
                    input_surface->numFilled = 1;
                }
                auto& src = input_surface->surfaceList[0];
                require(cudaMemcpy2D(src.dataPtr, src.pitch, shared_pixels, header[3]*3,
                                    header[3]*3, header[4], cudaMemcpyHostToDevice) == cudaSuccess,
                        "RGB upload failed");
                auto& rgba=rgba_surface->surfaceList[0];
                require(resize_rgb_rgba(src.dataPtr,header[3],header[4],src.pitch,
                                        rgba.dataPtr,width,height,rgba.pitch)==0,"CUDA RGB preparation failed");
                NvBufSurfTransformRect src_rect{}, dst_rect{};
                src_rect.width=dst_rect.width=width;src_rect.height=dst_rect.height=height;
                NvBufSurfTransformParams params{};
                params.src_rect = &src_rect; params.dst_rect = &dst_rect;
                params.transform_flag = NVBUFSURF_TRANSFORM_FILTER;
                params.transform_filter = NvBufSurfTransformInter_Bilinear;
                require(NvBufSurfTransform(rgba_surface, surface, &params) == NvBufSurfTransformError_Success,
                        "GPU RGB resize/convert failed");
            } else if (surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
                require(NvBufSurfaceMap(surface, 0, 0, NVBUF_MAP_WRITE) == 0, "surface map failed");
                auto dest = static_cast<unsigned char*>(buffer.mappedAddr.addr[0]);
                for (unsigned y=0; y<height; ++y)
                    memcpy(dest + y*buffer.pitch, pixels.data()+y*width*4, width*4);
                require(NvBufSurfaceSyncForDevice(surface, 0, 0) == 0, "surface sync failed");
                NvBufSurfaceUnMap(surface, 0, 0);
            } else {
                require(cudaMemcpy2D(buffer.dataPtr, buffer.pitch, pixels.data(), width*4,
                    width*4, height, cudaMemcpyHostToDevice) == cudaSuccess, "RGBA upload failed");
            }
            NvBufSurfaceParams* buffers[] = {&buffer};
            NvMOTFrame frame{};
            frame.streamID = 1; frame.seq_index = 0; frame.frameNum = header[0];
            frame.timeMeta.frameNum = header[0]; frame.timeMeta.localTimestamp = NvDsCurrentTimestampNs();
            frame.srcFrameWidth = width; frame.srcFrameHeight = height; frame.doTracking = true;
            frame.numBuffers = 1; frame.bufferList = buffers;
            frame.objectsIn = {header[1] != 0, detections.data(), header[2], header[2]};
            NvMOTProcessParams params{1, &frame};
            NvMOTTrackedObjList list{};
            list.streamID = 1; list.list = output.data(); list.numAllocated = capacity;
            NvMOTTrackedObjBatch batch{};
            batch.list = &list; batch.numAllocated = 1;
            require(process(context, &params, &batch) == NvMOTStatus_OK, "NvMOT_Process failed");
            uint32_t count = batch.numFilled && list.valid ? list.numFilled : 0;
            require(count <= capacity, "invalid output count");
            write_value(count, protocol);
            for (unsigned i=0; i<count; ++i) {
                const auto& object = output[i];
                int32_t detection = object.associatedObjectIn
                    ? static_cast<int32_t>(reinterpret_cast<uintptr_t>(object.associatedObjectIn->pPreservedData))-1 : -1;
                write_value(object.trackingId, protocol);
                float values[] = {object.bbox.x, object.bbox.y, object.bbox.width,
                                  object.bbox.height, object.confidence};
                require(fwrite(values, sizeof(values), 1, protocol) == 1, "protocol write failed");
                write_value(detection, protocol);
            }
            require(fflush(protocol) == 0, "protocol flush failed");
        }
        deinit(context); context = nullptr;
        if (input_surface) NvBufSurfaceDestroy(input_surface);
        if (rgba_surface) NvBufSurfaceDestroy(rgba_surface);
        if (shared_pixels != MAP_FAILED) munmap(shared_pixels, shared_bytes);
        if (shared_fd >= 0) close(shared_fd);
        NvBufSurfaceDestroy(surface); surface = nullptr;
        dlclose(library);
        fclose(protocol);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "NvDCF bridge: %s\n", error.what());
        if (context && deinit) deinit(context);
        if (input_surface) NvBufSurfaceDestroy(input_surface);
        if (rgba_surface) NvBufSurfaceDestroy(rgba_surface);
        if (shared_pixels != MAP_FAILED) munmap(shared_pixels, shared_bytes);
        if (shared_fd >= 0) close(shared_fd);
        if (surface) NvBufSurfaceDestroy(surface);
        return 1;
    }
}
