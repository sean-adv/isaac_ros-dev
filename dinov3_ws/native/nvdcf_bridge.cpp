// Isolated DeepStream NvMOT bridge. stdout is exclusively the binary protocol.
#include <nvdstracker.h>
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

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
    void* library = nullptr;
    decltype(&NvMOT_DeInit) deinit = nullptr;
    try {
        require(argc == 5, "usage: nvdcf_bridge LIB CONFIG WIDTH HEIGHT");
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
        auto& buffer = surface->surfaceList[0];
        uint32_t magic = 0x4e564443;
        write_value(magic, protocol); fflush(protocol);
        std::vector<unsigned char> pixels(width * height * 4);
        const unsigned capacity = capabilities.maxTargetsPerStream ? capabilities.maxTargetsPerStream : 150;
        require(capacity <= 10000, "unexpected tracker capacity");
        std::vector<NvMOTTrackedObj> output(capacity);
        while (true) {
            // frame number, detectionDone, count; then RGBA and count*(xywh,score).
            uint32_t header[3];
            if (!read_all(header, sizeof(header))) break;
            require(header[2] <= capacity, "too many detections");
            require(read_all(pixels.data(), pixels.size()), "truncated image");
            std::vector<NvMOTObjToTrack> detections(header[2]);
            for (unsigned i = 0; i < detections.size(); ++i) {
                float values[5];
                require(read_all(values, sizeof(values)), "truncated detections");
                auto& object = detections[i];
                object.classId = 0; object.bbox = {values[0],values[1],values[2],values[3]};
                object.confidence = values[4]; object.doTracking = true;
                object.pPreservedData = reinterpret_cast<void*>(static_cast<uintptr_t>(i + 1));
            }
            if (surface->memType == NVBUF_MEM_SURFACE_ARRAY) {
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
        NvBufSurfaceDestroy(surface); surface = nullptr;
        dlclose(library);
        fclose(protocol);
        return 0;
    } catch (const std::exception& error) {
        fprintf(stderr, "NvDCF bridge: %s\n", error.what());
        if (context && deinit) deinit(context);
        if (surface) NvBufSurfaceDestroy(surface);
        return 1;
    }
}
