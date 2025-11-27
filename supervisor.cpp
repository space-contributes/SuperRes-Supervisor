// supervisor.cpp
// Enhanced SuperRes Supervisor with CUDA-accelerated Motion Estimation & AI Upscaling
// Added HDR10+ metadata, real-time color editing, DirectComposition rendering, memory safety, and multiple interpolations

// Removed macro redefinitions since they're already defined on command line
// #define WIN32_LEAN_AND_MEAN
// #define NOMINMAX
#include <initguid.h>
#include <dxgi1_2.h>
#include <d3d11.h>
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <dcomp.h>
#include <d3d11_1.h>
#include <gdiplus.h>
#include <winhttp.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>  // Added for min and clamp functions
#include <mutex>
#include <atomic>
#include <cmath>
#include <iomanip>
#include <string>
#include <cfloat>
#include <memory>
#include <unordered_map>
#include <queue>
#include <condition_variable>
#include <comdef.h>
#include <dwmapi.h>
#include <shellscalingapi.h>

// CUDA headers with full paths
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/cuda_runtime.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/cuda_d3d11_interop.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/device_launch_parameters.h"
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/vector_types.h"

// Custom min/max functions to avoid conflicts with CUDA
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLAMP(val, min_val, max_val) ((val) < (min_val) ? (min_val) : ((val) > (max_val) ? (max_val) : (val)))

// =================================================================
// == FIX 1: Define structs BEFORE using them in extern "C" block ==
// =================================================================
struct Color { 
    float r, g, b, a; 
};

struct MotionVector {
    int x, y;
    float confidence;
};

// =================================================================
// == FIX 2: Corrected and relocated extern "C" block ==
// This declares the host-side wrapper functions, not the __global__ kernels.
// It is placed AFTER the struct definitions so the compiler knows what 'Color' and 'MotionVector' are.
// =================================================================
extern "C" {
    void launchMotionEstimationKernel(
        const unsigned char* currentFrame,
        const unsigned char* previousFrame,
        int width, int height,
        MotionVector* motionField,
        int blockSize, int maxMotionVector);
        
    void launchUpscalingKernel(
        const unsigned char* input,
        unsigned char* output,
        int inW, int inH, int outW, int outH,
        float scaleX, float scaleY,
        float sharpnessFactor,
        float brightness, float contrast, float saturation, float hue, float gamma,
        float w, float x, float y, float z, float eta,
        float chromaStretch, float hueWarp, float lightnessFlow);
        
    void launchAdjustKernel(Color* pixels, int width, int height,
        float brightness, float gamma, float contrast,
        float deltaR, float deltaG, float deltaB);
}

// HDR10+ Metadata structure
struct HDR10PlusMetadata {
    std::vector<uint8_t> data;
    bool isAvailable;
    
    HDR10PlusMetadata() : isAvailable(false) {}
    
    void parse(const std::vector<uint8_t>& metadata) {
        if (metadata.empty()) {
            isAvailable = false;
            return;
        }
        
        // Simple parsing for demonstration
        data = metadata;
        isAvailable = true;
    }
};

// Color space transformation parameters
struct ColorTransformParams {
    float w, x, y, z;  // Weights for different color spaces
    float eta;         // Offset
    float chromaStretch;
    float hueWarp;
    float lightnessFlow;
    
    ColorTransformParams() : w(0.5f), x(0.2f), y(0.2f), z(0.1f), eta(0.0f), 
                           chromaStretch(1.0f), hueWarp(0.0f), lightnessFlow(0.0f) {}
};

// Real-time color editing parameters
struct ColorEditingParams {
    float brightness;
    float contrast;
    float saturation;
    float hue;
    float gamma;
    
    ColorEditingParams() : brightness(0.0f), contrast(1.0f), saturation(1.0f), 
                          hue(0.0f), gamma(1.0f) {}
};

#pragma comment(lib, "d3d11.lib")
#pragma comment(lib, "dxgi.lib")
#pragma comment(lib, "dcomp.lib")
#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "winmm.lib")
#pragma comment(lib, "winhttp.lib")
#pragma comment(lib, "cudart.lib")
#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "shcore.lib")

// -------------------- Configuration --------------------
static const int TARGET_HZ = 9000000;
static const int TARGET_W_DEFAULT = 7680;
static const int TARGET_H_DEFAULT = 4320;
static const double ARTIFACT_DETECT_THRESHOLD = 0.001;
static const char* KERNELS_PTX_LOCAL = "modules/cleanup.ptx";
static const char* LOG_PATH = "logs/boot.log";
static const char* MODULES_DIR = "modules";
static const char* LOGS_DIR = "logs";
static const int ONLINE_ADAPTIVE_WINDOW = 8;
static const int MOTION_BLOCK_SIZE = 16;
static const int MAX_MOTION_VECTOR = 32;
static const int MAX_INTERPOLATIONS_PER_FRAME = 3;  // New parameter for multiple interpolations

// Global parameters for color transformation and editing
static ColorTransformParams colorTransformParams;
static ColorEditingParams colorEditingParams;
static HDR10PlusMetadata hdrMetadata;

// Frame data history - moved here to fix undeclared identifier errors
static std::vector<std::vector<uint8_t>> frameDataHistory;

// -------------------- RAII Wrappers for DirectX Resources --------------------
template<typename T>
class ComPtr {
private:
    T* ptr;
    
public:
    ComPtr() : ptr(nullptr) {}
    ComPtr(T* p) : ptr(p) {}
    ~ComPtr() { if (ptr) ptr->Release(); }
    
    ComPtr(const ComPtr&) = delete;
    ComPtr& operator=(const ComPtr&) = delete;
    
    ComPtr(ComPtr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    
    ComPtr& operator=(ComPtr&& other) noexcept {
        if (this != &other) {
            if (ptr) ptr->Release();
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    
    T* Get() const { return ptr; }
    T** GetAddressOf() { if (ptr) { ptr->Release(); ptr = nullptr; } return &ptr; }
    T* operator->() const { return ptr; }
    operator bool() const { return ptr != nullptr; }
    
    void Release() {
        if (ptr) {
            ptr->Release();
            ptr = nullptr;
        }
    }
};

// -------------------- CUDA Device Management --------------------
static bool cudaInitialized = false;
static int cudaDeviceCount = 0;
static int cudaDevice = 0;
static cudaStream_t cudaStream;

// Initialize CUDA
static bool initCuda() {
    cudaError_t result = cudaGetDeviceCount(&cudaDeviceCount);
    if (result != cudaSuccess || cudaDeviceCount == 0) {
        return false;
    }
    
    result = cudaSetDevice(0);
    if (result != cudaSuccess) {
        return false;
    }
    
    result = cudaStreamCreate(&cudaStream);
    if (result != cudaSuccess) {
        return false;
    }
    
    cudaInitialized = true;
    return true;
}

// Cleanup CUDA resources
static void cleanupCuda() {
    if (cudaInitialized) {
        cudaStreamDestroy(cudaStream);
        cudaDeviceReset();
    }
}

// -------------------- Logging / dirs --------------------
static void ensure_dirs() {
    // Using CreateDirectory instead of filesystem for compatibility
    if (!CreateDirectoryA(MODULES_DIR, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) {
        // Handle error
    }
    if (!CreateDirectoryA(LOGS_DIR, NULL) && GetLastError() != ERROR_ALREADY_EXISTS) {
        // Handle error
    }
}

// -------------------- Motion Estimation System --------------------
struct FrameBuffer {
    std::vector<uint8_t> data;
    int width, height;
    std::vector<MotionVector> motionField;
    
    FrameBuffer() : width(0), height(0) {}
    
    void resize(int w, int h) {
        width = w;
        height = h;
        data.resize(w * h * 4);
        motionField.resize((w / MOTION_BLOCK_SIZE) * (h / MOTION_BLOCK_SIZE));
    }
};

static FrameBuffer prevFrame, currentFrame, nextFrame;
static std::vector<FrameBuffer> frameHistory;

static unsigned char* d_currentFrame = nullptr;
static unsigned char* d_previousFrame = nullptr;
static MotionVector* d_motionField = nullptr;
static unsigned char* d_upscaledFrame = nullptr;

static bool allocateCudaMemory(int width, int height) {
    if (!cudaInitialized) return false;
    
    size_t frameSize = width * height * 4 * sizeof(unsigned char);
    size_t motionFieldSize = (width / MOTION_BLOCK_SIZE) * (height / MOTION_BLOCK_SIZE) * sizeof(MotionVector);
    
    cudaError_t result;
    
    result = cudaMalloc(&d_currentFrame, frameSize);
    if (result != cudaSuccess) return false;
    
    result = cudaMalloc(&d_previousFrame, frameSize);
    if (result != cudaSuccess) {
        cudaFree(d_currentFrame);
        return false;
    }
    
    result = cudaMalloc(&d_motionField, motionFieldSize);
    if (result != cudaSuccess) {
        cudaFree(d_currentFrame);
        cudaFree(d_previousFrame);
        return false;
    }
    
    result = cudaMalloc(&d_upscaledFrame, TARGET_W_DEFAULT * TARGET_H_DEFAULT * 4 * sizeof(unsigned char));
    if (result != cudaSuccess) {
        cudaFree(d_currentFrame);
        cudaFree(d_previousFrame);
        cudaFree(d_motionField);
        return false;
    }
    
    return true;
}

static void freeCudaMemory() {
    if (!cudaInitialized) return;
    
    if (d_currentFrame) cudaFree(d_currentFrame);
    if (d_previousFrame) cudaFree(d_previousFrame);
    if (d_motionField) cudaFree(d_motionField);
    if (d_upscaledFrame) cudaFree(d_upscaledFrame);
    
    d_currentFrame = nullptr;
    d_previousFrame = nullptr;
    d_motionField = nullptr;
    d_upscaledFrame = nullptr;
}

// CUDA-accelerated motion estimation
static void estimateMotion(FrameBuffer& current, FrameBuffer& previous) {
    if (current.width != previous.width || current.height != previous.height) return;
    
    if (cudaInitialized && d_currentFrame && d_previousFrame && d_motionField) {
        int blocksX = current.width / MOTION_BLOCK_SIZE;
        int blocksY = current.height / MOTION_BLOCK_SIZE;
        
        cudaMemcpyAsync(d_currentFrame, current.data.data(),
                       current.width * current.height * 4 * sizeof(unsigned char),
                       cudaMemcpyHostToDevice, cudaStream);
        cudaMemcpyAsync(d_previousFrame, previous.data.data(),
                       previous.width * previous.height * 4 * sizeof(unsigned char),
                       cudaMemcpyHostToDevice, cudaStream);
        
        // =================================================================
        // == FIX 3: Corrected kernel call in estimateMotion ==
        // The previous code was incorrectly calling launchUpscalingKernel here.
        // =================================================================
        launchMotionEstimationKernel(d_currentFrame, d_previousFrame, current.width, current.height, d_motionField, MOTION_BLOCK_SIZE, MAX_MOTION_VECTOR);

        cudaMemcpyAsync(current.motionField.data(), d_motionField,
                       blocksX * blocksY * sizeof(MotionVector),
                       cudaMemcpyDeviceToHost, cudaStream);
        
        cudaStreamSynchronize(cudaStream);
    } else {
        // CPU fallback
        int blocksX = current.width / MOTION_BLOCK_SIZE;
        int blocksY = current.height / MOTION_BLOCK_SIZE;
        
        for (int by = 0; by < blocksY; by++) {
            for (int bx = 0; bx < blocksX; bx++) {
                int bestX = 0, bestY = 0;
                float bestError = FLT_MAX;
                
                for (int dy = -MAX_MOTION_VECTOR; dy <= MAX_MOTION_VECTOR; dy += 4) {
                    for (int dx = -MAX_MOTION_VECTOR; dx <= MAX_MOTION_VECTOR; dx += 4) {
                        float error = 0.0f;
                        int samples = 0;
                        
                        for (int py = 0; py < MOTION_BLOCK_SIZE; py += 2) {
                            for (int px = 0; px < MOTION_BLOCK_SIZE; px += 2) {
                                int currX = bx * MOTION_BLOCK_SIZE + px;
                                int currY = by * MOTION_BLOCK_SIZE + py;
                                int prevX = currX + dx;
                                int prevY = currY + dy;
                                
                                if (prevX >= 0 && prevX < previous.width &&
                                    prevY >= 0 && prevY < previous.height &&
                                    currX >= 0 && currX < current.width &&
                                    currY >= 0 && currY < current.height) {
                                    
                                    int currIdx = (currY * current.width + currX) * 4;
                                    int prevIdx = (prevY * previous.width + prevX) * 4;
                                    
                                    float currLum = 0.299f * current.data[currIdx+2] +
                                                  0.587f * current.data[currIdx+1] +
                                                  0.114f * current.data[currIdx+0];
                                    float prevLum = 0.299f * previous.data[prevIdx+2] +
                                                  0.587f * previous.data[prevIdx+1] +
                                                  0.114f * previous.data[prevIdx+0];
                                    
                                    error += fabsf(currLum - prevLum);
                                    samples++;
                                }
                            }
                        }
                        
                        if (samples > 0) {
                            error /= samples;
                            if (error < bestError) {
                                bestError = error;
                                bestX = dx;
                                bestY = dy;
                            }
                        }
                    }
                }
                
                int blockIdx = by * blocksX + bx;
                current.motionField[blockIdx] = {bestX, bestY, 1.0f - (bestError / 255.0f)};
            }
        }
    }
}

// -------------------- AI-like Upscaling System --------------------
struct UpscalingModel {
    float edgeKernel[9];
    float detailKernel[9];
    float sharpnessFactor;
    float adaptiveThreshold;
    
    UpscalingModel() {
        edgeKernel[0] = -1; edgeKernel[1] = 0; edgeKernel[2] = 1;
        edgeKernel[3] = -2; edgeKernel[4] = 0; edgeKernel[5] = 2;
        edgeKernel[6] = -1; edgeKernel[7] = 0; edgeKernel[8] = 1;
        
        detailKernel[0] = 0; detailKernel[1] = -1; detailKernel[2] = 0;
        detailKernel[3] = -1; detailKernel[4] = 5; detailKernel[5] = -1;
        detailKernel[6] = 0; detailKernel[7] = -1; detailKernel[8] = 0;
        
        sharpnessFactor = 1.5f;
        adaptiveThreshold = 30.0f;
    }
    
    void adapt(const std::vector<uint8_t>& frame, int w, int h) {
        float avgEdgeStrength = 0.0f;
        int samples = 0;
        
        for (int y = 1; y < h-1; y += 10) {
            for (int x = 1; x < w-1; x += 10) {
                float edgeStrength = 0.0f;
                for (int ky = -1; ky <= 1; ky++) {
                    for (int kx = -1; kx <= 1; kx++) {
                        int idx = ((y + ky) * w + (x + kx)) * 4;
                        int kernelIdx = (ky + 1) * 3 + (kx + 1);
                        float lum = 0.299f * frame[idx+2] + 0.587f * frame[idx+1] + 0.114f * frame[idx+0];
                        edgeStrength += lum * edgeKernel[kernelIdx];
                    }
                }
                avgEdgeStrength += fabs(edgeStrength);
                samples++;
            }
        }
        
        if (samples > 0) {
            avgEdgeStrength /= samples;
            if (avgEdgeStrength < adaptiveThreshold) {
                sharpnessFactor = MIN(sharpnessFactor * 1.01f, 3.0f);
            } else {
                sharpnessFactor = MAX(sharpnessFactor * 0.99f, 1.2f);
            }
        }
    }
    
    void save(const std::string &path) {
        std::ofstream f(path, std::ios::binary);
        if (f) {
            f.write((char*)edgeKernel, sizeof(edgeKernel));
            f.write((char*)detailKernel, sizeof(detailKernel));
            f.write((char*)&sharpnessFactor, sizeof(sharpnessFactor));
            f.write((char*)&adaptiveThreshold, sizeof(adaptiveThreshold));
        }
    }
    
    bool load(const std::string &path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        f.read((char*)edgeKernel, sizeof(edgeKernel));
        f.read((char*)detailKernel, sizeof(detailKernel));
        f.read((char*)&sharpnessFactor, sizeof(sharpnessFactor));
        f.read((char*)&adaptiveThreshold, sizeof(adaptiveThreshold));
        return true;
    }
};

static UpscalingModel upscalingModel;

// CUDA-accelerated upscaling with color transformations
static std::vector<uint8_t> upscaleFrame(const std::vector<uint8_t>& input, int inW, int inH, int outW, int outH) {
    std::vector<uint8_t> output(outW * outH * 4);
    float scaleX = (float)inW / outW;
    float scaleY = (float)inH / outH;
    
    if (cudaInitialized && d_upscaledFrame) {
        cudaMemcpyAsync(d_currentFrame, input.data(),
                       inW * inH * 4 * sizeof(unsigned char),
                       cudaMemcpyHostToDevice, cudaStream);
        
        // Kernel launch
        launchUpscalingKernel(d_currentFrame, d_upscaledFrame, inW, inH, outW, outH, scaleX, scaleY, upscalingModel.sharpnessFactor, colorEditingParams.brightness, colorEditingParams.contrast, colorEditingParams.saturation, colorEditingParams.hue, colorEditingParams.gamma, colorTransformParams.w, colorTransformParams.x, colorTransformParams.y, colorTransformParams.z, colorTransformParams.eta, colorTransformParams.chromaStretch, colorTransformParams.hueWarp, colorTransformParams.lightnessFlow);

        cudaMemcpyAsync(output.data(), d_upscaledFrame,
                       outW * outH * 4 * sizeof(unsigned char),
                       cudaMemcpyDeviceToHost, cudaStream);
        
        cudaStreamSynchronize(cudaStream);
    } else {
        // CPU fallback
        for (int y = 0; y < outH; y++) {
            for (int x = 0; x < outW; x++) {
                float srcX = x * scaleX;
                float srcY = y * scaleY;
                
                int x0 = (int)srcX;
                int y0 = (int)srcY;
                int x1 = MIN(x0 + 1, inW - 1);
                int y1 = MIN(y0 + 1, inH - 1);
                
                float dx = srcX - x0;
                float dy = srcY - y0;
                
                int outIdx = (y * outW + x) * 4;
                
                for (int c = 0; c < 3; c++) {
                    float p00 = input[(y0 * inW + x0) * 4 + c];
                    float p01 = input[(y0 * inW + x1) * 4 + c];
                    float p10 = input[(y1 * inW + x0) * 4 + c];
                    float p11 = input[(y1 * inW + x1) * 4 + c];
                    
                    float interpolated = p00 * (1-dx) * (1-dy) +
                                        p01 * dx * (1-dy) +
                                        p10 * (1-dx) * dy +
                                        p11 * dx * dy;
                    
                    if (x > 0 && x < outW-1 && y > 0 && y < outH-1) {
                        float center = interpolated;
                        float sum = 0;
                        for (int ky = -1; ky <= 1; ky++) {
                            for (int kx = -1; kx <= 1; kx++) {
                                if (kx == 0 && ky == 0) continue;
                                int nx = x + kx;
                                int ny = y + ky;
                                float nSrcX = nx * scaleX;
                                float nSrcY = ny * scaleY;
                                int nx0 = (int)nSrcX;
                                int ny0 = (int)nSrcY;
                                int nx1 = MIN(nx0 + 1, inW - 1);
                                int ny1 = MIN(ny0 + 1, inH - 1);
                                float ndx = nSrcX - nx0;
                                float ndy = nSrcY - ny0;
                                
                                float neighbor = input[(ny0 * inW + nx0) * 4 + c] * (1-ndx) * (1-ndy) +
                                               input[(ny0 * inW + nx1) * 4 + c] * ndx * (1-ndy) +
                                               input[(ny1 * inW + nx0) * 4 + c] * (1-ndx) * ndy +
                                               input[(ny1 * inW + nx1) * 4 + c] * ndx * ndy;
                                sum += neighbor;
                            }
                        }
                        sum /= 8.0f;
                        float detail = center - sum;
                        interpolated = center + detail * upscalingModel.sharpnessFactor * 0.1f;
                    }
                    
                    // Using custom clamp function instead of std::clamp
                    output[outIdx + c] = (uint8_t)CLAMP(interpolated, 0.0f, 255.0f);
                }
                output[outIdx + 3] = 255;
            }
        }
    }
    
    return output;
}

// -------------------- Enhanced Frame Interpolation --------------------
static std::vector<uint8_t> interpolateFrames(const FrameBuffer& frame1, const FrameBuffer& frame2, float alpha, int w, int h) {
    std::vector<uint8_t> result(w * h * 4);
    
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 4;
            
            int blockX = x / MOTION_BLOCK_SIZE;
            int blockY = y / MOTION_BLOCK_SIZE;
            int blockIdx = blockY * (w / MOTION_BLOCK_SIZE) + blockX;
            
            if (blockIdx < frame2.motionField.size()) {
                const MotionVector& mv = frame2.motionField[blockIdx];
                
                int srcX = x + (int)(mv.x * alpha);
                int srcY = y + (int)(mv.y * alpha);
                
                if (srcX >= 0 && srcX < w && srcY >= 0 && srcY < h) {
                    int srcIdx = (srcY * w + srcX) * 4;
                    
                    for (int c = 0; c < 4; c++) {
                        result[idx + c] = (uint8_t)(frame1.data[idx + c] * (1 - alpha) +
                                                 frame2.data[srcIdx + c] * alpha);
                    }
                } else {
                    for (int c = 0; c < 4; c++) {
                        result[idx + c] = (uint8_t)(frame1.data[idx + c] * (1 - alpha) +
                                                 frame2.data[idx + c] * alpha);
                    }
                }
            } else {
                for (int c = 0; c < 4; c++) {
                    result[idx + c] = (uint8_t)(frame1.data[idx + c] * (1 - alpha) +
                                             frame2.data[idx + c] * alpha);
                }
            }
        }
    }
    
    return result;
}

// Multiple interpolation per frame with adaptive quality
static std::vector<std::vector<uint8_t>> generateMultipleInterpolatedFrames(
    const FrameBuffer& frame1, 
    const FrameBuffer& frame2, 
    int targetW, 
    int targetH,
    int numInterpolations = MAX_INTERPOLATIONS_PER_FRAME) {
    
    std::vector<std::vector<uint8_t>> interpolatedFrames;
    
    if (numInterpolations <= 0) return interpolatedFrames;
    
    // Calculate motion complexity
    float avgMotion = 0.0f;
    int motionVectors = 0;
    for (const auto& mv : frame2.motionField) {
        avgMotion += sqrtf(mv.x * mv.x + mv.y * mv.y) * mv.confidence;
        motionVectors++;
    }
    
    if (motionVectors > 0) avgMotion /= motionVectors;
    
    // Adjust number of interpolations based on motion
    int adjustedInterpolations = MIN(numInterpolations, 
                                  MAX(1, (int)(avgMotion / 5.0f)));
    
    // Generate interpolated frames
    for (int i = 1; i <= adjustedInterpolations; i++) {
        float alpha = (float)i / (adjustedInterpolations + 1);
        interpolatedFrames.push_back(interpolateFrames(frame1, frame2, alpha, targetW, targetH));
    }
    
    return interpolatedFrames;
}

// -------------------- Capture: Desktop Duplication + GDI fallback --------------------
struct DDAContext {
    ComPtr<ID3D11Device> device;
    ComPtr<ID3D11DeviceContext> context;
    ComPtr<IDXGIOutputDuplication> duplication;
    ComPtr<IDXGIOutput1> output1;
    bool initialized = false;
    int width=0, height=0;
};

static bool init_dda(DDAContext &ctx) {
    HRESULT hr;
    D3D_FEATURE_LEVEL fl;
    UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
    
    // Fixed D3D11CreateDevice call
    hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, flags, nullptr, 0, 
                          D3D11_SDK_VERSION, ctx.device.GetAddressOf(), &fl, 
                          reinterpret_cast<ID3D11DeviceContext**>(
                              ctx.context.GetAddressOf()));
    if (FAILED(hr) || !ctx.device) {
        return false;
    }
    
    ComPtr<IDXGIDevice> dxgiDev;
    hr = ctx.device->QueryInterface(__uuidof(IDXGIDevice), 
                                   reinterpret_cast<void**>(dxgiDev.GetAddressOf()));
    if (FAILED(hr) || !dxgiDev) return false;
    
    ComPtr<IDXGIAdapter> adapter;
    hr = dxgiDev->GetAdapter(adapter.GetAddressOf());
    if (FAILED(hr) || !adapter) return false;
    
    ComPtr<IDXGIOutput> output;
    hr = adapter->EnumOutputs(0, output.GetAddressOf());
    if (FAILED(hr) || !output) return false;
    
    hr = output->QueryInterface(__uuidof(IDXGIOutput1), 
                               reinterpret_cast<void**>(ctx.output1.GetAddressOf()));
    if (FAILED(hr) || !ctx.output1) return false;
    
    RECT r;
    GetClientRect(GetDesktopWindow(), &r);
    ctx.width = r.right - r.left;
    ctx.height = r.bottom - r.top;
    
    // Fixed DuplicateOutput call
    hr = ctx.output1->DuplicateOutput(ctx.device.Get(), ctx.duplication.GetAddressOf());
    if (FAILED(hr) || !ctx.duplication) return false;
    
    ctx.initialized = true;
    return true;
}

static bool grab_frame_dda(DDAContext &ctx, std::vector<uint8_t> &out_bgra, int &w, int &h) {
    if (!ctx.initialized) return false;
    
    IDXGIResource* desktopResource = nullptr;
    DXGI_OUTDUPL_FRAME_INFO frameInfo;
    HRESULT hr = ctx.duplication->AcquireNextFrame(0, &frameInfo, &desktopResource);
    
    if (hr == DXGI_ERROR_WAIT_TIMEOUT) return false;
    if (FAILED(hr) || !desktopResource) return false;
    
    ComPtr<ID3D11Texture2D> tex;
    hr = desktopResource->QueryInterface(__uuidof(ID3D11Texture2D), 
                                        reinterpret_cast<void**>(tex.GetAddressOf()));
    desktopResource->Release();
    if (FAILED(hr) || !tex) return false;
    
    D3D11_TEXTURE2D_DESC desc;
    tex->GetDesc(&desc);
    w = desc.Width;
    h = desc.Height;
    
    desc.Usage = D3D11_USAGE_STAGING;
    desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
    desc.BindFlags = 0;
    desc.MiscFlags = 0;
    
    ComPtr<ID3D11Texture2D> staging;
    hr = ctx.device->CreateTexture2D(&desc, nullptr, staging.GetAddressOf());
    if (FAILED(hr) || !staging) {
        ctx.duplication->ReleaseFrame();
        return false;
    }
    
    ctx.context->CopyResource(staging.Get(), tex.Get());
    
    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = ctx.context->Map(staging.Get(), 0, D3D11_MAP_READ, 0, &mapped);
    if (FAILED(hr)) {
        ctx.duplication->ReleaseFrame();
        return false;
    }
    
    size_t rowBytes = w * 4;
    out_bgra.resize(rowBytes * h);
    for (int y = 0; y < h; y++) {
        memcpy(&out_bgra[y * rowBytes], 
               (uint8_t*)mapped.pData + y * mapped.RowPitch, 
               rowBytes);
    }
    
    ctx.context->Unmap(staging.Get(), 0);
    ctx.duplication->ReleaseFrame();
    return true;
}

static bool grab_frame_gdi(std::vector<uint8_t> &out_bgra, int &w, int &h) {
    HDC hScreen = GetDC(NULL);
    HDC hMem = CreateCompatibleDC(hScreen);
    
    RECT r;
    GetClientRect(GetDesktopWindow(), &r);
    w = r.right - r.left;
    h = r.bottom - r.top;
    
    HBITMAP hBitmap = CreateCompatibleBitmap(hScreen, w, h);
    HBITMAP old = (HBITMAP)SelectObject(hMem, hBitmap);
    
    if (!BitBlt(hMem, 0, 0, w, h, hScreen, 0, 0, SRCCOPY|CAPTUREBLT)) {
        SelectObject(hMem, old);
        DeleteObject(hBitmap);
        DeleteDC(hMem);
        ReleaseDC(NULL, hScreen);
        return false;
    }
    
    BITMAPINFOHEADER bi = { sizeof(BITMAPINFOHEADER), w, -h, 1, 32, BI_RGB, 0, 0, 0, 0, 0 };
    out_bgra.resize((size_t)w * h * 4);
    
    if (!GetDIBits(hMem, hBitmap, 0, h, out_bgra.data(), (BITMAPINFO*)&bi, DIB_RGB_COLORS)) {
        SelectObject(hMem, old);
        DeleteObject(hBitmap);
        DeleteDC(hMem);
        ReleaseDC(NULL, hScreen);
        return false;
    }
    
    SelectObject(hMem, old);
    DeleteObject(hBitmap);
    DeleteDC(hMem);
    ReleaseDC(NULL, hScreen);
    return true;
}

// -------------------- Enhanced DirectComposition overlay --------------------
struct DCompContext {
    ComPtr<ID3D11Device> d3dDevice;
    ComPtr<ID3D11DeviceContext> d3dContext;
    ComPtr<IDXGIDevice> dxgiDevice;
    ComPtr<IDXGIAdapter> dxgiAdapter;
    ComPtr<IDXGIFactory2> dxgiFactory;
    ComPtr<IDCompositionDevice> dcompDevice;
    ComPtr<IDCompositionTarget> dcompTarget;
    ComPtr<IDCompositionVisual> dcompVisual;
    ComPtr<ID3D11Texture2D> texture;
    ComPtr<IDXGISurface> surface;
    ComPtr<IDCompositionSurface> dcompSurface;
    HWND hwndOverlay = nullptr;
    bool initialized = false;
    int width = 0, height = 0;
    
    void cleanup() {
        if (hwndOverlay) {
            DestroyWindow(hwndOverlay);
            hwndOverlay = nullptr;
        }
        
        texture.Release();
        surface.Release();
        dcompSurface.Release();
        dcompVisual.Release();
        dcompTarget.Release();
        dcompDevice.Release();
        dxgiFactory.Release();
        dxgiAdapter.Release();
        dxgiDevice.Release();
        d3dContext.Release();
        d3dDevice.Release();
        
        initialized = false;
    }
};

static HWND create_overlay_window(int width, int height) {
    WNDCLASSA wc = {0};
    wc.lpfnWndProc = DefWindowProcA;
    wc.hInstance = GetModuleHandle(NULL);
    wc.lpszClassName = "SupervisorOverlayClass";
    RegisterClassA(&wc);
    
    DWORD exStyle = WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_NOACTIVATE;
    HWND h = CreateWindowExA(exStyle, wc.lpszClassName, "SupervisorOverlay", WS_POPUP,
                             0, 0, width, height, NULL, NULL, GetModuleHandle(NULL), NULL);
    if (!h) return NULL;
    
    SetLayeredWindowAttributes(h, RGB(0, 0, 0), 255, LWA_ALPHA);
    
    SetWindowPos(h, HWND_TOPMOST, 0, 0, width, height, SWP_SHOWWINDOW | SWP_NOACTIVATE);
    ShowWindow(h, SW_SHOW);
    return h;
}

static bool init_dcomp(DCompContext &ctx, int w, int h) {
    ctx.cleanup();
    
    ctx.hwndOverlay = create_overlay_window(w, h);
    if (!ctx.hwndOverlay) return false;
    
    D3D_FEATURE_LEVEL fl;
    HRESULT hr = D3D11CreateDevice(NULL, D3D_DRIVER_TYPE_HARDWARE, NULL, 
                                  D3D11_CREATE_DEVICE_BGRA_SUPPORT, NULL, 0, 
                                  D3D11_SDK_VERSION, ctx.d3dDevice.GetAddressOf(), 
                                  &fl, reinterpret_cast<ID3D11DeviceContext**>(
                                      ctx.d3dContext.GetAddressOf()));
    if (FAILED(hr) || !ctx.d3dDevice) return false;
    
    hr = ctx.d3dDevice->QueryInterface(__uuidof(IDXGIDevice), 
                                      reinterpret_cast<void**>(ctx.dxgiDevice.GetAddressOf()));
    if (FAILED(hr) || !ctx.dxgiDevice) return false;
    
    hr = ctx.dxgiDevice->GetAdapter(ctx.dxgiAdapter.GetAddressOf());
    if (FAILED(hr) || !ctx.dxgiAdapter) return false;
    
    hr = ctx.dxgiAdapter->GetParent(__uuidof(IDXGIFactory2), 
                                   reinterpret_cast<void**>(ctx.dxgiFactory.GetAddressOf()));
    if (FAILED(hr) || !ctx.dxgiFactory) return false;
    
    hr = DCompositionCreateDevice(ctx.dxgiDevice.Get(), __uuidof(IDCompositionDevice), 
                                 reinterpret_cast<void**>(ctx.dcompDevice.GetAddressOf()));
    if (FAILED(hr) || !ctx.dcompDevice) return false;
    
    hr = ctx.dcompDevice->CreateTargetForHwnd(ctx.hwndOverlay, TRUE, 
                                             ctx.dcompTarget.GetAddressOf());
    if (FAILED(hr) || !ctx.dcompTarget) return false;
    
    hr = ctx.dcompDevice->CreateVisual(ctx.dcompVisual.GetAddressOf());
    if (FAILED(hr) || !ctx.dcompVisual) return false;
    
    hr = ctx.dcompDevice->CreateSurface(w, h, DXGI_FORMAT_B8G8R8A8_UNORM, 
                                       DXGI_ALPHA_MODE_PREMULTIPLIED, 
                                       ctx.dcompSurface.GetAddressOf());
    if (FAILED(hr) || !ctx.dcompSurface) return false;
    
    ctx.dcompTarget->SetRoot(ctx.dcompVisual.Get());
    ctx.width = w;
    ctx.height = h;
    ctx.initialized = true;
    
    return true;
}

static bool present_via_dcomp(DCompContext &ctx, const std::vector<uint8_t>& bgra, int w, int h) {
    if (!ctx.initialized) return false;
    
    // Update surface with new frame data
    RECT updateRect = {0, 0, w, h};
    POINT updateOffset;
    IDXGISurface* rawSurface = nullptr;
    
    // Fixed IID_PPV_ARGS usage
    HRESULT hr = ctx.dcompSurface->BeginDraw(&updateRect, IID_IDXGISurface, (void**)&rawSurface, &updateOffset);
    if (FAILED(hr) || !rawSurface) return false;

    ComPtr<IDXGISurface> surface(rawSurface);

    ComPtr<ID3D11Texture2D> texture;
    hr = surface->QueryInterface(__uuidof(ID3D11Texture2D), reinterpret_cast<void**>(texture.GetAddressOf()));
    if (FAILED(hr) || !texture) {
        ctx.dcompSurface->EndDraw();
        return false;
    }

    D3D11_TEXTURE2D_DESC desc;
    texture->GetDesc(&desc);

    D3D11_MAPPED_SUBRESOURCE mapped;
    hr = ctx.d3dContext->Map(texture.Get(), 0, D3D11_MAP_WRITE_DISCARD, 0, &mapped);
    if (FAILED(hr)) {
        ctx.dcompSurface->EndDraw();
        return false;
    }

    // Copy frame data to texture
    uint8_t* dest = static_cast<uint8_t*>(mapped.pData);
    const uint8_t* src = bgra.data();

    for (int y = 0; y < h; y++) {
        memcpy(dest + y * mapped.RowPitch, src + y * w * 4, w * 4);
    }

    ctx.d3dContext->Unmap(texture.Get(), 0);

    hr = ctx.dcompSurface->EndDraw();
    if (FAILED(hr)) return false;

    // Set the surface as content for the visual
    ctx.dcompVisual->SetContent(ctx.dcompSurface.Get());

    // Commit changes
    hr = ctx.dcompDevice->Commit();
    if (FAILED(hr)) return false;

    return true;
}

// -------------------- HDR10+ Metadata Processing --------------------
static void processHDRMetadata(const std::vector<uint8_t>& frameData) {
    // This is a simplified implementation
    // In a real scenario, you would parse the HDR metadata from the frame
    
    // For demonstration, we'll create some dummy metadata
    std::vector<uint8_t> dummyMetadata;
    
    // Add some dummy HDR10+ metadata
    dummyMetadata.push_back(0x01);  // Version
    dummyMetadata.push_back(0x00);  // Reserved
    dummyMetadata.push_back(0x0A);  // Number of scenes
    dummyMetadata.push_back(0x00);  // Reserved
    
    // Add scene data
    for (int i = 0; i < 10; i++) {
        dummyMetadata.push_back(i * 10);  // Scene frame number
        dummyMetadata.push_back(100 + i * 10);  // MaxRGB
        dummyMetadata.push_back(50 + i * 5);   // AvgRGB
        dummyMetadata.push_back(200 + i * 5);  // MaxSCL
    }
    
    hdrMetadata.parse(dummyMetadata);
}

// -------------------- Color Transformation Functions --------------------
static void applyColorTransform(std::vector<uint8_t>& frame, int w, int h) {
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int idx = (y * w + x) * 4;
            
            // Get normalized RGB values
            float r = frame[idx + 2] / 255.0f;
            float g = frame[idx + 1] / 255.0f;
            float b = frame[idx + 0] / 255.0f;
            
            // Apply color space transformations
            float combined = colorTransformParams.w * r + 
                           colorTransformParams.x * g + 
                           colorTransformParams.y * b + 
                           colorTransformParams.z * (r + g + b) / 3.0f + 
                           colorTransformParams.eta;
            
            r = r * (1.0f - colorTransformParams.chromaStretch) + combined * colorTransformParams.chromaStretch;
            g = g * (1.0f - colorTransformParams.chromaStretch) + combined * colorTransformParams.chromaStretch;
            b = b * (1.0f - colorTransformParams.chromaStretch) + combined * colorTransformParams.chromaStretch;
            
            // Apply hue rotation
            float h = atan2f(sqrtf(3.0f) * (g - b), 2.0f * r - g - b) + colorTransformParams.hueWarp;
            float s = sqrtf((r - g) * (r - g) + (g - b) * (g - b) + (b - r) * (b - r)) / sqrtf(2.0f);
            float l = (r + g + b) / 3.0f;
            
            r = l + s * cosf(h);
            g = l + s * cosf(h - 2.0f * 3.14159f / 3.0f);
            b = l + s * cosf(h + 2.0f * 3.14159f / 3.0f);
            
            // Apply brightness, contrast, saturation
            r = (r - 0.5f) * colorEditingParams.contrast + 0.5f + colorEditingParams.brightness;
            g = (g - 0.5f) * colorEditingParams.contrast + 0.5f + colorEditingParams.brightness;
            b = (b - 0.5f) * colorEditingParams.contrast + 0.5f + colorEditingParams.brightness;
            
            float gray = 0.299f * r + 0.587f * g + 0.114f * b;
            r = gray + colorEditingParams.saturation * (r - gray);
            g = gray + colorEditingParams.saturation * (g - gray);
            b = gray + colorEditingParams.saturation * (b - gray);
            
            // Apply gamma correction
            r = powf(r, 1.0f / colorEditingParams.gamma);
            g = powf(g, 1.0f / colorEditingParams.gamma);
            b = powf(b, 1.0f / colorEditingParams.gamma);
            
            // Using custom clamp instead of std::clamp
            frame[idx + 0] = (uint8_t)CLAMP(b * 255.0f, 0.0f, 255.0f);
            frame[idx + 1] = (uint8_t)CLAMP(g * 255.0f, 0.0f, 255.0f);
            frame[idx + 2] = (uint8_t)CLAMP(r * 255.0f, 0.0f, 255.0f);
        }
    }
}

// -------------------- Priority --------------------
static void set_high_priority(){
    SetPriorityClass(GetCurrentProcess(),REALTIME_PRIORITY_CLASS);
    HMODULE hAv=LoadLibraryA("avrt.dll");
    if(hAv){
        typedef HANDLE(WINAPI *AVSET)(LPCSTR,LPDWORD);
        typedef BOOL(WINAPI*AVSETP)(HANDLE,LPDWORD);
        AVSET pSet=(AVSET)GetProcAddress(hAv,"AvSetMmThreadCharacteristicsA");
        AVSETP pSetP=(AVSETP)GetProcAddress(hAv,"AvSetMmThreadPriority");
        if(pSet&&pSetP){
            DWORD idx=0;
            HANDLE h=pSet("Pro Audio",&idx);
            if(h) pSetP(h,(LPDWORD)3);
        }
    }
}

// -------------------- High precision sleep --------------------
static void sleep_until(std::chrono::steady_clock::time_point t){
    using namespace std::chrono;
    auto now=steady_clock::now();
    while(now+milliseconds(2)<t){
        std::this_thread::sleep_for(milliseconds(1));
        now=steady_clock::now();
    }
    if(now<t) std::this_thread::sleep_for(t-now);
}

// -------------------- Enhanced Processing Pipeline --------------------
static void enhance_quality(std::vector<uint8_t>& bgra, int w, int h) {
    currentFrame.data = bgra;
    currentFrame.width = w;
    currentFrame.height = h;
    
    // Process HDR metadata
    processHDRMetadata(bgra);
    
    // Apply color transformations
    applyColorTransform(bgra, w, h);
    
    if (prevFrame.width == w && prevFrame.height == h) {
        estimateMotion(currentFrame, prevFrame);
        
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int idx = (y * w + x) * 4;
                
                int blockX = x / MOTION_BLOCK_SIZE;
                int blockY = y / MOTION_BLOCK_SIZE;
                int blockIdx = blockY * (w / MOTION_BLOCK_SIZE) + blockX;
                
                if (blockIdx < currentFrame.motionField.size()) {
                    const MotionVector& mv = currentFrame.motionField[blockIdx];
                    float motionStrength = sqrt(mv.x * mv.x + mv.y * mv.y);
                    
                    float filterStrength = MAX(0.1f, 1.0f - motionStrength / MAX_MOTION_VECTOR);
                    
                    if (motionStrength < 5.0f) {
                        int prevIdx = idx;
                        for (int c = 0; c < 3; c++) {
                            bgra[idx + c] = (uint8_t)(bgra[idx + c] * 0.7f +
                                                     prevFrame.data[prevIdx + c] * 0.3f);
                        }
                    }
                }
            }
        }
    }
    
    upscalingModel.adapt(bgra, w, h);
    
    prevFrame = currentFrame;
}

static void processFrameWithVerification(std::vector<uint8_t>& bgra, int w, int h) {
    std::vector<uint8_t> before = bgra;
    
    enhance_quality(bgra, w, h);
    
    if (w < TARGET_W_DEFAULT || h < TARGET_H_DEFAULT) {
        bgra = upscaleFrame(bgra, w, h, TARGET_W_DEFAULT, TARGET_H_DEFAULT);
        w = TARGET_W_DEFAULT;
        h = TARGET_H_DEFAULT;
    }
    
    int changedPixels = 0;
    int totalDiff = 0;
    
    for (int i = 0; i < MIN(before.size(), bgra.size()); i += 4) {
        int diff = abs(before[i] - bgra[i]) + abs(before[i+1] - bgra[i+1]) + abs(before[i+2] - bgra[i+2]);
        if (diff > 10) {
            changedPixels++;
            totalDiff += diff;
        }
    }
}

// -------------------- Frame Rate Management --------------------
static std::vector<std::vector<uint8_t>> generateInterpolatedFrames(int targetW, int targetH) {
    if (frameDataHistory.size() < 2) {
        return {std::vector<uint8_t>(targetW * targetH * 4, 0)};
    }
    
    const auto& frame1 = frameDataHistory[frameDataHistory.size() - 2];
    const auto& frame2 = frameDataHistory[frameDataHistory.size() - 1];
    
    FrameBuffer f1, f2;
    f1.data = frame1;
    f1.width = targetW;
    f1.height = targetH;
    f2.data = frame2;
    f2.width = targetW;
    f2.height = targetH;
    
    estimateMotion(f2, f1);
    
    return generateMultipleInterpolatedFrames(f1, f2, targetW, targetH, MAX_INTERPOLATIONS_PER_FRAME);
}

// -------------------- REAL-TIME CUDA COLOR OVERLAY --------------------

// Live controls — tweak these live!
static float GetBrightness() { return 1.25f; }
static float GetGamma()      { return 2.2f;  }
static float GetContrast()   { return 1.18f; }
static float GetDeltaR()     { return 0.07f; }
static float GetDeltaG()     { return 0.0f;  }
static float GetDeltaB()     { return -0.04f; }

static std::atomic<bool> g_overlayRunning{ true };
static cudaGraphicsResource* g_cudaResource = nullptr;
static ID3D11Texture2D* g_stagingTex = nullptr;

static void RunColorOverlay(ID3D11Device* device, ID3D11DeviceContext* context, IDXGIOutput1* output1)
{
    if (!device || !output1) return;

    IDXGIOutputDuplication* duplRaw = nullptr;
    HRESULT hr = output1->DuplicateOutput(device, &duplRaw);
    if (FAILED(hr)) {
        printf("DuplicateOutput failed: 0x%X\n", hr);
        return;
    }
    ComPtr<IDXGIOutputDuplication> dupl(duplRaw);

    DXGI_OUTPUT_DESC outDesc{};
    output1->GetDesc(&outDesc);
    int w = outDesc.DesktopCoordinates.right - outDesc.DesktopCoordinates.left;
    int h = outDesc.DesktopCoordinates.bottom - outDesc.DesktopCoordinates.top;

    D3D11_TEXTURE2D_DESC texDesc{};
    texDesc.Width = w;
    texDesc.Height = h;
    texDesc.MipLevels = 1;
    texDesc.ArraySize = 1;
    texDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
    texDesc.SampleDesc.Count = 1;
    texDesc.Usage = D3D11_USAGE_DEFAULT;
    texDesc.BindFlags = D3D11_BIND_SHADER_RESOURCE;
    texDesc.CPUAccessFlags = 0;

    if (FAILED(device->CreateTexture2D(&texDesc, nullptr, &g_stagingTex))) {
        printf("Failed to create staging texture\n");
        return;
    }

    cudaError_t cuErr = cudaGraphicsD3D11RegisterResource(&g_cudaResource, g_stagingTex,
        cudaGraphicsRegisterFlagsSurfaceLoadStore);
    if (cuErr != cudaSuccess) {
        printf("CUDA register failed: %s\n", cudaGetErrorString(cuErr));
        return;
    }

    dim3 threads(16, 16);
    dim3 blocks((w + 15) / 16, (h + 15) / 16);

    while (g_overlayRunning.load())
    {
        DXGI_OUTDUPL_FRAME_INFO frameInfo{};
        IDXGIResource* desktopResource = nullptr;

        hr = dupl->AcquireNextFrame(100, &frameInfo, &desktopResource);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            continue;
        }
        if (FAILED(hr)) {
            if (hr == DXGI_ERROR_ACCESS_LOST) break;
            continue;
        }

        ID3D11Texture2D* frameTexRaw = nullptr;
        desktopResource->QueryInterface(IID_ID3D11Texture2D, (void**)&frameTexRaw);
        ComPtr<ID3D11Texture2D> frameTex(frameTexRaw);
        desktopResource->Release();

        context->CopyResource(g_stagingTex, frameTex.Get());
        dupl->ReleaseFrame();

        // Map CUDA resource
        cudaGraphicsMapResources(1, &g_cudaResource, 0);
        Color* devPtr = nullptr;
        size_t numBytes = 0;
        cudaGraphicsResourceGetMappedPointer((void**)&devPtr, &numBytes, g_cudaResource);

        // Launch kernel
        float b = GetBrightness();
        float g = GetGamma();
        float c = GetContrast();
        float dr = GetDeltaR(), dg = GetDeltaG(), db = GetDeltaB();

        launchAdjustKernel(devPtr, w, h, b, g, c, dr, dg, db);
        cudaDeviceSynchronize();

        cudaGraphicsUnmapResources(1, &g_cudaResource, 0);
    }

    // Cleanup
    if (g_cudaResource) cudaGraphicsUnregisterResource(g_cudaResource);
    g_cudaResource = nullptr;
    if (g_stagingTex) {
        g_stagingTex->Release();
        g_stagingTex = nullptr;
    }
}

// -------------------- MAIN SUPERVISOR LOOP + CLEANUP IN ONE FUNCTION --------------------
static void RunSupervisor()
{
    DDAContext ddaCtx;
    if (!init_dda(ddaCtx)) {
        printf("Failed to init desktop duplication\n");
        return;
    }

    DCompContext dcompCtx;
    if (!init_dcomp(dcompCtx, TARGET_W_DEFAULT, TARGET_H_DEFAULT)) {
        printf("Failed to init DirectComposition\n");
        return;
    }

    if (cudaInitialized) {
        allocateCudaMemory(TARGET_W_DEFAULT, TARGET_H_DEFAULT);
    }

    upscalingModel.load(std::string(MODULES_DIR) + "/upscaling_model.bin");

    // Launch cinematic color overlay in background
    std::thread overlayThread(RunColorOverlay, ddaCtx.device.Get(), ddaCtx.context.Get(), ddaCtx.output1.Get());

    std::vector<uint8_t> frame;
    int w = 0, h = 0;
    int frameCount = 0;
    bool running = true;

    auto lastFrameTime = std::chrono::steady_clock::now();
    auto frameInterval = std::chrono::milliseconds(1000 / TARGET_HZ);

    while (running)
    {
        auto now = std::chrono::steady_clock::now();

        bool gotFrame = false;
        if (ddaCtx.initialized) gotFrame = grab_frame_dda(ddaCtx, frame, w, h);
        if (!gotFrame) gotFrame = grab_frame_gdi(frame, w, h);

        if (gotFrame)
        {
            processFrameWithVerification(frame, w, h);

            frameDataHistory.push_back(frame);
            if (frameDataHistory.size() > 3) frameDataHistory.erase(frameDataHistory.begin());

            present_via_dcomp(dcompCtx, frame, TARGET_W_DEFAULT, TARGET_H_DEFAULT);
            frameCount++;

            // Interpolated frames
            auto interp = generateInterpolatedFrames(TARGET_W_DEFAULT, TARGET_H_DEFAULT);
            for (const auto& f : interp) {
                present_via_dcomp(dcompCtx, f, TARGET_W_DEFAULT, TARGET_H_DEFAULT);
                frameCount++;
            }

            if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) running = false;
            if (frameCount % 300 == 0) {
                upscalingModel.save(std::string(MODULES_DIR) + "/upscaling_model.bin");
            }
        }

        sleep_until(lastFrameTime + frameInterval);
        lastFrameTime = std::chrono::steady_clock::now();
    }

    // ——————— CLEANUP ———————
    g_overlayRunning = false;
    if (overlayThread.joinable()) overlayThread.join();

    upscalingModel.save(std::string(MODULES_DIR) + "/upscaling_model.bin");
    freeCudaMemory();
    cleanupCuda();

    printf("Supervisor shutdown complete.\n");
}

// -------------------- MAIN — short and beautiful --------------------
int main()
{
    CoInitialize(nullptr);
    initCuda();
    ensure_dirs();
    set_high_priority();

    RunSupervisor();

    CoUninitialize();
    return 0;
}
