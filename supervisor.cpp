// supervisor.cpp
// Enhanced SuperRes Supervisor with CUDA-accelerated Motion Estimation & AI Upscaling
// Added HDR10+ metadata, real-time color editing, DirectComposition rendering, memory safety, and multiple interpolations

// Removed macro redefinitions since they're already defined on command line
// #define WIN32_LEAN_AND_MEAN
// #define NOMINMAX

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

// Use filesystem alias to avoid namespace issues


// SINGLE MotionVector definition
struct MotionVector {
    int x, y;
    float confidence;
};

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

// -------------------- CUDA Kernels --------------------
// Motion estimation kernel
__global__ void motionEstimationKernel(
    const unsigned char* currentFrame,
    const unsigned char* previousFrame,
    int width, int height,
    MotionVector* motionField,
    int blockSize, int maxMotionVector)
{
    int blockIdxX = blockIdx.x;
    int blockIdxY = blockIdx.y;
    int blocksX = (width + blockSize - 1) / blockSize;
    
    int bestX = 0, bestY = 0;
    float bestError = FLT_MAX;
    
    for (int dy = -maxMotionVector; dy <= maxMotionVector; dy += 2) {
        for (int dx = -maxMotionVector; dx <= maxMotionVector; dx += 2) {
            float error = 0.0f;
            int samples = 0;
            
            for (int py = 0; py < blockSize; py += 2) {
                for (int px = 0; px < blockSize; px += 2) {
                    int currX = blockIdxX * blockSize + px;
                    int currY = blockIdxY * blockSize + py;
                    int prevX = currX + dx;
                    int prevY = currY + dy;
                    
                    if (prevX >= 0 && prevX < width &&
                        prevY >= 0 && prevY < height &&
                        currX >= 0 && currX < width &&
                        currY >= 0 && currY < height) {
                        
                        int currIdx = (currY * width + currX) * 4;
                        int prevIdx = (prevY * width + prevX) * 4;
                        
                        float currLum = 0.299f * currentFrame[currIdx+2] +
                                      0.587f * currentFrame[currIdx+1] +
                                      0.114f * currentFrame[currIdx+0];
                        float prevLum = 0.299f * previousFrame[prevIdx+2] +
                                      0.587f * previousFrame[prevIdx+1] +
                                      0.114f * previousFrame[prevIdx+0];
                        
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
    
    int blockIdx = blockIdxY * blocksX + blockIdxX;
    MotionVector mv = {bestX, bestY, 1.0f - (bestError / 255.0f)};
    motionField[blockIdx] = mv;
}

// Upscaling kernel with color transformation
__global__ void upscalingKernel(
    const unsigned char* input,
    unsigned char* output,
    int inW, int inH, int outW, int outH,
    float scaleX, float scaleY,
    float sharpnessFactor,
    float brightness, float contrast, float saturation, float hue, float gamma,
    float w, float x, float y, float z, float eta,
    float chromaStretch, float hueWarp, float lightnessFlow)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= outW || y >= outH) return;
    
    float srcX = x * scaleX;
    float srcY = y * scaleY;
    
    int x0 = (int)srcX;
    int y0 = (int)srcY;
    int x1 = fminf(x0 + 1, inW - 1);
    int y1 = fminf(y0 + 1, inH - 1);
    
    float dx = srcX - x0;
    float dy = srcY - y0;
    
    int outIdx = (y * outW + x) * 4;
    
    // Bilinear interpolation
    float r = 0.0f, g = 0.0f, b = 0.0f;
    for (int c = 0; c < 3; c++) {
        float p00 = input[(y0 * inW + x0) * 4 + c];
        float p01 = input[(y0 * inW + x1) * 4 + c];
        float p10 = input[(y1 * inW + x0) * 4 + c];
        float p11 = input[(y1 * inW + x1) * 4 + c];
        
        float interpolated = p00 * (1-dx) * (1-dy) +
                            p01 * dx * (1-dy) +
                            p10 * (1-dx) * dy +
                            p11 * dx * dy;
        
        // Apply sharpening
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
                    int nx1 = fminf(nx0 + 1, inW - 1);
                    int ny1 = fminf(ny0 + 1, inH - 1);
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
            interpolated = center + detail * sharpnessFactor * 0.1f;
        }
        
        if (c == 0) r = interpolated;
        else if (c == 1) g = interpolated;
        else b = interpolated;
    }
    
    // Convert to normalized values
    r /= 255.0f;
    g /= 255.0f;
    b /= 255.0f;
    
    // Apply color space transformations (simplified)
    float combined = w * r + x * g + y * b + z * (r + g + b) / 3.0f + eta;
    r = r * (1.0f - chromaStretch) + combined * chromaStretch;
    g = g * (1.0f - chromaStretch) + combined * chromaStretch;
    b = b * (1.0f - chromaStretch) + combined * chromaStretch;
    
    // Apply hue rotation
    float h = atan2f(sqrtf(3.0f) * (g - b), 2.0f * r - g - b) + hue;
    float s = sqrtf((r - g) * (r - g) + (g - b) * (g - b) + (b - r) * (b - r)) / sqrtf(2.0f);
    float l = (r + g + b) / 3.0f;
    
    r = l + s * cosf(h);
    g = l + s * cosf(h - 2.0f * 3.14159f / 3.0f);
    b = l + s * cosf(h + 2.0f * 3.14159f / 3.0f);
    
    // Apply brightness, contrast, saturation
    r = (r - 0.5f) * contrast + 0.5f + brightness;
    g = (g - 0.5f) * contrast + 0.5f + brightness;
    b = (b - 0.5f) * contrast + 0.5f + brightness;
    
    float gray = 0.299f * r + 0.587f * g + 0.114f * b;
    r = gray + saturation * (r - gray);
    g = gray + saturation * (g - gray);
    b = gray + saturation * (b - gray);
    
    // Apply gamma correction
    r = powf(r, 1.0f / gamma);
    g = powf(g, 1.0f / gamma);
    b = powf(b, 1.0f / gamma);
    
    // Convert back to 8-bit values
    output[outIdx + 0] = (unsigned char)fminf(fmaxf(b * 255.0f, 0.0f), 255.0f);
    output[outIdx + 1] = (unsigned char)fminf(fmaxf(g * 255.0f, 0.0f), 255.0f);
    output[outIdx + 2] = (unsigned char)fminf(fmaxf(r * 255.0f, 0.0f), 255.0f);
    output[outIdx + 3] = 255;
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
        
        // Kernel launch
        dim3 gridDim(blocksX, blocksY);
        dim3 blockDim(1);
        
        void* kernelArgs[] = {
            (void*)&d_currentFrame,
            (void*)&d_previousFrame,
            (void*)&current.width,
            (void*)&current.height,
            (void*)&d_motionField,
            (void*)&MOTION_BLOCK_SIZE,
            (void*)&MAX_MOTION_VECTOR
        };
        
        cudaLaunchKernel((void*)motionEstimationKernel, gridDim, blockDim,
                         kernelArgs, 0, cudaStream);
        
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
                                    
                                    error += fabs(currLum - prevLum);
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
                sharpnessFactor = std::min(sharpnessFactor * 1.01f, 3.0f);
            } else {
                sharpnessFactor = std::max(sharpnessFactor * 0.99f, 1.2f);
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
        dim3 gridDim((outW + 15) / 16, (outH + 15) / 16);
        dim3 blockDim(16, 16);
        
        void* kernelArgs[] = {
            (void*)&d_currentFrame,
            (void*)&d_upscaledFrame,
            (void*)&inW,
            (void*)&inH,
            (void*)&outW,
            (void*)&outH,
            (void*)&scaleX,
            (void*)&scaleY,
            (void*)&upscalingModel.sharpnessFactor,
            (void*)&colorEditingParams.brightness,
            (void*)&colorEditingParams.contrast,
            (void*)&colorEditingParams.saturation,
            (void*)&colorEditingParams.hue,
            (void*)&colorEditingParams.gamma,
            (void*)&colorTransformParams.w,
            (void*)&colorTransformParams.x,
            (void*)&colorTransformParams.y,
            (void*)&colorTransformParams.z,
            (void*)&colorTransformParams.eta,
            (void*)&colorTransformParams.chromaStretch,
            (void*)&colorTransformParams.hueWarp,
            (void*)&colorTransformParams.lightnessFlow
        };
        
        cudaLaunchKernel((void*)upscalingKernel, gridDim, blockDim,
                         kernelArgs, 0, cudaStream);
        
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
                int x1 = (int)fminf(x0 + 1, inW - 1);
                int y1 = (int)fminf(y0 + 1, inH - 1);
                
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
                                int nx1 = std::min(nx0 + 1, inW - 1);
                                int ny1 = std::min(ny0 + 1, inH - 1);
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
                    output[outIdx + c] = (uint8_t)(interpolated < 0.0f ? 0.0f : (interpolated > 255.0f ? 255.0f : interpolated));
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
    int adjustedInterpolations = std::min(numInterpolations, 
                                        std::max(1, (int)(avgMotion / 5.0f)));
    
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

HRESULT BeginDraw(
  [in, optional] const RECT *updateRect,
  [in] REFIID iid,
  [out] void **interface
);

if (FAILED(hr) || !rawSurface) return false;

// Use ComPtr for resource safety
ComPtr<IDXGISurface> surface(rawSurface);

// ... use surface as needed below ...
    
ComPtr<ID3D11Texture2D> texture;
hr = surface->QueryInterface(__uuidof(ID3D11Texture2D), 
                            reinterpret_cast<void**>(texture.GetAddressOf()));
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
            frame[idx + 0] = (uint8_t)(b * 255.0f < 0.0f ? 0.0f : (b * 255.0f > 255.0f ? 255.0f : b * 255.0f));
            frame[idx + 1] = (uint8_t)(g * 255.0f < 0.0f ? 0.0f : (g * 255.0f > 255.0f ? 255.0f : g * 255.0f));
            frame[idx + 2] = (uint8_t)(r * 255.0f < 0.0f ? 0.0f : (r * 255.0f > 255.0f ? 255.0f : r * 255.0f));
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
                    
                    float filterStrength = std::max(0.1f, 1.0f - motionStrength / MAX_MOTION_VECTOR);
                    
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
    
    for (int i = 0; i < std::min(before.size(), bgra.size()); i += 4) {
        int diff = abs(before[i] - bgra[i]) + abs(before[i+1] - bgra[i+1]) + abs(before[i+2] - bgra[i+2]);
        if (diff > 10) {
            changedPixels++;
            totalDiff += diff;
        }
    }
}

// -------------------- Frame Rate Management --------------------
static std::vector<std::vector<uint8_t>> frameDataHistory;

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

// -------------------- Main with Enhanced Processing --------------------
int main() {
    initCuda();
    
    ensure_dirs();
    set_high_priority();
    
    DDAContext ddaCtx;
    init_dda(ddaCtx);
    DCompContext dcompCtx;
    init_dcomp(dcompCtx, TARGET_W_DEFAULT, TARGET_H_DEFAULT);
    
    if (cudaInitialized) {
        allocateCudaMemory(TARGET_W_DEFAULT, TARGET_H_DEFAULT);
    }
    
    // Load color transformation parameters
    upscalingModel.load(std::string(MODULES_DIR) + "/upscaling_model.bin");
    
    std::vector<uint8_t> frame;
    int w, h;
    int frameCount = 0;
    bool running = true;
    
    auto lastFrameTime = std::chrono::steady_clock::now();
    auto frameInterval = std::chrono::milliseconds(1000 / TARGET_HZ);
    
    // Color editing controls (simplified for demonstration)
    bool showColorControls = false;
    
    while(running) {
        auto currentTime = std::chrono::steady_clock::now();
        
        bool got = false;
        if(ddaCtx.initialized) got = grab_frame_dda(ddaCtx, frame, w, h);
        if(!got) got = grab_frame_gdi(frame, w, h);
        
        if(got) {
            processFrameWithVerification(frame, w, h);
            
            frameDataHistory.push_back(frame);
            if (frameDataHistory.size() > 3) {
                frameDataHistory.erase(frameDataHistory.begin());
            }
            
            present_via_dcomp(dcompCtx, frame, TARGET_W_DEFAULT, TARGET_H_DEFAULT);
            
            frameCount++;
            
            // Generate and display interpolated frames
            auto interpolatedFrames = generateInterpolatedFrames(TARGET_W_DEFAULT, TARGET_H_DEFAULT);
            for (const auto& interpolatedFrame : interpolatedFrames) {
                present_via_dcomp(dcompCtx, interpolatedFrame, TARGET_W_DEFAULT, TARGET_H_DEFAULT);
                frameCount++;
            }
            
            // Adjust color parameters based on user input (simplified)
            if (GetAsyncKeyState(VK_F1) & 0x8000) {
                colorEditingParams.brightness += 0.05f;
                colorEditingParams.brightness = std::min(colorEditingParams.brightness, 1.0f);
            }
            if (GetAsyncKeyState(VK_F2) & 0x8000) {
                colorEditingParams.brightness -= 0.05f;
                colorEditingParams.brightness = std::max(colorEditingParams.brightness, -1.0f);
            }
            if (GetAsyncKeyState(VK_F3) & 0x8000) {
                colorEditingParams.contrast += 0.1f;
                colorEditingParams.contrast = std::min(colorEditingParams.contrast, 3.0f);
            }
            if (GetAsyncKeyState(VK_F4) & 0x8000) {
                colorEditingParams.contrast -= 0.1f;
                colorEditingParams.contrast = std::max(colorEditingParams.contrast, 0.1f);
            }
            if (GetAsyncKeyState(VK_F5) & 0x8000) {
                colorEditingParams.saturation += 0.1f;
                colorEditingParams.saturation = std::min(colorEditingParams.saturation, 3.0f);
            }
            if (GetAsyncKeyState(VK_F6) & 0x8000) {
                colorEditingParams.saturation -= 0.1f;
                colorEditingParams.saturation = std::max(colorEditingParams.saturation, 0.0f);
            }
            if (GetAsyncKeyState(VK_F7) & 0x8000) {
                colorEditingParams.hue += 0.1f;
            }
            if (GetAsyncKeyState(VK_F8) & 0x8000) {
                colorEditingParams.hue -= 0.1f;
            }
            if (GetAsyncKeyState(VK_F9) & 0x8000) {
                colorEditingParams.gamma += 0.1f;
                colorEditingParams.gamma = std::min(colorEditingParams.gamma, 3.0f);
            }
            if (GetAsyncKeyState(VK_F10) & 0x8000) {
                colorEditingParams.gamma -= 0.1f;
                colorEditingParams.gamma = std::max(colorEditingParams.gamma, 0.1f);
            }
            
            if(frameCount % 60 == 0) {
                // Log frame processing stats
            }
        }
        
        if (GetAsyncKeyState(VK_ESCAPE) & 0x8000) {
            running = false;
        }
        
        if (frameCount % 300 == 0) {
            upscalingModel.save(std::string(MODULES_DIR) + "/upscaling_model.bin");
        }
        
        std::this_thread::sleep_until(currentTime + frameInterval);
    }
    
    freeCudaMemory();
    cleanupCuda();
    
    upscalingModel.save(std::string(MODULES_DIR) + "/upscaling_model.bin");
    return 0;
}
