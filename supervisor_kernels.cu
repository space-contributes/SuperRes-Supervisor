// supervisor_kernels.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>
#include <cfloat>  // FLT_MAX
#include "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.0/include/vector_types.h"

// Custom min/max functions to avoid conflicts with CUDA
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define CLAMP(val, min_val, max_val) ((val) < (min_val) ? (min_val) : ((val) > (max_val) ? (max_val) : (val)))

struct MotionVector {
    int x, y;
    float confidence;
};

struct Color { 
    float r, g, b, a; 
};

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
    // Fixed variable names to avoid conflicts with function parameters
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (px >= outW || py >= outH) return;
    
    float srcX = px * scaleX;
    float srcY = py * scaleY;
    
    int x0 = (int)srcX;
    int y0 = (int)srcY;
    int x1 = MIN(x0 + 1, inW - 1);
    int y1 = MIN(y0 + 1, inH - 1);
    
    float dx = srcX - x0;
    float dy = srcY - y0;
    
    int outIdx = (py * outW + px) * 4;
    
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
        if (px > 0 && px < outW-1 && py > 0 && py < outH-1) {
            float center = interpolated;
            float sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    if (kx == 0 && ky == 0) continue;
                    int nx = px + kx;
                    int ny = py + ky;
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
    output[outIdx + 0] = (unsigned char)CLAMP(b * 255.0f, 0.0f, 255.0f);
    output[outIdx + 1] = (unsigned char)CLAMP(g * 255.0f, 0.0f, 255.0f);
    output[outIdx + 2] = (unsigned char)CLAMP(r * 255.0f, 0.0f, 255.0f);
    output[outIdx + 3] = 255;
}

// Color adjustment kernel
extern "C" __global__ void AdjustKernel(Color* pixels, int width, int height,
                            float brightness, float gamma, float contrast,
                            float deltaR, float deltaG, float deltaB)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    int idx = y * width + x;
    Color c = pixels[idx];

    float invGamma = 1.0f / gamma;
    c.r = fminf(fmaxf(powf(c.r * brightness, invGamma) * contrast + deltaR, 0.0f), 1.0f);
    c.g = fminf(fmaxf(powf(c.g * brightness, invGamma) * contrast + deltaG, 0.0f), 1.0f);
    c.b = fminf(fmaxf(powf(c.b * brightness, invGamma) * contrast + deltaB, 0.0f), 1.0f);

    pixels[idx] = c;
}
