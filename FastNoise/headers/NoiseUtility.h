#pragma once
#ifndef NOISEUTILITY_H
#define NOISEUTILITY_H
#include "FractalNoise3d.h"

extern "C" {
	FAST_NOISE_DLL_API extern float* GetSphereSurfaceNoiseSIMD(int width, int height, int octaves, float lacunarity, float frequency, float gain, float offset,int fractalType, int noiseType, float* outMin, float * outMax);
	FAST_NOISE_DLL_API extern float* GetSphereSurfaceNoise(int width, int height, int octaves, float lacunarity, float frequency, float gain, float offset,int fractalType, int noiseType, float* outMin, float* outMax);
	FAST_NOISE_DLL_API extern void CleanUpNoiseSIMD(float * resultArray);
	FAST_NOISE_DLL_API extern void CleanUpNoise(float * resultArray);
}

#endif
