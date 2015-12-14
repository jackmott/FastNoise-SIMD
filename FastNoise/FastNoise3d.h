#pragma once
#ifndef FASTNOISEEXTERNS_H
#define FASTNOISEEXTERNS_H
#include "FastNoise.h"


enum NoiseType { FBMSIMD, TURBULENCESIMD, RIDGESIMD, PLAINSIMD, FBM, TURBULENCE, RIDGE, PLAIN };

extern "C" {

		
	FAST_NOISE_DLL_API inline extern float noise3d(float x, float y, float z);
	FAST_NOISE_DLL_API inline extern SIMD noiseSIMD3d(SIMD* x, SIMD* y, SIMD* z);

	FAST_NOISE_DLL_API inline extern void fbmSIMD3d(SIMD* out, Settings* S);
	FAST_NOISE_DLL_API inline extern void plainSIMD3d(SIMD* out, Settings* S);
	FAST_NOISE_DLL_API inline extern void turbulenceSIMD3d(SIMD* out, Settings* S);
	FAST_NOISE_DLL_API inline extern void ridgeSIMD3d(SIMD* out, Settings* S);

	FAST_NOISE_DLL_API inline extern float fbm3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset);
	FAST_NOISE_DLL_API inline extern float plain3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset);
	FAST_NOISE_DLL_API inline extern float turbulence3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset);
	FAST_NOISE_DLL_API inline extern float ridge3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset);


	FAST_NOISE_DLL_API extern float* GetSphereSurfaceNoiseSIMD(int width, int height, int octaves, float lacunarity, float frequency, float gain, float offset, int noiseType);
	FAST_NOISE_DLL_API extern float* GetSphereSurfaceNoise(int width, int height, int octaves, float lacunarity, float frequency, float gain, float offset, int noiseType);
	FAST_NOISE_DLL_API extern void CleanUpNoiseSIMD(float * resultArray);
	FAST_NOISE_DLL_API extern void CleanUpNoise(float * resultArray);
	
}

#endif
