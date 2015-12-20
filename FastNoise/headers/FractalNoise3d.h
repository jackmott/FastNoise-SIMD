#pragma once
#ifndef FRACTALNOISE3D_H
#define FRACTALNOISE3D_H
#include "FastNoise.h"
#include "FastNoise3d.h"
#include <math.h>

extern "C"
{
	
	FAST_NOISE_DLL_API inline extern void fbmSIMD3d(SIMD* __restrict out, Settings* __restrict S,ISIMDNoise3d noise);
	FAST_NOISE_DLL_API inline extern void plainSIMD3d(SIMD* __restrict out, Settings* __restrict S, ISIMDNoise3d noise);
	FAST_NOISE_DLL_API inline extern void turbulenceSIMD3d(SIMD* out, Settings* S, ISIMDNoise3d noise);
	FAST_NOISE_DLL_API inline extern void ridgeSIMD3d(SIMD* out, Settings* S, ISIMDNoise3d noise);
	FAST_NOISE_DLL_API inline extern void ridgePlainSIMD3d(SIMD* __restrict out, Settings* __restrict S, ISIMDNoise3d noise);


	FAST_NOISE_DLL_API inline extern float fbm3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset,INoise3d noise);
	FAST_NOISE_DLL_API inline extern float plain3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset, INoise3d noise);
	FAST_NOISE_DLL_API inline extern float turbulence3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset, INoise3d noise);
	FAST_NOISE_DLL_API inline extern float ridge3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset, INoise3d noise);
	FAST_NOISE_DLL_API inline extern float ridgePlain3d(float x, float y, float z, float lacunarity, float gain, float frequency, int octaves, float offset, INoise3d noise);
}
#endif
