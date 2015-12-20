#pragma once
#ifndef FASTNOIS3D_H
#define FASTNOISE3D_H
#include "FastNoise.h"


extern "C" {
	FAST_NOISE_DLL_API inline extern SIMD simplexSIMD3d(SIMD* x, SIMD* y, SIMD* z);		
	FAST_NOISE_DLL_API inline extern float simplex3d(float x, float y, float z);
	FAST_NOISE_DLL_API inline extern float perlin3d(float x, float y, float z);
	FAST_NOISE_DLL_API inline extern SIMD perlinSIMD3d(SIMD* __restrict x, SIMD* __restrict y, SIMD* __restrict z);
}

#endif
