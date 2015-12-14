#pragma once
#ifndef FASTNOISEEXTERNS_H
#define FASTNOISEEXTERNS_H
#include "FastNoise.h"
#define FAST_NOISE_DLL_API __declspec(dllexport)

extern "C" {
	FAST_NOISE_DLL_API float* GetSpherSurfaceNoiseSIMD(int width, int height, int octaves, float lacunarity, float frequency, float gain, float offset, ISIMDNoise noise);
	FAST_NOISE_DLL_API void CleanUpSIMDNoise(float * resultArray);
	FAST_NOISE_DLL_API void CleanUpNoise(float * resultArray);


}

#endif
