#define PERLIN_NOISE_DLL_API __declspec(dllexport)

extern "C" {
	PERLIN_NOISE_DLL_API float* GetSphericalPerlinNoise(int width, int height, int octaves, float lacunarity, float gain, float stretch, float offsetx, float offsety);
	PERLIN_NOISE_DLL_API void CleanUpSphericalPerlinNoise(float * resultArray);
}