#include "FastNoiseExterns.h"
#include <math.h>
#include <time.h>
#include <stdio.h>
//Must be called by the caller of noise producing functions
void CleanUpSIMDNoise(float * resultArray)
{
	_aligned_free(resultArray);
}

//Must be called by the caller of noise producing functions
void CleanUpNoise(float * resultArray)
{
	free(resultArray);
}

float* GetSpherSurfaceNoiseSIMD(int width, int height, int octaves, float lacunarity, float frequency, float gain, float offset, ISIMDNoise noise)
{
	Settings S;
	initSIMD(&S, frequency, lacunarity, offset, gain, octaves);

	SIMD* result = (SIMD*)_aligned_malloc(width*height*  sizeof(float), MEMORY_ALIGNMENT);


	//set up spherical stuff
	int count = 0;
	const float piOverHeight = pi / height;
	const float twoPiOverWidth = twopi / width;
	float phi = 0;
	float x3d, y3d, z3d;
	float sinPhi, theta;


	for (int y = 0; y < height; y = y + 1)
	{
		phi = phi + piOverHeight;
		z3d = cosf(phi);
		sinPhi = sinf(phi);
		theta = 0;

		S.z.m = SetOne(z3d);

		for (int x = 0; x < width - (VECTOR_SIZE - 1); x = x + VECTOR_SIZE)
		{

			for (int j = 0; j < VECTOR_SIZE; j++)
			{
				theta = theta + twoPiOverWidth;
				x3d = cosf(theta) * sinPhi;
				y3d = sinf(theta) * sinPhi;

				S.x.a[j] = x3d;
				S.y.a[j] = y3d;

			}

			noise(&result[count], &S);
			count = count + 1;

		}
	}

	return (float*)result;

}

float* GetSpherSurfaceNoise(int width, int height, int octaves, float lacunarity, float frequency, float gain, float offset, INoise noise)
{

	float* result = (float*)malloc(width*height*  sizeof(float));


	//set up spherical stuff
	int count = 0;
	const float piOverHeight = pi / height;
	const float twoPiOverWidth = twopi / width;
	float phi = 0;
	float x3d, y3d, z3d;
	float sinPhi, theta;


	for (int y = 0; y < height; y = y + 1)
	{
		phi = phi + piOverHeight;
		z3d = cosf(phi);
		sinPhi = sinf(phi);
		theta = 0;
		for (int x = 0; x < width; x = x + 1)
		{


			theta = theta + twoPiOverWidth;
			x3d = cosf(theta) * sinPhi;
			y3d = sinf(theta) * sinPhi;

			result[count] = noise(x3d, y3d, z3d, frequency, lacunarity, gain, octaves, offset);
			count = count + 1;

		}
	}

	return result;

}

int main()
{

	clock_t t1, t2;
	t1 = clock();

	float * r = GetSpherSurfaceNoiseSIMD(4096, 2048, 3, 2, 1, .5, 0, fbmSIMD);

	t2 = clock();

	float diff = (float)t2 - (float)t1;
	printf("Time Taken for Sphere SIMD: %f\n", diff);
	printf("random result:%f\n", r[3000]);
	CleanUpSIMDNoise(r);


	t1 = clock();

	r = GetSpherSurfaceNoise(4096, 2048, 3, 2, 1, .5, 0, fbm);

	t2 = clock();

	diff = t2 - t1;
	printf("Time Taken for Sphere NON SIMD: %f\n", diff);
	printf("random result:%f\n", r[3000]);
	CleanUpNoise(r);


	int xy;
	scanf("%d", &xy);

	
}