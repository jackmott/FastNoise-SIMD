#include "headers\FractalNoise3d.h"
#include <stdio.h>
//If you ever call something with 1 octave, call this instead
 void plainSIMD3d(SIMD* out, Settings*  S,ISIMDNoise3d noise )
{

	SIMD vfx = Mul(S->x.m, S->frequency);
	SIMD vfy = Mul(S->y.m, S->frequency);
	SIMD vfz = Mul(S->z.m, S->frequency);

	*out = noise(&vfx, &vfy, &vfz);
}

 float plain3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset,INoise3d noise)
{
	return noise(x*frequency, y*frequency, z*frequency);
}



 void ridgePlainSIMD3d(SIMD* __restrict out, Settings* __restrict S, ISIMDNoise3d noise)
{
	SIMD vfx = Mul(S->x.m, S->frequency);
	SIMD vfy = Mul(S->y.m, S->frequency);
	SIMD vfz = Mul(S->z.m, S->frequency);
	SIMD r = noise(&vfx, &vfy, &vfz);
	//abs of r
	*out = Max(Sub(zero, r), r);

}

 float ridgePlain3d(float x, float y, float z, float lacunarity, float gain, float frequency, int octaves, float offset, INoise3d noise)
{
	return (float)fabs(noise(x*frequency, y*frequency, z*frequency));
}



//Fractal brownian motions using SIMD
 void fbmSIMD3d(SIMD* __restrict out, Settings* __restrict S, ISIMDNoise3d noise)
{
	SIMD amplitude, localFrequency;
	*out = SetZero();
	amplitude = SetOne(1);
	localFrequency = S->frequency;
	for (int i = S->octaves; i != 0; i--)
	{
		SIMD vfx = Mul(S->x.m, localFrequency);
		SIMD vfy = Mul(S->y.m, localFrequency);
		SIMD vfz = Mul(S->z.m, localFrequency);
		*out = Add(*out, Mul(amplitude, noise(&vfx, &vfy, &vfz)));
		localFrequency = Mul(localFrequency, S->lacunarity);
		amplitude = Mul(amplitude, S->gain);
	}	

}

//fractal brownian motion without SIMD
 float fbm3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset, INoise3d noise)
{
	float sum = 0;
	float amplitude = 1.0f;
	for (int i = octaves; i != 0; i--)
	{
		sum += noise(x*frequency, y*frequency, z*frequency)*amplitude;
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}


//turbulence  using SIMD
 void turbulenceSIMD3d(SIMD *out,Settings* S, ISIMDNoise3d noise)
{
	SIMD amplitude, localFrequency;
	*out = SetZero();
	amplitude = SetOne(1.0f);
	localFrequency = S->frequency;
	for (int i = S->octaves; i != 0; i--)
	{
		SIMD vfx = Mul(S->x.m, localFrequency);
		SIMD vfy = Mul(S->y.m, localFrequency);
		SIMD vfz = Mul(S->z.m, localFrequency);
		SIMD r = Mul(amplitude, noise(&vfx, &vfy, &vfz));
		//get abs of r by trickery
		r = Max(Sub(zero, r), r);
		*out = Add(*out, r);
		localFrequency = Mul(localFrequency, S->lacunarity);
		amplitude = Mul(amplitude, S->gain);
	}	
}

 float turbulence3d(float x, float y, float z, float lacunarity, float gain, float frequency, int octaves, float offset, INoise3d noise)
{
	float sum = 0;
	float amplitude = 1;
	for (int i = octaves; i != 0; i--)
	{
		sum += (float)fabs(noise(x*frequency, y*frequency, z*frequency)*amplitude);
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}


 void ridgeSIMD3d(SIMD* out, Settings* S, ISIMDNoise3d noise)
{
	SIMD amplitude, prev, localFrequency;
	*out = SetZero();
	amplitude = SetOne(1.0f);
	prev = SetOne(1.0f);
	localFrequency = S->frequency;
	for (int i = S->octaves; i != 0; i--)
	{
		SIMD vfx = Mul(S->x.m, localFrequency);
		SIMD vfy = Mul(S->y.m, localFrequency);
		SIMD vfz = Mul(S->z.m, localFrequency);
		SIMD r = noise(&vfx, &vfy, &vfz);
		//get abs of r by trickery
		r = Max(Sub(zero, r), r);
		r = Sub(S->offset, r);
		r = Mul(r, r);
		r = Mul(r, amplitude);
		r = Mul(r, prev);
		*out = Add(*out, r);
		prev = Load((const float*)&r);
		localFrequency = Mul(localFrequency, S->lacunarity);
		amplitude = Mul(amplitude, S->gain);
	}
	
}

 float ridge3d(float x, float y, float z, float lacunarity, float gain, float frequency, int octaves, float offset, INoise3d noise)
{
	float sum = 0;
	float amplitude = 0.5f;
	float prev = 1.0f;
	for (int i = octaves; i != 0; i--)
	{
		float r = (float)fabs(noise(x*frequency, y*frequency, z*frequency));
		r = offset - r;
		r = r*r;
		sum += r*amplitude*prev;
		prev = r;
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}

