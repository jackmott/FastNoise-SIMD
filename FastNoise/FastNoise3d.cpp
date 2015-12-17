#include "FastNoise3d.h"
#include <math.h>
#include <stdio.h>
#include <thread>

//---------------------------------------------------------------------

/*
* Helper functions to compute gradients-dot-residualvectors (1D to 4D)
* Note that these generate gradients of more than unit length. To make
* a close match with the value range of classic Perlin noise, the final
* noise values need to be rescaled. To match the RenderMan noise in a
* statistical sense, the approximate scaling values (empirically
* determined from test renderings) are:
* 1D noise needs rescaling with 0.188
* 2D noise needs rescaling with 0.507
* 3D noise needs rescaling with 0.936
* 4D noise needs rescaling with 0.87
* Note that these noise functions are the most practical and useful
* signed version of Perlin noise. To return values according to the
* RenderMan specification from the SL noise() and pnoise() functions,
* the noise values need to be scaled and offset to [0,1], like this:
* float SLnoise = (Noise1234::noise(x,y,z) + 1.0) * 0.5;
*/

inline float grad3d(int hash, float x, float y, float z) {
	int h = hash & 15;     // Convert low 4 bits of hash code into 12 simple
	float u = h < 8 ? x : y; // gradient directions, and compute dot product.
	float v = h < 4 ? y : h == 12 || h == 14 ? x : z; // Fix repeats at h = 12 to 15
	return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}


inline SIMD  gradSIMD3d(SIMDi *hash, SIMD *x, SIMD *y, SIMD *z) {

	SIMDi h = Andi(*hash, fifteeni);
	SIMD h1 = ConvertToFloat(Equali(zeroi, Andi(h, one)));
	SIMD h2 = ConvertToFloat(Equali(zeroi, Andi(h, two)));


	//if h < 8 then x, else y
	SIMD u = CastToFloat(LessThani(h, eight));
	u = Or(And(u, *x), AndNot(u, *y));

	//if h < 4 then y else if h is 12 or 14 then x else z
	SIMD v = CastToFloat(LessThani(h, four));
	SIMD h12o14 = CastToFloat(Equali(zeroi, Ori(Equali(h, twelve), Equali(h, fourteen))));
	h12o14 = Or(AndNot(h12o14, *x), And(h12o14, *z));
	v = Or(And(v, *y), AndNot(v, h12o14));


	//if h1 then -u else u	
	//if h2 then -v else v
	//then add them
	return Add(Or(AndNot(h1, Sub(zero, u)), And(h1, u)), Or(AndNot(h2, Sub(zero, v)), And(h2, v)));
}


inline SIMD noiseSIMD3d(SIMD* x, SIMD* y, SIMD* z)
{
	uSIMDi ix0, iy0, ix1, iy1, iz0, iz1;
	SIMD fx0, fy0, fz0, fx1, fy1, fz1;

	//use built in floor if we have it
#ifdef SSE41
	ix0.m = ConvertToInt(Floor(*x));
	iy0.m = ConvertToInt(Floor(*y));
	iz0.m = ConvertToInt(Floor(*z));
#endif
	//drop out to scalar if we don't
#ifndef SSE41
	uSIMD* ux = x;
	uSIMD* uy = y;
	uSIMD* uz = z;
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		ix0.a[i] = FASTFLOOR((*ux).a[i]);
		iy0.a[i] = FASTFLOOR((*uy).a[i]);
		iz0.a[i] = FASTFLOOR((*uz).a[i]);
	}
#endif

	fx0 = Sub(*x, ConvertToFloat(ix0.m));
	fy0 = Sub(*y, ConvertToFloat(iy0.m));
	fz0 = Sub(*z, ConvertToFloat(iz0.m));

	fx1 = Sub(fx0, onef);
	fy1 = Sub(fy0, onef);
	fz1 = Sub(fz0, onef);

	ix1.m = Andi(Addi(ix0.m, one), ff);
	iy1.m = Andi(Addi(iy0.m, one), ff);
	iz1.m = Andi(Addi(iz0.m, one), ff);

	ix0.m = Andi(ix0.m, ff);
	iy0.m = Andi(iy0.m, ff);
	iz0.m = Andi(iz0.m, ff);


	SIMD
		r = Mul(fz0, six);
	r = Sub(r, fifteen);
	r = Mul(r, fz0);
	r = Add(r, ten);
	r = Mul(r, fz0);
	r = Mul(r, fz0);
	r = Mul(r, fz0);

	SIMD
		t = Mul(fy0, six);
	t = Sub(t, fifteen);
	t = Mul(t, fy0);
	t = Add(t, ten);
	t = Mul(t, fy0);
	t = Mul(t, fy0);
	t = Mul(t, fy0);

	SIMD
		s = Mul(fx0, six);
	s = Sub(s, fifteen);
	s = Mul(s, fx0);
	s = Add(s, ten);
	s = Mul(s, fx0);
	s = Mul(s, fx0);
	s = Mul(s, fx0);


	uSIMDi p[8];
#ifndef USEGATHER

	
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		p[0].a[i] = perm[ix0.a[i] + perm[iy0.a[i] + perm[iz0.a[i]]]];
		p[1].a[i] = perm[ix0.a[i] + perm[iy0.a[i] + perm[iz1.a[i]]]];
		p[2].a[i] = perm[ix0.a[i] + perm[iy1.a[i] + perm[iz0.a[i]]]];
		p[3].a[i] = perm[ix0.a[i] + perm[iy1.a[i] + perm[iz1.a[i]]]];
		p[4].a[i] = perm[ix1.a[i] + perm[iy0.a[i] + perm[iz0.a[i]]]];
		p[5].a[i] = perm[ix1.a[i] + perm[iy0.a[i] + perm[iz1.a[i]]]];
		p[6].a[i] = perm[ix1.a[i] + perm[iy1.a[i] + perm[iz0.a[i]]]];
		p[7].a[i] = perm[ix1.a[i] + perm[iy1.a[i] + perm[iz1.a[i]]]];

	}
#endif // !AVX
#ifdef USEGATHER 
	SIMDi pz0, pz1, pz0y0, pz0y1, pz1y1, pz1y0;

	pz0 = Gather(perm, iz0.m, 4);
	pz1 = Gather(perm, iz1.m, 4);

	pz0y0 = Gather(perm, Addi(iy0.m, pz0), 4);
	pz0y1 = Gather(perm, Addi(iy1.m, pz0), 4);
	pz1y0 = Gather(perm, Addi(iy0.m, pz1), 4);
	pz1y1 = Gather(perm, Addi(iy1.m, pz1), 4);

	p[0].m = Addi(ix0.m, pz0y0);
	p[0].m = Gather(perm, p[0].m, 4);

	p[1].m = Addi(ix0.m, pz1y0);
	p[1].m = Gather(perm, p[1].m, 4);

	p[2].m = Addi(ix0.m, pz0y1);
	p[2].m = Gather(perm, p[2].m, 4);

	p[3].m = Addi(ix0.m, pz1y1);
	p[3].m = Gather(perm, p[3].m, 4);

	p[4].m = Addi(ix1.m, pz0y0);
	p[4].m = Gather(perm, p[4].m, 4);

	p[5].m = Addi(ix1.m, pz1y0);
	p[5].m = Gather(perm, p[5].m, 4);

	p[6].m = Addi(ix1.m, pz0y1);
	p[6].m = Gather(perm, p[6].m, 4);

	p[7].m = Addi(ix1.m, pz1y1);
	p[7].m = Gather(perm, p[7].m, 4);


#endif // AVX


	SIMD nxy0 = gradSIMD3d(&p[0].m, &fx0, &fy0, &fz0);
	SIMD nxy1 = gradSIMD3d(&p[1].m, &fx0, &fy0, &fz1);
	SIMD nx0 = Add(nxy0, Mul(r, Sub(nxy1, nxy0)));

	nxy0 = gradSIMD3d(&p[2].m, &fx0, &fy1, &fz0);
	nxy1 = gradSIMD3d(&p[3].m, &fx0, &fy1, &fz1);
	SIMD nx1 = Add(nxy0, Mul(r, Sub(nxy1, nxy0)));

	SIMD n0 = Add(nx0, Mul(t, Sub(nx1, nx0)));

	nxy0 = gradSIMD3d(&p[4].m, &fx1, &fy0, &fz0);
	nxy1 = gradSIMD3d(&p[5].m, &fx1, &fy0, &fz1);
	nx0 = Add(nxy0, Mul(r, Sub(nxy1, nxy0)));

	nxy0 = gradSIMD3d(&p[6].m, &fx1, &fy1, &fz0);
	nxy1 = gradSIMD3d(&p[7].m, &fx1, &fy1, &fz1);
	nx1 = Add(nxy0, Mul(r, Sub(nxy1, nxy0)));

	SIMD n1 = Add(nx0, Mul(t, Sub(nx1, nx0)));

	return  Mul(Sub(Add(n0, Mul(s, Sub(n1, n0))),poffset),pscale);

}



//---------------------------------------------------------------------
/** 3D float Perlin noise.
*/
inline float noise3d(float x, float y, float z)
{
	int ix0, iy0, ix1, iy1, iz0, iz1;
	float fx0, fy0, fz0, fx1, fy1, fz1;
	float s, t, r;
	float nxy0, nxy1, nx0, nx1, n0, n1;

	ix0 = FASTFLOOR(x); // (int)x; // Integer part of x
	iy0 = FASTFLOOR(y); // Integer part of y
	iz0 = FASTFLOOR(z); // Integer part of z
	fx0 = x - ix0;        // Fractional part of x
	fy0 = y - iy0;        // Fractional part of y
	fz0 = z - iz0;        // Fractional part of z
	fx1 = fx0 - 1.0f;
	fy1 = fy0 - 1.0f;
	fz1 = fz0 - 1.0f;
	ix1 = (ix0 + 1) & 0xff; // Wrap to 0..255
	iy1 = (iy0 + 1) & 0xff;
	iz1 = (iz0 + 1) & 0xff;
	ix0 = ix0 & 0xff;
	iy0 = iy0 & 0xff;
	iz0 = iz0 & 0xff;

	r = FADE(fz0);
	t = FADE(fy0);
	s = FADE(fx0);

	

	nxy0 = grad3d(perm[ix0 + perm[iy0 + perm[iz0]]], fx0, fy0, fz0);
	nxy1 = grad3d(perm[ix0 + perm[iy0 + perm[iz1]]], fx0, fy0, fz1);
	nx0 = LERP(r, nxy0, nxy1);
	

	nxy0 = grad3d(perm[ix0 + perm[iy1 + perm[iz0]]], fx0, fy1, fz0);
	nxy1 = grad3d(perm[ix0 + perm[iy1 + perm[iz1]]], fx0, fy1, fz1);
	nx1 = LERP(r, nxy0, nxy1);

	n0 = LERP(t, nx0, nx1);

	nxy0 = grad3d(perm[ix1 + perm[iy0 + perm[iz0]]], fx1, fy0, fz0);
	nxy1 = grad3d(perm[ix1 + perm[iy0 + perm[iz1]]], fx1, fy0, fz1);
	nx0 = LERP(r, nxy0, nxy1);

	nxy0 = grad3d(perm[ix1 + perm[iy1 + perm[iz0]]], fx1, fy1, fz0);
	nxy1 = grad3d(perm[ix1 + perm[iy1 + perm[iz1]]], fx1, fy1, fz1);
	nx1 = LERP(r, nxy0, nxy1);

	n1 = LERP(t, nx0, nx1);



	return (LERP(s, n0, n1) - OFFSET)*SCALE;
}



//If you ever call something with 1 octave, call this instead
inline void plainSIMD3d(SIMD* out, Settings* S)
{
	SIMD vfx = Mul(S->x.m, S->frequency);
	SIMD vfy = Mul(S->y.m, S->frequency);
	SIMD vfz = Mul(S->z.m, S->frequency);
	*out = noiseSIMD3d(&vfx, &vfy, &vfz);
}

inline float plain3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset)
{
	return noise3d(x*frequency, y*frequency, z*frequency);
}

inline void ridgePlainSIMD3d(SIMD* out, Settings* S)
{
	SIMD vfx = Mul(S->x.m, S->frequency);
	SIMD vfy = Mul(S->y.m, S->frequency);
	SIMD vfz = Mul(S->z.m, S->frequency);
	SIMD r = noiseSIMD3d(&vfx, &vfy, &vfz);
	//abs of r
	*out = Max(Sub(zero, r), r);
	
}

inline float ridgePlain3d(float x, float y, float z, float lacunarity, float gain, float frequency, int octaves, float offset)
{
	return (float)fabs(noise3d(x*frequency, y*frequency, z*frequency));
}




//Fractal brownian motions using SIMD
inline void fbmSIMD3d(SIMD* out, Settings* S)
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
		*out = Add(*out, Mul(amplitude, noiseSIMD3d(&vfx, &vfy, &vfz)));
		localFrequency = Mul(localFrequency, S->lacunarity);
		amplitude = Mul(amplitude, S->gain);
	}


}

//fractal brownian motion without SIMD
inline float fbm3d(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset)
{
	float sum = 0;
	float amplitude = 1.0f;
	for (int i = octaves; i != 0; i--)
	{
		sum += noise3d(x*frequency, y*frequency, z*frequency)*amplitude;
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}


//turbulence  using SIMD
inline void turbulenceSIMD3d(SIMD* out, Settings* S)
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
		SIMD r = Mul(amplitude, noiseSIMD3d(&vfx, &vfy, &vfz));		
		//get abs of r by trickery
		r = Max(Sub(zero, r), r);
		*out = Add(*out, r);
		localFrequency = Mul(localFrequency, S->lacunarity);
		amplitude = Mul(amplitude, S->gain);
	}

}

inline float turbulence3d(float x, float y, float z, float lacunarity, float gain, float frequency, int octaves, float offset)
{
	float sum = 0;
	float amplitude = 1;
	for (int i = octaves; i != 0; i--)
	{
		sum += (float)fabs(noise3d(x*frequency, y*frequency, z*frequency)*amplitude);
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}






inline void ridgeSIMD3d(SIMD* out, Settings* S)
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
		SIMD r = noiseSIMD3d(&vfx, &vfy, &vfz);
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

inline float ridge3d(float x, float y, float z, float lacunarity, float gain, float frequency, int octaves, float offset)
{
	float sum = 0;
	float amplitude = 0.5f;
	float prev = 1.0f;
	for (int i = octaves; i != 0; i--)
	{
		float r = (float)fabs(noise3d(x*frequency, y*frequency, z*frequency));
		r = offset - r;
		r = r*r;
		sum += r*amplitude*prev;
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}




//Must be called by the caller of noise producing functions
void CleanUpNoiseSIMD(float * resultArray)
{
	_aligned_free(resultArray);
}

//Must be called by the caller of noise producing functions
void CleanUpNoise(float * resultArray)
{
	free(resultArray);
}

void SphereSurfaceNoiseSIMDThread(int start, int end,int width, int height, Settings* S, float* xcos, float* ysin, ISIMDNoise noiseFunction, SIMD* result, float *outMin, float *outMax)
{
	const float piOverHeight = pi / (height + 1);	
	float phi = piOverHeight*start;
	float sinPhi;
	
	int count = start*width / VECTOR_SIZE;

	uSIMD min;
	uSIMD max;
	min.m = SetOne(999);
	max.m = SetOne(-999);
	for (int y = start; y < end; y = y + 1)
	{
		phi = phi + piOverHeight;
		S->z.m = SetOne(cosf(phi));
		sinPhi = sinf(phi);


		for (int x = 0; x < width - (VECTOR_SIZE - 1); x = x + VECTOR_SIZE)
		{

			for (int j = 0; j < VECTOR_SIZE; j++)
			{
				S->x.a[j] = xcos[x+j] * sinPhi;
				S->y.a[j] = ysin[x+j] * sinPhi;

			}

			//printf("setting count:%i ...", count);
			noiseFunction(&result[count], S);
		//	printf(" done setting count:%i\n", count);
			min.m = Min(min.m, result[count]);
			max.m = Max(max.m, result[count]);
			count = count + 1;
		}
	}

	*outMin = 999;
	*outMax = -999;
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		*outMin = fminf(*outMin, min.a[i]);
		*outMax = fmaxf(*outMax, max.a[i]);
	}

}

float* GetSphereSurfaceNoiseSIMD(int width, int height, int octaves, float lacunarity, float frequency, float gain, float offset, int noiseType, float* outMin, float *outMax)
{
	
	SIMD* result = (SIMD*)_aligned_malloc(width*height*  sizeof(float), MEMORY_ALIGNMENT);

	ISIMDNoise noiseFunction;

	switch ((NoiseType)noiseType)
	{
	case FBM: noiseFunction = fbmSIMD3d; break;
	case TURBULENCE: noiseFunction = turbulenceSIMD3d; break;
	case RIDGE: noiseFunction = ridgeSIMD3d; break;
	case PLAIN: noiseFunction = plainSIMD3d; break;
	default:return 0;
	}

	//set up spherical stuff
	int count = 0;
	const float piOverHeight = pi / (height + 1);
	const float twoPiOverWidth = twopi / width;
	float phi = 0;
	float sinPhi, theta;

	
	float* xcos = new float[width];
	float* ysin = new float[width];

	theta = 0;


	for (int x = 0; x < width; x = x + 1)
	{
		theta = theta + twoPiOverWidth;
		xcos[x] = cosf(theta);
		ysin[x] = sinf(theta);
	}

	unsigned cpuCount = std::thread::hardware_concurrency();
	
	std::thread* threads = new std::thread[cpuCount];
	int start = 0;
	float* min = new float[cpuCount];
	float* max = new float[cpuCount];
	for (int i = 0; i < cpuCount; i++)
	{
		Settings* S = new Settings();
		initSIMD(S, frequency, lacunarity, offset, gain, octaves);
		int end = start + (height / cpuCount);
		//SphereSurfaceNoiseSIMDThread(int start, int end,int width, int height, Settings* S, SIMD* min, SIMD *max, float* xcos, float* ysin, ISIMDNoise noiseFunction, SIMD* result)
		threads[i] = std::thread(SphereSurfaceNoiseSIMDThread, start, end, width, height, S, xcos, ysin, noiseFunction, result,&min[i],&max[i]);
		start = end;
	}

	*outMin = 999;
	*outMax = -999;
	for (int i = 0; i < cpuCount; i++)
	{
		threads[i].join();
		*outMin = fminf(*outMin, min[i]);
		*outMax = fmaxf(*outMax, max[i]);
	}
	
	
	return (float*)result;

}

float* GetSphereSurfaceNoise(int width, int height, int octaves, float lacunarity, float frequency, float gain, float offset, int noiseType, float *outMin, float *outMax)
{

	float* result = (float*)malloc(width*height*  sizeof(float));
	INoise noiseFunction;

	switch ((NoiseType)noiseType)
	{
	case FBM: noiseFunction = fbm3d; break;
	case TURBULENCE: noiseFunction = turbulence3d; break;
	case RIDGE: noiseFunction = ridge3d; break;
	case PLAIN: noiseFunction = plain3d; break;
	
	default:return 0;
	}

	//set up spherical stuff
	int count = 0;
	const float piOverHeight = pi / (height + 1);
	const float twoPiOverWidth = twopi / width;
	float phi = 0;
	float x3d, y3d, z3d;
	float sinPhi, theta;

	*outMin = 999;
	*outMax = -999;

	float* xcos = new float[width];
	float* ysin = new float[width];
	//Precalculate cos/sin	
	theta = 0;
	for (int x = 0; x < width; x = x + 1)
	{
		theta = theta + twoPiOverWidth;
		ysin[x] = sinf(theta);
		xcos[x] = cosf(theta);		
	}

	for (int y = 0; y < height; y = y + 1)
	{
		phi = phi + piOverHeight;
		z3d = cosf(phi);
		sinPhi = sinf(phi);
				
		for (int x = 0; x < width; x = x + 1)
		{		
			//use cos/sin lookup tables
			x3d = xcos[x] * sinPhi;
			y3d = ysin[x] * sinPhi;

			result[count] = noiseFunction(x3d, y3d, z3d, frequency, lacunarity, gain, octaves, offset);

			*outMin = fminf(*outMin, result[count]);
			*outMax = fmaxf(*outMax, result[count]);
			count = count + 1;

		}
	}

	return result;

}

