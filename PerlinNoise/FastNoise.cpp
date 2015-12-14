#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "FastNoise.h"

void initSIMD(Settings *S, float frequency, float lacunarity, float offset, float gain, int octaves)
{
	S->frequency = SetOne(frequency);
	S->lacunarity = SetOne(lacunarity);
	S->offset = SetOne(offset);
	S->gain = SetOne(gain);
	S->octaves = octaves;

	//integer constants	
	zeroi = SetOnei(0);
	one = SetOnei(1);
	two = SetOnei(2);
	four = SetOnei(4);
	eight = SetOnei(8);
	twelve = SetOnei(12);
	fourteen = SetOnei(14);
	fifteeni = SetOnei(15);
	ff = SetOnei(0xff);

	//float constants
	minusonef = SetOne(-1);
	zero = SetZero();
	onef = SetOne(1);
	six = SetOne(6);
	ten = SetOne(10);
	fifteen = SetOne(15);

	//final scaling constant
	scale = SetOne(.936f);

}


#ifndef USEGATHER
const unsigned char perm[] =
#endif
#ifdef USEGATHER
const int32_t perm[] =
#endif
{ 151,160,137,91,90,15,
131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
151,160,137,91,90,15,
131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
};

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

inline float grad(int hash, float x, float y, float z) {
	int h = hash & 15;     // Convert low 4 bits of hash code into 12 simple
	float u = h < 8 ? x : y; // gradient directions, and compute dot product.
	float v = h < 4 ? y : h == 12 || h == 14 ? x : z; // Fix repeats at h = 12 to 15
	return ((h & 1) ? -u : u) + ((h & 2) ? -v : v);
}


inline SIMD  gradV(SIMDi *hash, SIMD *x, SIMD *y, SIMD *z) {

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


inline SIMD noiseSIMD(SIMD* x, SIMD* y, SIMD* z)
{
	union uSIMDi ix0, iy0, ix1, iy1, iz0, iz1;
	SIMD fx0, fy0, fz0, fx1, fy1, fz1;

	//use built in floor if we have it
#ifdef SSE41
	ix0.m = ConvertToInt(Floor(*x));
	iy0.m = ConvertToInt(Floor(*y));
	iz0.m = ConvertToInt(Floor(*z));
#endif
	//drop out to scalar if we don't
#ifndef SSE41
	union uSIMD* ux = x;
	union uSIMD* uy = y;
	union uSIMD* uz = z;
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


	union uSIMDi p[8];
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


	SIMD nxy0 = gradV(&p[0].m, &fx0, &fy0, &fz0);
	SIMD nxy1 = gradV(&p[1].m, &fx0, &fy0, &fz1);
	SIMD nx0 = Add(nxy0, Mul(r, Sub(nxy1, nxy0)));

	nxy0 = gradV(&p[2].m, &fx0, &fy1, &fz0);
	nxy1 = gradV(&p[3].m, &fx0, &fy1, &fz1);
	SIMD nx1 = Add(nxy0, Mul(r, Sub(nxy1, nxy0)));

	SIMD n0 = Add(nx0, Mul(t, Sub(nx1, nx0)));

	nxy0 = gradV(&p[4].m, &fx1, &fy0, &fz0);
	nxy1 = gradV(&p[5].m, &fx1, &fy0, &fz1);
	nx0 = Add(nxy0, Mul(r, Sub(nxy1, nxy0)));

	nxy0 = gradV(&p[6].m, &fx1, &fy1, &fz0);
	nxy1 = gradV(&p[7].m, &fx1, &fy1, &fz1);
	nx1 = Add(nxy0, Mul(r, Sub(nxy1, nxy0)));

	SIMD n1 = Add(nx0, Mul(t, Sub(nx1, nx0)));

	n1 = Add(n0, Mul(s, Sub(n1, n0)));
	return Mul(scale, n1);



}



//---------------------------------------------------------------------
/** 3D float Perlin noise.
*/
float noise(float x, float y, float z)
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



	nxy0 = grad(perm[ix0 + perm[iy0 + perm[iz0]]], fx0, fy0, fz0);
	nxy1 = grad(perm[ix0 + perm[iy0 + perm[iz1]]], fx0, fy0, fz1);
	nx0 = LERP(r, nxy0, nxy1);

	nxy0 = grad(perm[ix0 + perm[iy1 + perm[iz0]]], fx0, fy1, fz0);
	nxy1 = grad(perm[ix0 + perm[iy1 + perm[iz1]]], fx0, fy1, fz1);
	nx1 = LERP(r, nxy0, nxy1);

	n0 = LERP(t, nx0, nx1);

	nxy0 = grad(perm[ix1 + perm[iy0 + perm[iz0]]], fx1, fy0, fz0);
	nxy1 = grad(perm[ix1 + perm[iy0 + perm[iz1]]], fx1, fy0, fz1);
	nx0 = LERP(r, nxy0, nxy1);

	nxy0 = grad(perm[ix1 + perm[iy1 + perm[iz0]]], fx1, fy1, fz0);
	nxy1 = grad(perm[ix1 + perm[iy1 + perm[iz1]]], fx1, fy1, fz1);
	nx1 = LERP(r, nxy0, nxy1);

	n1 = LERP(t, nx0, nx1);

	return 0.936f * (LERP(s, n0, n1));
}

//If you ever call something with 1 octave, call this instead
inline void plainSIMD(SIMD* out, Settings* S)
{	
		SIMD vfx = Mul(S->x.m, S->frequency);
		SIMD vfy = Mul(S->y.m, S->frequency);
		SIMD vfz = Mul(S->z.m, S->frequency);
		*out = noiseSIMD(&vfx, &vfy, &vfz);		
}

inline float plain(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset)
{
	return noise(x*frequency, y*frequency, z*frequency);
}


//Fractal brownian motions using SIMD
inline void fbmSIMD(SIMD* out, Settings* S)
{
	SIMD amplitude, localFrequency;
	*out = SetZero();
	amplitude = SetOne(0.5f);
	localFrequency = Load((const float*)&S->frequency);
	for (int i = S->octaves; i != 0; i--)
	{
		SIMD vfx = Mul(S->x.m, localFrequency);
		SIMD vfy = Mul(S->y.m, localFrequency);
		SIMD vfz = Mul(S->z.m, localFrequency);
		*out = Add(*out, Mul(amplitude, noiseSIMD(&vfx, &vfy, &vfz)));
		localFrequency = Mul(localFrequency, S->lacunarity);
		amplitude = Mul(amplitude, S->gain);
	}


}

//fractal brownian motion without SIMD
inline float fbm(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset)
{
	float sum = 0;
	float amplitude = 0.5f;
	for (int i = octaves; i != 0; i--)
	{
		sum += noise(x*frequency, y*frequency, z*frequency)*amplitude;
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}


//turbulence  using SIMD
inline void turbulenceSIMD(SIMD* out, Settings* S)
{
	SIMD amplitude, localFrequency;
	*out = SetZero();
	amplitude = SetOne(1.0f);
	localFrequency = Load((const float*)&S->frequency);
	for (int i = S->octaves; i != 0; i--)
	{
		SIMD r = Mul(amplitude, noiseSIMD(&S->x.m, &S->y.m, &S->z.m));
		//get abs of r by trickery
		r = Max(Sub(zero, r), r);
		*out = Add(*out, r);
		localFrequency = Mul(localFrequency, S->lacunarity);
		amplitude = Mul(amplitude, S->gain);
	}

}

inline float turbulence(float x, float y, float z, float lacunarity, float gain, float frequency, float offset, int octaves)
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


inline void ridgeSIMD(SIMD* out, Settings* S)
{
	SIMD amplitude, prev, localFrequency;
	*out = SetZero();
	amplitude = SetOne(1.0f);
	prev = SetOne(1.0f);
	localFrequency = Load((const float*)&S->frequency);
	for (int i = S->octaves; i != 0; i--)
	{
		SIMD r = noiseSIMD(&S->x.m, &S->y.m, &S->z.m);
		//get abs of r by trickery
		r = Max(Sub(zero, r), r);
		r = Sub(S->offset, r);
		r = Mul(r, r);
		r = Mul(r, amplitude);
		r = Mul(r, prev);
		*out = Add(*out, r);
		localFrequency = Mul(localFrequency, S->lacunarity);
		amplitude = Mul(amplitude, S->gain);
	}

}

inline float ridge(float x, float y, float z, float lacunarity, float gain, float frequency, float offset, int octaves)
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
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}


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
		for (int j = 0; j < VECTOR_SIZE; j++)
		{
			S.z.a[j] = z3d;
		}
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


int different(float a, float b)
{

	float threshold = 0.0001f;
	if ((a - b < threshold) && (a - b > -threshold))
	{
		return 0;
	}
	else {
		return 1;
	}
}

//Tests that SIMD and non SIMD versions get the same result
//over a variety of inputs and settings
void test(INoise noise, ISIMDNoise simdNoise)
{

	float frequency, lacunarity, octaves, gain, offset;
	Settings S;

	//0,0,0
	float x, y, z;

	for (int i = 1; i < 4; i++)
	{
		frequency = i;
		lacunarity = i;
		octaves = i;
		gain = i / 10.0f;
		offset = 0;
		initSIMD(&S, frequency, lacunarity, offset, gain, octaves);

		x = 0; y = 0; z = 0;
		S.x.m = SetOne(x);
		S.y.m = SetOne(y);
		S.z.m = SetOne(z);

		float r = noise(x, y, z, frequency, lacunarity, gain, octaves, 0);
		union uSIMD simdR;
		simdNoise((SIMD*)&simdR, &S);

		if (different(r, simdR.a[0])) printf("(%.4f,%.4f,%.4f): scalar:%.9f  SIMD:%.9f\n", x, y, z, r, simdR.a[0]);

		//1,1,1	
		x = 1; y = 1; z = 1;
		S.x.m = SetOne(x);
		S.y.m = SetOne(y);
		S.z.m = SetOne(z);

		r = noise(x, y, z, frequency, lacunarity, gain, octaves, 0);
		simdNoise((SIMD*)&simdR, &S);

		if (different(r, simdR.a[0])) printf("(%.4f,%.4f,%.4f): scalar:%.9f  SIMD:%.9f\n", x, y, z, r, simdR.a[0]);

		//-1,-1,-1	
		x = -1; y = -1; z = -1;
		S.x.m = SetOne(x);
		S.y.m = SetOne(y);
		S.z.m = SetOne(z);

		r = noise(x, y, z, frequency, lacunarity, gain, octaves, 0);
		simdNoise((SIMD*)&simdR, &S);

		if (different(r, simdR.a[0])) printf("(%.4f,%.4f,%.4f): scalar:%.9f  SIMD:%.9f\n", x, y, z, r, simdR.a[0]);


		x = .5; y = .5; z = .5;
		S.x.m = SetOne(x);
		S.y.m = SetOne(y);
		S.z.m = SetOne(z);

		r = noise(x, y, z, frequency, lacunarity, gain, octaves, 0);
		simdNoise((SIMD*)&simdR, &S);

		if (different(r, simdR.a[0])) printf("(%.4f,%.4f,%.4f): scalar:%.9f  SIMD:%.9f\n", x, y, z, r, simdR.a[0]);


		x = -.5; y = -.5; z = -.5;
		S.x.m = SetOne(x);
		S.y.m = SetOne(y);
		S.z.m = SetOne(z);

		r = noise(x, y, z, frequency, lacunarity, gain, octaves, 0);
		simdNoise((SIMD*)&simdR, &S);

		if (different(r, simdR.a[0])) printf("(%.4f,%.4f,%.4f): scalar:%.9f  SIMD:%.9f\n", x, y, z, r, simdR.a[0]);


		x = .1; y = .2; z = .3;
		S.x.m = SetOne(x);
		S.y.m = SetOne(y);
		S.z.m = SetOne(z);

		r = noise(x, y, z, frequency, lacunarity, gain, octaves, 0);
		simdNoise((SIMD*)&simdR, &S);

		if (different(r, simdR.a[0])) printf("(%.4f,%.4f,%.4f): scalar:%.9f  SIMD:%.9f\n", x, y, z, r, simdR.a[0]);


		x = .3; y = .2; z = .1;
		S.x.m = SetOne(x);
		S.y.m = SetOne(y);
		S.z.m = SetOne(z);

		r = noise(x, y, z, frequency, lacunarity, gain, octaves, 0);
		simdNoise((SIMD*)&simdR, &S);

		if (different(r, simdR.a[0])) printf("(%.4f,%.4f,%.4f): scalar:%.9f  SIMD:%.9f\n", x, y, z, r, simdR.a[0]);


		x = -.3; y = .2; z = -.1;
		S.x.m = SetOne(x);
		S.y.m = SetOne(y);
		S.z.m = SetOne(z);

		r = noise(x, y, z, frequency, lacunarity, gain, octaves, 0);
		simdNoise((SIMD*)&simdR, &S);

		if (different(r, simdR.a[0])) printf("(%.4f,%.4f,%.4f): scalar:%.9f  SIMD:%.9f\n", x, y, z, r, simdR.a[0]);
	}
	printf("test complete\n");
}


int main()
{

	//always test
	test(plain, plainSIMD);

	clock_t t1, t2;
	t1 = clock();
		
	float * r = GetSpherSurfaceNoiseSIMD(4096, 2048, 3, 2, 1, .5, 0,fbmSIMD);
	
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


