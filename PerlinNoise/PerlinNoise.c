#include <sys\timeb.h>
#include <stdio.h>
#include <math.h>
#include <xmmintrin.h>
#include <emmintrin.h> 
#include <smmintrin.h>


// Original Author: Stefan Gustavson (stegu@itn.liu.se)
//
// This library is public domain software, released by the author
// into the public domain in February 2011. You may do anything
// you like with it. You may even remove all attributions,
// but of course I'd appreciate it if you kept my name somewhere.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// General Public License for more details.

/** \file
\brief Implements the Noise1234 class for producing Perlin noise.
\author Stefan Gustavson (stegu@itn.liu.se)
*/

/*
* This implementation is "Improved Noise" as presented by
* Ken Perlin at Siggraph 2002. The 3D function is a direct port
* of his Java reference code available on www.noisemachine.com
* (although I cleaned it up and made the code more readable),
* but the 1D, 2D and 4D cases were implemented from scratch
* by me.
*
* This is a highly reusable class. It has no dependencies
* on any other file, apart from its own header file.
*/


/* Jack Mott - converted to SSE2/4 for 4x speedup and added
*  fractal brownian noise, also SSE enhanced
*/


// For non SIMD only
#define FADE(t) ( t * t * t * ( t * ( t * 6 - 15 ) + 10 ) )
#define FASTFLOOR(x) ( ((x)>0) ? ((int)x) : ((int)x-1 ) )
#define LERP(t, a, b) ((a) + (t)*((b)-(a)))


//We use this union hack for easy
//access to the floats for unvectorizeable
//lookup table access
union isimd {
	__m128i m;
	int a[4];
};

union fsimd {
	__m128 m;
	float a[4];
};


//constants
__m128i one, two, four, eight, twelve, fourteen, fifteeni, ff;
__m128 minusonef, zero, onef, six, fifteen, ten, scale;


const float pi = 3.14159265359f;
const float twopi = 6.2831853f;




typedef __m128 (*ISIMDNoise)(__m128*, __m128*, __m128*, __m128*, __m128*,__m128*, __m128*, int);
typedef float(*INoise)(float, float, float, float, float,float,float, int);

void init(int octaves)
{

	//integer constants	
	one = _mm_set1_epi32(1);
	two = _mm_set1_epi32(2);
	four = _mm_set1_epi32(4);
	eight = _mm_set1_epi32(8);
	twelve = _mm_set1_epi32(12);
	fourteen = _mm_set1_epi32(14);
	fifteeni = _mm_set1_epi32(15);
	ff = _mm_set1_epi32(0xff);

	//float constants
	minusonef = _mm_set1_ps(-1);
	zero = _mm_setzero_ps();
	onef = _mm_set1_ps(1);
	six = _mm_set1_ps(6);
	ten = _mm_set1_ps(10);
	fifteen = _mm_set1_ps(15);

	//final scaling constant
	scale = _mm_set1_ps(.936f);

}


//To scale output to [0..1]
inline void SetUpOffsetScale(int octaves, __m128* fbmOffset, __m128* fbmScale)
{
	switch (octaves)
	{
	case 1:
		*fbmOffset = _mm_setzero_ps();
		*fbmScale = _mm_set1_ps(1.066f);
		break;
	case 2:
		*fbmOffset = _mm_set1_ps(.073f);
		*fbmScale = _mm_set1_ps(.8584f);
		break;
	case 3:
		*fbmOffset = _mm_set1_ps(.1189f);
		*fbmScale = _mm_set1_ps(.8120f);
		break;
	case 4:
		*fbmOffset = _mm_set1_ps(.1440f);
		*fbmScale = _mm_set1_ps(.8083f);
		break;
	case 5:
		*fbmOffset = _mm_set1_ps(.1530f);
		*fbmScale = _mm_set1_ps(.8049f);
		break;
	default:
		*fbmOffset = _mm_set1_ps(.16f);
		*fbmScale = _mm_set1_ps(.8003f);
	}

}

//---------------------------------------------------------------------
// Static data

/*
* Permutation table. This is just a random jumble of all numbers 0-255,
* repeated twice to avoid wrapping the index at 255 for each lookup.
* This needs to be exactly the same for all instances on all platforms,
* so it's easiest to just keep it as static explicit data.
* This also removes the need for any initialisation of this class.
*
* Note that making this an int[] instead of a char[] might make the
* code run faster on platforms with a high penalty for unaligned single
* byte addressing. Intel x86 is generally single-byte-friendly, but
* some other CPUs are faster with 4-aligned reads.
* However, a char[] is smaller, which avoids cache trashing, and that
* is probably the most important aspect on most architectures.
* This array is accessed a *lot* by the noise functions.
* A vector-valued noise over 3D accesses it 96 times, and a
* float-valued 4D noise 64 times. We want this to fit in the cache!
*/
const unsigned char perm[] = { 151,160,137,91,90,15,
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


inline __m128  gradV(__m128i *hash, __m128 *x, __m128 *y, __m128 *z) {

	__m128i h = _mm_and_si128(*hash, fifteeni);
	__m128 h1 = _mm_cmpneq_ps(zero, _mm_castsi128_ps(_mm_and_si128(h, one)));
	__m128 h2 = _mm_cmpneq_ps(zero, _mm_castsi128_ps(_mm_and_si128(h, two)));


	//if h < 8 then x, else y
	__m128 u = _mm_castsi128_ps(_mm_cmplt_epi32(h, eight));
	u = _mm_or_ps(_mm_and_ps(u, *x), _mm_andnot_ps(u, *y));

	//if h < 4 then y else if h is 12 or 14 then x else z
	__m128 v = _mm_castsi128_ps(_mm_cmplt_epi32(h, four));
	__m128 h12o14 = _mm_cmpneq_ps(zero, _mm_castsi128_ps(_mm_or_si128(_mm_cmpeq_epi32(h, twelve), _mm_cmpeq_epi32(h, fourteen))));
	h12o14 = _mm_or_ps(_mm_and_ps(h12o14, *x), _mm_andnot_ps(h12o14, *z));
	v = _mm_or_ps(_mm_and_ps(v, *y), _mm_andnot_ps(v, h12o14));

	//get -u and -v
	__m128 minusU = _mm_sub_ps(zero,u);
	__m128 minusV = _mm_sub_ps(zero,v);

	//if h1 then -u else u
	u = _mm_or_ps(_mm_and_ps(h1, minusU), _mm_andnot_ps(h1, u));
	//if h2 then -v else v
	v = _mm_or_ps(_mm_and_ps(h2, minusV), _mm_andnot_ps(h2, v));
	return _mm_add_ps(u, v);
}


inline __m128 noiseSIMD(__m128* x, __m128* y, __m128* z)
{
	union isimd ix0, iy0, ix1, iy1, iz0, iz1;
	__m128 fx0, fy0, fz0, fx1, fy1, fz1;

	//mm_floor is the only SSE4 instruction
	//you can get SSE2 compatbility by rolling your own floor
	ix0.m = _mm_cvtps_epi32(_mm_floor_ps(*x));
	iy0.m = _mm_cvtps_epi32(_mm_floor_ps(*y));
	iz0.m = _mm_cvtps_epi32(_mm_floor_ps(*z));


	fx0 = _mm_sub_ps(*x, _mm_cvtepi32_ps(ix0.m));
	fy0 = _mm_sub_ps(*y, _mm_cvtepi32_ps(iy0.m));
	fz0 = _mm_sub_ps(*z, _mm_cvtepi32_ps(iz0.m));

	fx1 = _mm_sub_ps(fx0, onef);
	fy1 = _mm_sub_ps(fy0, onef);
	fz1 = _mm_sub_ps(fz0, onef);

	ix1.m = _mm_and_si128(_mm_add_epi32(ix0.m, one), ff);
	iy1.m = _mm_and_si128(_mm_add_epi32(iy0.m, one), ff);
	iz1.m = _mm_and_si128(_mm_add_epi32(iz0.m, one), ff);

	ix0.m = _mm_and_si128(ix0.m, ff);
	iy0.m = _mm_and_si128(iy0.m, ff);
	iz0.m = _mm_and_si128(iz0.m, ff);


	__m128
		r = _mm_mul_ps(fz0, six);
	r = _mm_sub_ps(r, fifteen);
	r = _mm_mul_ps(r, fz0);
	r = _mm_add_ps(r, ten);
	r = _mm_mul_ps(r, fz0);
	r = _mm_mul_ps(r, fz0);
	r = _mm_mul_ps(r, fz0);

	__m128
		t = _mm_mul_ps(fy0, six);
	t = _mm_sub_ps(t, fifteen);
	t = _mm_mul_ps(t, fy0);
	t = _mm_add_ps(t, ten);
	t = _mm_mul_ps(t, fy0);
	t = _mm_mul_ps(t, fy0);
	t = _mm_mul_ps(t, fy0);

	__m128
		s = _mm_mul_ps(fx0, six);
	s = _mm_sub_ps(s, fifteen);
	s = _mm_mul_ps(s, fx0);
	s = _mm_add_ps(s, ten);
	s = _mm_mul_ps(s, fx0);
	s = _mm_mul_ps(s, fx0);
	s = _mm_mul_ps(s, fx0);


	//This section may be vectorizeable with AVX gather instructions.	
	union isimd p1, p2, p3, p4, p5, p6, p7, p8;
	for (int i = 0; i < 4; i++)
	{
		p1.a[i] = perm[ix0.a[i] + perm[iy0.a[i] + perm[iz0.a[i]]]];
		p2.a[i] = perm[ix0.a[i] + perm[iy0.a[i] + perm[iz1.a[i]]]];

		p3.a[i] = perm[ix0.a[i] + perm[iy1.a[i] + perm[iz0.a[i]]]];
		p4.a[i] = perm[ix0.a[i] + perm[iy1.a[i] + perm[iz1.a[i]]]];

		p5.a[i] = perm[ix1.a[i] + perm[iy0.a[i] + perm[iz0.a[i]]]];
		p6.a[i] = perm[ix1.a[i] + perm[iy0.a[i] + perm[iz1.a[i]]]];

		p7.a[i] = perm[ix1.a[i] + perm[iy1.a[i] + perm[iz0.a[i]]]];
		p8.a[i] = perm[ix1.a[i] + perm[iy1.a[i] + perm[iz1.a[i]]]];

	}

	__m128 nxy0 = gradV(&p1.m, &fx0, &fy0, &fz0);
	__m128 nxy1 = gradV(&p2.m, &fx0, &fy0, &fz1);
	__m128 nx0 = _mm_add_ps(nxy0, _mm_mul_ps(r, _mm_sub_ps(nxy1, nxy0)));

	nxy0 = gradV(&p3.m, &fx0, &fy1, &fz0);
	nxy1 = gradV(&p4.m, &fx0, &fy1, &fz1);
	__m128 nx1 = _mm_add_ps(nxy0, _mm_mul_ps(r, _mm_sub_ps(nxy1, nxy0)));

	__m128 n0 = _mm_add_ps(nx0, _mm_mul_ps(t, _mm_sub_ps(nx1, nx0)));

	nxy0 = gradV(&p5.m, &fx1, &fy0, &fz0);
	nxy1 = gradV(&p6.m, &fx1, &fy0, &fz1);
	nx0 = _mm_add_ps(nxy0, _mm_mul_ps(r, _mm_sub_ps(nxy1, nxy0)));

	nxy0 = gradV(&p7.m, &fx1, &fy1, &fz0);
	nxy1 = gradV(&p8.m, &fx1, &fy1, &fz1);
	nx1 = _mm_add_ps(nxy0, _mm_mul_ps(r, _mm_sub_ps(nxy1, nxy0)));

	__m128 n1 = _mm_add_ps(nx0, _mm_mul_ps(t, _mm_sub_ps(nx1, nx0)));

	n1 = _mm_add_ps(n0, _mm_mul_ps(s, _mm_sub_ps(n1, n0)));
	return _mm_mul_ps(scale, n1);



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

//Fractal brownian motions using SIMD
inline __m128 fbmSIMD(__m128 *vX, __m128 *vY, __m128 *vZ, __m128* vLacunarity, __m128* vGain, __m128* vFrequency, __m128* vOffset, int octaves)
{
	__m128 vSum, vAmplitude;
	vSum = _mm_setzero_ps();	
	vAmplitude = _mm_set1_ps(0.5f);
	for (int i = 0; i < octaves; i++)
	{
		__m128 vfx = _mm_mul_ps(*vX,*vFrequency);
		__m128 vfy = _mm_mul_ps(*vY,*vFrequency);
		__m128 vfz = _mm_mul_ps(*vZ,*vFrequency);		
		vSum = _mm_add_ps(vSum, _mm_mul_ps(vAmplitude, noiseSIMD(&vfx, &vfy, &vfz)));
		*vFrequency = _mm_mul_ps(*vFrequency, *vLacunarity);
		vAmplitude = _mm_mul_ps(vAmplitude, *vGain);
	}


	return vSum;

}

//fractal brownian motion without SIMD
inline float fbm(float x, float y, float z, float lacunarity, float gain, float frequency, float offset, int octaves)
{
	float sum = 0;
	float amplitude = 1;	
	for (int i = 0; i < octaves; i++)
	{
		sum += noise(x, y, z)*frequency;
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}


//turbulence  using SIMD
inline __m128 turbulenceSIMD(__m128 *vX, __m128 *vY, __m128 *vZ, __m128* vLacunarity, __m128* vGain, __m128* vFrequency, __m128* vOffset, int octaves)
{
	__m128 vSum, vAmplitude;
	vSum = _mm_setzero_ps();
	vAmplitude = _mm_set1_ps(1.0f);
	for (int i = 0; i < octaves; i++)
	{
		__m128 vfx = _mm_mul_ps(*vX, *vFrequency);
		__m128 vfy = _mm_mul_ps(*vY, *vFrequency);
		__m128 vfz = _mm_mul_ps(*vZ, *vFrequency);
		__m128 r = _mm_mul_ps(vAmplitude, noiseSIMD(&vfx, &vfy, &vfz));
		//get abs of r by trickery
		r = _mm_max_ps(_mm_sub_ps(zero, r), r);
		vSum = _mm_add_ps( vSum,r);
		*vFrequency = _mm_mul_ps(*vFrequency, *vLacunarity);
		vAmplitude = _mm_mul_ps(vAmplitude, *vGain);
	}


	return vSum;

}

inline float turbulence(float x, float y, float z, float lacunarity, float gain, float frequency, float offset, int octaves)
{
	float sum = 0;	
	float amplitude = 1;
	for (int i = 0; i < octaves; i++)
	{
		sum += fabs(noise(x*frequency, y*frequency, z*frequency)*amplitude);
		frequency *= lacunarity;
		amplitude *= gain;
	}
	return sum;
}

//ridge noise SIMD
inline __m128 ridgeSIMD(__m128 *vX, __m128 *vY, __m128 *vZ, __m128* vLacunarity, __m128* vGain, __m128* vFrequency, __m128* vOffset, int octaves)
{
	__m128 vSum, vAmplitude, vPrev;
	vSum = _mm_setzero_ps();
	vAmplitude = _mm_set1_ps(1.0f);
	vPrev = _mm_set1_ps(1.0f);
	for (int i = 0; i < octaves; i++)
	{
		__m128 vfx = _mm_mul_ps(*vX, *vFrequency);
		__m128 vfy = _mm_mul_ps(*vY, *vFrequency);
		__m128 vfz = _mm_mul_ps(*vZ, *vFrequency);
		__m128 r =  noiseSIMD(&vfx, &vfy, &vfz);
		//get abs of r by trickery
		r = _mm_max_ps(_mm_sub_ps(zero, r), r);
		r = _mm_sub_ps(*vOffset, r);
		r = _mm_mul_ps(r, r);
		r = _mm_mul_ps(r, vAmplitude);
		r = _mm_mul_ps(r, vPrev);
		vSum = _mm_add_ps(vSum, r);
		*vFrequency = _mm_mul_ps(*vFrequency, *vLacunarity);
		vAmplitude = _mm_mul_ps(vAmplitude, *vGain);
	}


	return vSum;

}

inline float ridge(float x, float y, float z, float lacunarity, float gain, float frequency, float offset, int octaves)
{
	float sum = 0;
	float amplitude = 0.5f;
	float prev = 1.0f;
	for (int i = 0; i < octaves; i++)
	{
		float r = abs(noise(x*frequency, y*frequency, z*frequency));
		r= offset - r;
		r = r*r;
		sum += r*amplitude*prev;
		frequency *= lacunarity;
		amplitude *= gain;
	}
}

//---------------------------------------------------------------------
//Get noise on the surface of a sphere
//This could be SIMD up as wel, but was not clear to me what cpus 
//support SVML for the transcendental math. Not going to speed things up
//much either.

void CleanUpSphericalPerlinNoise(float * resultArray)
{
	_aligned_free(resultArray);
}

float* GetSphericalPerlinNoise(int width, int height, int octaves, float lacunarity, float gain, float stretch, float offsetx, float offsety, ISIMDNoise noise)
{
	init(octaves);

	float *result = (float*)_aligned_malloc(width*height*  sizeof(float), 16);

	//Debugging shows this keeps them closer together in memory
	float *VxStore = (float*)_aligned_malloc(3 * 4 * sizeof(float), 16);
	float *VyStore = VxStore + 4;
	float *VzStore = VxStore + 8;


	//set up spherical stuff
	int count = 0;
	const float piOverHeight = pi / height;
	const float twoPiOverWidth = twopi / width;
	float phi = 0;
	float x3d, y3d, z3d;
	float sinPhi, theta;

	__m128 fbmOffset, fbmScale;
	SetUpOffsetScale(octaves, &fbmOffset, &fbmScale);

	__m128 vGain = _mm_set1_ps(gain);
	__m128 vLacunarity = _mm_set1_ps(lacunarity);
	__m128 vFrequency = _mm_set1_ps(1);
	__m128 vOffset = _mm_set1_ps(1);

	for (int y = 0; y < height; y = y + 1)
	{
		phi = phi + piOverHeight;
		z3d = cosf(phi)*stretch;
		sinPhi = sinf(phi);
		theta = 0;
		for (int x = 0; x < width - 3; x = x + 4)
		{

			for (int j = 0; j < 4; j++)
			{
				theta = theta + twoPiOverWidth;
				x3d = cosf(theta) * sinPhi;
				y3d = sinf(theta) * sinPhi;

				VxStore[j] = x3d * 2 + offsetx;
				VyStore[j] = y3d * 2 + offsety;
				VzStore[j] = z3d * 2;

			}
			__m128 r = noise((__m128*)VxStore, (__m128*)VyStore, (__m128*)VzStore, &vGain, &vLacunarity,&vFrequency,&vOffset, octaves);
			_mm_store_ps(&result[count], _mm_mul_ps(_mm_add_ps(r, fbmOffset), fbmScale));
			count = count + 4;

		}
	}
	_aligned_free(VxStore);
	return result;

}


#define TEST_COUNT 4096
void testNoise(INoise n)
{
	struct timeb starttime, endtime;
	int diff;
	ftime(&starttime);

	float accumulator = 0;
	for (int x = 0; x < TEST_COUNT; x++)
	{
		for (int y = 0; y < TEST_COUNT; y++)
		{
			accumulator += n(x, y, x, 1, 1, 1,1,1);
		}
	}

	printf("%f\n", accumulator);
	ftime(&endtime);
	diff = (int)(1000.0 * (endtime.time - starttime.time)
		+ (endtime.millitm - starttime.millitm));

	printf("Time Taken for Non SIMD: %d\n", diff);
	
	
}

void testSIMDNoise(ISIMDNoise n)
{
	struct timeb starttime, endtime;
	int diff;
	ftime(&starttime);

	__m128 r;

	//Debugging shows this keeps them closer together in memory
	float *VxStore = (float*)_aligned_malloc(3 * 4 * sizeof(float), 16);
	float *VyStore = VxStore + 4;
	float *VzStore = VxStore + 8;
	__m128 result = _mm_setzero_ps();
	__m128 vGain = _mm_set1_ps(1);
	__m128 vLacunarity = _mm_set1_ps(1);
	__m128 vFrequency = _mm_set1_ps(1);
	__m128 vOffset = _mm_set1_ps(1);
	int octaves = 1;
	for (int x = 0; x < TEST_COUNT; x++)
	{
		for (int y = 0; y < TEST_COUNT - 3; y = y + 4)
		{
			for (int j = 0; j < 4; j++)
			{

				VxStore[j] = x;
				VyStore[j] = y;
				VzStore[j] = x;
			}

			result = _mm_add_ps(result, n((__m128*)VxStore, (__m128*)VyStore, (__m128*)VzStore, &vGain, &vLacunarity,&vFrequency,&vOffset, octaves));
		}
	}



	ftime(&endtime);
	diff = (int)(1000.0 * (endtime.time - starttime.time)
		+ (endtime.millitm - starttime.millitm));

	printf("Time Taken for SIMD: %d\n", diff);
	
}

int main()
{
	//to evaluate how fast we went			
	//float *x = GetSphericalPerlinNoise(4096, 4096, 3, 2, 0.5, 1, 0, 0,fbmSIMD);
	testNoise(fbm);
	testSIMDNoise(fbmSIMD);
	int xy;
	scanf("%d", &xy);


}


