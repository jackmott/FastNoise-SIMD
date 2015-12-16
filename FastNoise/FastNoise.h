#pragma once
#ifndef FASTNOISE_H
#define FASTNOISE_H

#define FAST_NOISE_DLL_API __declspec(dllexport)


#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE 2
#include <smmintrin.h> // SSE4.1
#include <immintrin.h> //avx2
#include <stdint.h>
//#include <zmmintrin.h> //avx512 the world is not yet ready...SOON


/**  This code is a distant Derivative of noise code by  Stefan Gustavson (stegu@itn.liu.se)
*    Jack Mott - SIMD conversion, addition of fbm, turbulence, and ridge variations
**/


// For non SIMD only
#define FADE(t) ( t * t * t * ( t * ( t * 6 - 15 ) + 10 ) )
#define DERIVFADE(t) (t * t * ( t *(30 * t - 60 ) + 30) )
#define FASTFLOOR(x) ( ((x)>0) ? ((int)x) : ((int)x-1 ) )
#define LERP(t, a, b) ((a) + (t)*((b)-(a)))

#define SCALE 1.754f
#define OFFSET .05f

#define SSE2  //indicates we want SSE2
#define SSE41 //indicates we want SSE4.1 instructions (floor is available)
#define AVX2 //indicates we want AVX2 instructions (double speed!) 
#define USEGATHER  //use the avx gather instruction to index the perm array

//creat types we can use in either the 128 or 256 case
#ifndef AVX2
// m128 will be our base type
typedef __m128 SIMD;
typedef __m128i SIMDi;

//we process 4 at a time
#define VECTOR_SIZE 4
#define MEMORY_ALIGNMENT 16

//intrinsic functions
#define Store(x,y) _mm_store_ps(x,y)
#define Load(x) _mm_load_ps(x)
#define SetOne(x) _mm_set1_ps(x)
#define SetZero() _mm_setzero_ps()
#define SetOnei(x) _mm_set1_epi32(x)
#define SetZeroi(x) _mm_setzero_epi32(x)
#define Add(x,y) _mm_add_ps(x,y)
#define Sub(x,y) _mm_sub_ps(x,y)
#define Addi(x,y) _mm_add_epi32(x,y)
#define Subi(x,y) _mm_sub_epi32(x,y)
#define Mul(x,y) _mm_mul_ps(x,y)
#define Muli(x,y) _mm_mul_epi32(x,y)
#define And(x,y) _mm_and_ps(x,y)
#define Andi(x,y) _mm_and_si128(x,y)
#define AndNot(x,y) _mm_andnot_ps(x,y)
#define Or(x,y) _mm_or_ps(x,y)
#define Ori(x,y) _mm_or_si128(x,y)
#define CastToFloat(x) _mm_castsi128_ps(x)
#define CastToInt(x) _mm_castps_si128(x)
#define ConvertToInt(x) _mm_cvtps_epi32(x)
#define ConvertToFloat(x) _mm_cvtepi32_ps(x)
#define Equal(x,y)  _mm_cmpeq_ps(x,y) 
#define Equali(x,y) _mm_cmpeq_epi32(x,y)
#define GreaterThan(x,y) _mm_cmpgt_ps(x,y)
#define GreaterThani(x,y) _mm_cmpgt_epi32(x,y)
#define LessThan(x,y) _mm_cmplt_ps(x,y)
#define LessThani(x,y) _mm_cmpgt_epi32(y,x) 
#define NotEqual(x,y) _mm_cmpneq_ps(x,y)
#define Floor(x) _mm_floor_ps(x)
#define Max(x,y) _mm_max_ps(x,y)
#define Maxi(x,y) _mm_max_epi32(x,y)
#define Min(x,y) _mm_min_ps(x,y)
#endif
#ifdef AVX2

// m256 will be our base type
typedef __m256 SIMD;
typedef __m256i SIMDi;

//process 8 at t time
#define VECTOR_SIZE 8
#define MEMORY_ALIGNMENT 32

//intrinsic functions
#define Store(x,y) _mm256_store_ps(x,y)
#define Load(x) _mm256_load_ps(x)
#define Set(x,y,z,w,a,b,c,d) _mm256_set_ps(x,y,z,w,a,b,c,d);
#define SetOne(x) _mm256_set1_ps(x)
#define SetZero() _mm256_setzero_ps()
#define SetOnei(x) _mm256_set1_epi32(x)
#define SetZeroi(x) _mm256_setzero_epi32(x)
#define Add(x,y) _mm256_add_ps(x,y)
#define Sub(x,y) _mm256_sub_ps(x,y)
#define Addi(x,y) _mm256_add_epi32(x,y)
#define Subi(x,y) _mm256_sub_epi32(x,y)
#define Mul(x,y) _mm256_mul_ps(x,y)
#define Muli(x,y) _mm256_mul_epi32(x,y)
#define And(x,y) _mm256_and_ps(x,y)
#define Andi(x,y) _mm256_and_si256(x,y)
#define AndNot(x,y) _mm256_andnot_ps(x,y)
#define Or(x,y) _mm256_or_ps(x,y)
#define Ori(x,y) _mm256_or_si256(x,y)
#define CastToFloat(x) _mm256_castsi256_ps(x)
#define CastToInt(x) _mm256_castps_si256(x)
#define ConvertToInt(x) _mm256_cvtps_epi32(x)
#define ConvertToFloat(x) _mm256_cvtepi32_ps(x)
#define Equal(x,y)  _mm256_cmp_ps(x,y,_CMP_EQ_OQ) 
#define Equali(x,y) _mm256_cmpeq_epi32(x,y)
#define GreaterThan(x,y) _mm256_cmp_ps(x,y,_CMP_GT_OQ)
#define GreaterThani(x,y) _mm256_cmpgt_epi32(x,y)
#define LessThan(x,y) _mm256_cmp_ps(x,y,_CMP_LT_OQ)
#define LessThani(x,y) _mm256_cmpgt_epi32(y,x) 
#define NotEqual(x,y) _mm256_cmp_ps(x,y,_CMP_NEQ_OQ)
#define Floor(x) _mm256_floor_ps(x)
#define Max(x,y) _mm256_max_ps(x,y)
#define Maxi(x,y) _mm256_max_epi32(x,y)
#define Min(x,y) _mm256_min_ps(x,y)
#define Gather(x,y,z) _mm256_i32gather_epi32(x,y,z)
#endif



//We use this union hack for easy
//access to the floats for unvectorizeable
//lookup table access
typedef union  {
	SIMDi m;
	int a[VECTOR_SIZE];
} uSIMDi;

typedef union  {
	SIMD m;
	float a[VECTOR_SIZE];
} uSIMD;


typedef struct
{
	uSIMD x;
	uSIMD y;
	uSIMD z;
	SIMD frequency;
	SIMD lacunarity;
	SIMD offset;
	SIMD gain;
	unsigned char octaves;
} Settings;


enum NoiseType { FBM, TURBULENCE, RIDGE, PLAIN,BILLOWY,RIDGE2};
extern SIMDi zeroi, one, two, four, eight, twelve, fourteen, fifteeni, ff;
extern SIMD minusonef, zero, onef, six, fifteen, ten, pscale, poffset;

typedef void(*ISIMDNoise)(SIMD*, Settings*);
typedef float(*INoise)(float, float, float, float, float, float, int, float);

FAST_NOISE_DLL_API inline extern void initSIMD(Settings *S, float frequency, float lacunarity, float offset, float gain, int octaves);

const float pi = 3.141593;
const float twopi = 6.283185;


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

#endif
