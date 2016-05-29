#pragma once
#ifndef FASTNOISE_H
#define FASTNOISE_H

#define FAST_NOISE_DLL_API __declspec(dllexport)

#include <stdint.h>

#define SSE2  //indicates we want SSE2
#define SSE41 //indicates we want SSE4.1 instructions (floor and blend is available)
#define AVX2  //indicates we want AVX2 instructions (double speed!) 


#ifndef AVX2
#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE 2
#endif

#ifdef SSE41
#include <smmintrin.h> // SSE4.1
#endif

#ifdef AVX2
#include <immintrin.h> //avx2
#endif


//#include <zmmintrin.h> //avx512 the world is not yet ready...SOON


/**  This code is a distant Derivative of noise code by  Stefan Gustavson (stegu@itn.liu.se)
*    Jack Mott - SIMD conversion, addition of fbm, turbulence, and ridge variations
**/




// create types we can use in either the 128 or 256 case
#ifndef AVX2
// m128 will be our base type
typedef __m128 SIMD;   //for floats
typedef __m128i SIMDi; //for integers

//we process 4 at a time
#define VECTOR_SIZE 4
#define MEMORY_ALIGNMENT 16

//intrinsic functions
#define Store(x,y) _mm_store_ps(x,y)
#define Load(x) _mm_load_ps(x)
#define SetOne(x) _mm_set1_ps(x)
#define SetZero() _mm_setzero_ps()
#define SetOnei(x) _mm_set1_epi32(x)
#define SetZeroi() _mm_setzero_epi32()
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
#define GreaterThanOrEq(x,y) _mm_cmpge_ps(x,y)
#define LessThan(x,y) _mm_cmplt_ps(x,y)
#define LessThani(x,y) _mm_cmpgt_epi32(y,x) 
#define LessThanOrEq(x,y) _mm_cmple_ps(x,y)
#define NotEqual(x,y) _mm_cmpneq_ps(x,y)
#ifdef SSE41
#define Floor(x) _mm_floor_ps(x)
#endif
#define Max(x,y) _mm_max_ps(x,y)
#define Maxi(x,y) _mm_max_epi32(x,y)
#define Min(x,y) _mm_min_ps(x,y)
#ifndef SSE41
#define BlendV(x,y,z) Or(AndNot(z,x), And(z,y))   //if we don't have sse4
#else
#define BlendV(x,y,z) _mm_blendv_ps(x,y,z)	
#endif
	
#endif
#ifdef AVX2

// m256 will be our base type
typedef __m256 SIMD;  //for floats
typedef __m256i SIMDi; //for integers

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
#define SetZeroi() _mm256_setzero_epi32()
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
#define LessThanOrEq(x,y) _mm256_cmp_ps(x,y,_CMP_LE_OQ)
#define GreaterThanOrEq(x,y) _mm256_cmp_ps(x,y,_CMP_GE_OQ)
#define NotEqual(x,y) _mm256_cmp_ps(x,y,_CMP_NEQ_OQ)
#define Floor(x) _mm256_floor_ps(x)
#define Max(x,y) _mm256_max_ps(x,y)
#define Maxi(x,y) _mm256_max_epi32(x,y)
#define Min(x,y) _mm256_min_ps(x,y)
#define Gather(x,y,z) _mm256_i32gather_epi32(x,y,z)
#define Gatherf(x,y,z) _mm256_i32gather_ps(x,y,z)
#define BlendV(x,y,z) _mm256_blendv_ps(x,y,z)
#endif


#define PI 3.141593f
#define TWOPI 6.283185f


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


enum FractalType { FBM, TURBULENCE, RIDGE, PLAIN};
enum NoiseType {PERLIN, SIMPLEX};



typedef SIMD(*ISIMDNoise3d)(SIMD* x, SIMD* y, SIMD* z);
typedef float(*INoise3d)(float x, float y, float z);

typedef void(*ISIMDFractal3d)(SIMD* out,Settings*,ISIMDNoise3d);
typedef float(*IFractal3d)(float, float, float, float, float, float, int, float,INoise3d);


extern SIMDi zeroi, one, two, four, eight, twelve, fourteen, fifteeni, ff;
extern SIMD minusonef, zero,psix, onef, six, fifteen, ten, thirtytwo,F3, G3,G32,G33;


FAST_NOISE_DLL_API inline extern void initSIMD(Settings * __restrict S, float frequency, float lacunarity, float offset, float gain, int octaves);
FAST_NOISE_DLL_API inline extern void initSIMDSimplex();




const float gradX[] =
{
	1,-1, 1,-1,
	1,-1, 1,-1,
	0, 0, 0, 0
};

const float gradY[] =
{
	1, 1,-1,-1,
	0, 0, 0, 0,
	1,-1, 1,-1
};

const float gradZ[] =
{
	0, 0, 0, 0,
	1, 1,-1,-1,
	1, 1,-1,-1
};


#ifndef AVX2
const unsigned char perm[] =
#else
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

//Used for simplex
#ifndef AVX2
const unsigned char permMOD12[] =
#else
const int32_t permMOD12[] =
#endif
{
7, 4, 5, 7, 6, 3, 11, 1, 9, 11, 0, 5, 2, 5, 7, 9, 8, 0, 7, 6, 9, 10, 8, 3,
1, 0, 9, 10, 11, 10, 6, 4, 7, 0, 6, 3, 0, 2, 5, 2, 10, 0, 3, 11, 9, 11, 11,
8, 9, 9, 9, 4, 9, 5, 8, 3, 6, 8, 5, 4, 3, 0, 8, 7, 2, 9, 11, 2, 7, 0, 3, 10,
5, 2, 2, 3, 11, 3, 1, 2, 0, 7, 1, 2, 4, 9, 8, 5, 7, 10, 5, 4, 4, 6, 11, 6,
5, 1, 3, 5, 1, 0, 8, 1, 5, 4, 0, 7, 4, 5, 6, 1, 8, 4, 3, 10, 8, 8, 3, 2, 8,
4, 1, 6, 5, 6, 3, 4, 4, 1, 10, 10, 4, 3, 5, 10, 2, 3, 10, 6, 3, 10, 1, 8, 3,
2, 11, 11, 11, 4, 10, 5, 2, 9, 4, 6, 7, 3, 2, 9, 11, 8, 8, 2, 8, 10, 7, 10, 5,
9, 5, 11, 11, 7, 4, 9, 9, 10, 3, 1, 7, 2, 0, 2, 7, 5, 8, 4, 10, 5, 4, 8, 2, 6,
1, 0, 11, 10, 2, 1, 10, 6, 0, 0, 11, 11, 6, 1, 9, 3, 1, 7, 9, 2, 11, 11, 1, 0,
10, 7, 1, 7, 10, 1, 4, 0, 0, 8, 7, 1, 2, 9, 7, 4, 6, 2, 6, 8, 1, 9, 6, 6, 7, 5,
0, 0, 3, 9, 8, 3, 6, 6, 11, 1, 0, 0,
7, 4, 5, 7, 6, 3, 11, 1, 9, 11, 0, 5, 2, 5, 7, 9, 8, 0, 7, 6, 9, 10, 8, 3,
1, 0, 9, 10, 11, 10, 6, 4, 7, 0, 6, 3, 0, 2, 5, 2, 10, 0, 3, 11, 9, 11, 11,
8, 9, 9, 9, 4, 9, 5, 8, 3, 6, 8, 5, 4, 3, 0, 8, 7, 2, 9, 11, 2, 7, 0, 3, 10,
5, 2, 2, 3, 11, 3, 1, 2, 0, 7, 1, 2, 4, 9, 8, 5, 7, 10, 5, 4, 4, 6, 11, 6,
5, 1, 3, 5, 1, 0, 8, 1, 5, 4, 0, 7, 4, 5, 6, 1, 8, 4, 3, 10, 8, 8, 3, 2, 8,
4, 1, 6, 5, 6, 3, 4, 4, 1, 10, 10, 4, 3, 5, 10, 2, 3, 10, 6, 3, 10, 1, 8, 3,
2, 11, 11, 11, 4, 10, 5, 2, 9, 4, 6, 7, 3, 2, 9, 11, 8, 8, 2, 8, 10, 7, 10, 5,
9, 5, 11, 11, 7, 4, 9, 9, 10, 3, 1, 7, 2, 0, 2, 7, 5, 8, 4, 10, 5, 4, 8, 2, 6,
1, 0, 11, 10, 2, 1, 10, 6, 0, 0, 11, 11, 6, 1, 9, 3, 1, 7, 9, 2, 11, 11, 1, 0,
10, 7, 1, 7, 10, 1, 4, 0, 0, 8, 7, 1, 2, 9, 7, 4, 6, 2, 6, 8, 1, 9, 6, 6, 7, 5,
0, 0, 3, 9, 8, 3, 6, 6, 11, 1, 0, 0
};