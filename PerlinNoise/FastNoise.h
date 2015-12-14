#include <xmmintrin.h> //SSE
#include <emmintrin.h> //SSE 2
#include <smmintrin.h> // SSE4.1
#include <immintrin.h> //avx2
//#include <zmmintrin.h> //avx512 the world is not yet ready...SOON


/**  This code is a distant Derivative of noise code by  Stefan Gustavson (stegu@itn.liu.se)
*    Jack Mott - SIMD conversion, addition of fbm, turbulence, and ridge variations
**/


// For non SIMD only
#define FADE(t) ( t * t * t * ( t * ( t * 6 - 15 ) + 10 ) )
#define FASTFLOOR(x) ( ((x)>0) ? ((int)x) : ((int)x-1 ) )
#define LERP(t, a, b) ((a) + (t)*((b)-(a)))

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
#define Gather(x,y,z) _mm256_i32gather_epi32(x,y,z)
#endif

//We use this union hack for easy
//access to the floats for unvectorizeable
//lookup table access
typedef union uSIMDi {
	SIMDi m;
	int a[VECTOR_SIZE];
};

typedef union uSIMD {
	SIMD m;
	float a[VECTOR_SIZE];
};


typedef struct
{
	union uSIMD x;
	union uSIMD y;
	union uSIMD z;
	SIMD frequency;
	SIMD lacunarity;
	SIMD offset;
	SIMD gain;
	unsigned char octaves;
} Settings;


void fbmSIMD(SIMD* out, Settings* S);
void plainSIMD(SIMD* out, Settings* S);
float fbm(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset);
float plain(float x, float y, float z, float frequency, float lacunarity, float gain, int octaves, float offset);

typedef void(*ISIMDNoise)(SIMD*, Settings*);
typedef float(*INoise)(float, float, float, float, float, float, int, float);


SIMDi zeroi, one, two, four, eight, twelve, fourteen, fifteeni, ff;
SIMD minusonef, zero, onef, six, fifteen, ten, scale;

const float pi = 3.14159265359f;
const float twopi = 6.2831853f;

