#include <time.h>
#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include "headers\FastNoise.h"



SIMDi zeroi, one, two, four, eight, twelve, fourteen, fifteeni, ff;
SIMD minusonef, zero,psix, onef, six, fifteen, ten,thirtytwo,F3, G3,G32,G33;

extern inline void initSIMD(Settings * __restrict S, float frequency, float lacunarity, float offset, float gain, int octaves)
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
				
}

inline void initSIMDSimplex()
{
	F3 = SetOne(1.0f / 3.0f);
	G3 = SetOne(1.0f / 6.0f);
	G32 = SetOne((1.0f / 6.0f) * 2.0f);
	G33 = SetOne((1.0f / 6.0f) * 3.0f);

	psix = SetOne(0.6);
	thirtytwo = SetOne(32.0);



}