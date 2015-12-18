#include "headers\FastNoise3d.h"

inline SIMD dot(SIMD x1, SIMD y1, SIMD z1, SIMD x2, SIMD y2, SIMD z2)
{
	SIMD xx = Mul(x1, x2);
	SIMD yy = Mul(y1, y2);
	SIMD zz = Mul(z1, z2);
	return Add(xx, Add(yy, zz));
}

inline SIMD simplexSIMD3d(SIMD* x, SIMD* y, SIMD* z) {
	uSIMDi i, j, k;

	uSIMD s;
	s.m = Mul(F3, Add(*x, Add(*y, *z)));

#ifdef SSE41
	i.m = ConvertToInt(Floor(Add(*x,s.m)));
	j.m = ConvertToInt(Floor(Add(*y,s.m)));
	k.m = ConvertToInt(Floor(Add(*z,s.m)));
#endif
	//drop out to scalar if we don't
#ifndef SSE41
	uSIMD* ux = x;
	uSIMD* uy = y;
	uSIMD* uz = z;
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		i.a[i] = FASTFLOOR((*ux).a[i]+s.a[i]);
		j.a[i] = FASTFLOOR((*uy).a[i]+s.a[i]);
		k.a[i] = FASTFLOOR((*uz).a[i]+s.a[i]);
	}
#endif

	SIMD t = Mul(ConvertToFloat(Addi(i.m, Addi(j.m, k.m))), G3);
	SIMD X0 = Sub(ConvertToFloat(i.m), t);
	SIMD Y0 = Sub(ConvertToFloat(j.m), t);
	SIMD Z0 = Sub(ConvertToFloat(k.m), t);
	SIMD x0 = Sub(*x, X0);
	SIMD y0 = Sub(*y, Y0);
	SIMD z0 = Sub(*z, Z0);


	//This determines what simplex we are in, in the irregular tetrahedron 
	// -(Stefan Gustavson (stegu@itn.liu.se).)
	//The following mess accomplishes this transofmration without branching
	//Because we can't branch in SIMD -Jack Mott
	/*       ijk1 ijk2
	x>y>z -> 100  110
	x>z>y -> 100  101
	z>x>y -> 001  101
	z>y>x -> 001  011
	y>z>x -> 010  011
	y>x>z -> 010  110
	*/
	uSIMDi i1, i2, j1, j2, k1, k2;
	i1.m = Andi(one, Andi(CastToInt(LessThan(y0, x0)), CastToInt(LessThan(z0, x0))));
	j1.m = Andi(one, Andi(CastToInt(LessThan(x0, y0)), CastToInt(LessThan(z0, y0))));
	k1.m = Andi(one, Andi(CastToInt(LessThan(x0, z0)), CastToInt(LessThan(y0, z0))));

	//for i2
	SIMDi yx_xz = Andi(CastToInt(LessThan(y0, x0)), CastToInt(LessThan(x0, z0)));
	SIMDi zx_xy = Andi(CastToInt(LessThan(z0, x0)), CastToInt(LessThan(x0, y0)));

	//for j2
	SIMDi xy_yz = Andi(CastToInt(LessThan(x0, y0)), CastToInt(LessThan(y0, z0)));
	SIMDi zy_yx = Andi(CastToInt(LessThan(z0, y0)), CastToInt(LessThan(y0, x0)));

	//for k2
	SIMDi yz_zx = Andi(CastToInt(LessThan(y0, z0)), CastToInt(LessThan(z0, x0)));
	SIMDi xz_zy = Andi(CastToInt(LessThan(x0, z0)), CastToInt(LessThan(z0, y0)));

	i2.m = Andi(one, Ori(i1.m, Ori(yx_xz, zx_xy)));
	j2.m = Andi(one, Ori(j1.m, Ori(xy_yz, zy_yx)));
	k2.m = Andi(one, Ori(k1.m, Ori(yz_zx, xz_zy)));

	// A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
	// a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
	// a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
	// c = 1/6. -Stefan Gustavson (stegu@itn.liu.se).
	SIMD x1 = Add(Sub(x0, ConvertToFloat(i1.m)), G3);
	SIMD y1 = Add(Sub(y0, ConvertToFloat(j1.m)), G3);
	SIMD z1 = Add(Sub(z0, ConvertToFloat(k1.m)), G3);
	SIMD x2 = Add(Sub(x0, ConvertToFloat(i2.m)), G32);
	SIMD y2 = Add(Sub(y0, ConvertToFloat(j2.m)), G32);
	SIMD z2 = Add(Sub(z0, ConvertToFloat(k2.m)), G32);
	SIMD x3 = Add(Sub(x0, onef), G33);
	SIMD y3 = Add(Sub(y0, onef), G33);
	SIMD z3 = Add(Sub(z0, onef), G33);


	uSIMDi ii;
	ii.m = Andi(i.m, ff);
	uSIMDi jj;
	jj.m = Andi(j.m, ff);
	uSIMDi kk;
	kk.m = Andi(k.m, ff);
	uSIMDi gi0, gi1, gi2, gi3;
#ifndef USEGATHER
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		gi0.a[i] = permMOD12[ii.a[i] + perm[jj.a[i] + perm[kk.a[i]]]];
		gi1.a[i] = permMOD12[ii.a[i] + i1.a[i] + perm[jj.a[i] + j1.a[i] + perm[kk.a[i]+k1.a[i]]]];
		gi2.a[i] = permMOD12[ii.a[i] + i2.a[i] + perm[jj.a[i] + j2.a[i] + perm[kk.a[i]+k2.a[i]]]];
		gi3.a[i] = permMOD12[ii.a[i] + 1 + perm[jj.a[i] + 1 + perm[kk.a[i]]]];
	}
#endif
#ifdef USEGATHER
	SIMDi pkk = Gather(perm, kk.m, 4);	
	SIMDi pkkk1 = Gather(perm, Addi(kk.m, k1.m), 4);
	SIMDi pkkk2 = Gather(perm, Addi(kk.m, k2.m), 4);
	SIMDi pkk1 = Gather(perm, Addi(kk.m, one), 4);

	SIMDi pjj = Gather(perm, Addi(jj.m, pkk), 4);
	SIMDi pjjj1 = Gather(perm, Addi(jj.m, Addi(j1.m, pkkk1)), 4);
	SIMDi pjjj2 = Gather(perm, Addi(jj.m, Addi(j2.m, pkkk2)), 4);
	SIMDi pjj1 = Gather(perm, Addi(jj.m, Addi(one, pkk1)), 4);


	gi0.m = Gather(permMOD12, Addi(ii.m, pjj), 4);
	gi1.m = Gather(permMOD12, Addi(i1.m,Addi(ii.m, pjjj1)), 4);
	gi2.m = Gather(permMOD12, Addi(i2.m,Addi(ii.m, pjjj2)), 4);
	gi3.m = Gather(permMOD12, Addi(one,Addi(ii.m, pjj1)), 4);
#endif

	//ti = .6 - xi*xi - yi*yi - zi*zi
	
	SIMD t0 = Sub(Sub(Sub(psix, Mul(x0, x0)), Mul(y0, y0)), Mul(z0, z0));
	SIMD t1 = Sub(Sub(Sub(psix, Mul(x1, x1)), Mul(y1, y1)), Mul(z1, z1));
	SIMD t2 = Sub(Sub(Sub(psix, Mul(x2, x2)), Mul(y2, y2)), Mul(z2, z2));
	SIMD t3 = Sub(Sub(Sub(psix, Mul(x3, x3)), Mul(y3, y3)), Mul(z3, z3));

	//ti*ti*ti*ti
	SIMD t0q = Mul(t0, t0);
	t0q = Mul(t0q, t0q);
	SIMD t1q = Mul(t1, t1);
	t1q = Mul(t1q, t1q);
	SIMD t2q = Mul(t2, t2);
	t2q = Mul(t2q, t2q);
	SIMD t3q = Mul(t3, t3);
	t3q = Mul(t3q, t3q);


	uSIMD
		gi0x, gi0y, gi0z,
		gi1x, gi1y, gi1z,
		gi2x, gi2y, gi2z,
		gi3x, gi3y, gi3z;
#ifndef USEGATHER
	for (int i = 0; i < VECTOR_SIZE; i++)
	{
		gi0x.a[i] = gradX[gi0.a[i]];
		gi0y.a[i] = gradY[gi0.a[i]];
		gi0z.a[i] = gradZ[gi0.a[i]];

		gi1x.a[i] = gradX[gi1.a[i]];
		gi1y.a[i] = gradY[gi1.a[i]];
		gi1z.a[i] = gradZ[gi1.a[i]];
		
		gi2x.a[i] = gradX[gi2.a[i]];
		gi2y.a[i] = gradY[gi2.a[i]];
		gi2z.a[i] = gradZ[gi2.a[i]];

		gi3x.a[i] = gradX[gi3.a[i]];
		gi3y.a[i] = gradY[gi3.a[i]];
		gi3z.a[i] = gradZ[gi3.a[i]];

	}
#endif
#ifdef USEGATHER
	gi0x.m = Gatherf(gradX, gi0.m, 4);
	gi0y.m = Gatherf(gradY, gi0.m, 4);
	gi0z.m = Gatherf(gradZ, gi0.m, 4);

	gi1x.m = Gatherf(gradX, gi1.m, 4);
	gi1y.m = Gatherf(gradY, gi1.m, 4);
	gi1z.m = Gatherf(gradZ, gi1.m, 4);

	gi2x.m = Gatherf(gradX, gi2.m, 4);
	gi2y.m = Gatherf(gradY, gi2.m, 4);
	gi2z.m = Gatherf(gradZ, gi2.m, 4);
	
	gi3x.m = Gatherf(gradX, gi3.m, 4);
	gi3y.m = Gatherf(gradY, gi3.m, 4);
	gi3z.m = Gatherf(gradZ, gi3.m, 4);
#endif

	SIMD n0 = Mul(t0q, dot(gi0x.m, gi0y.m, gi0z.m, x0, y0, z0));
	SIMD n1 = Mul(t1q, dot(gi1x.m, gi1y.m, gi1z.m, x1, y1, z1));
	SIMD n2 = Mul(t2q, dot(gi2x.m, gi2y.m, gi2z.m, x2, y2, z2));
	SIMD n3 = Mul(t3q, dot(gi3x.m, gi3y.m, gi3z.m, x3, y3, z3));



	//if ti < 0 then 0 else ni
	SIMD cond;
	cond = LessThan(t0, zero);
	n0 = Or(And(cond, zero), AndNot(cond, n0));
	cond = LessThan(t1, zero);
	n1 = Or(And(cond, zero), AndNot(cond, n1));
	cond = LessThan(t2, zero);
	n2 = Or(And(cond, zero), AndNot(cond, n2));
	cond = LessThan(t3, zero);
	n3 = Or(And(cond, zero), AndNot(cond, n3));


	return  Mul(thirtytwo, Add(n0, Add(n1, Add(n2, n3))));
}

inline float simplex3d(float x, float y, float z)
{
	return 1.0f;
}

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


inline SIMD gradSIMD3d(SIMDi *hash, SIMD *x, SIMD *y, SIMD *z) {

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


inline SIMD perlinSIMD3d(SIMD* x, SIMD* y, SIMD* z)
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
inline float perlin3d(float x, float y, float z)
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



