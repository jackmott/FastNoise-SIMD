# FastNoise SIMD
Ultra fast Perlin and Simplex noise functions sped up with SSE2,SSE4, and AVX2 instructions. 
If you are interested in this you may want to refer to: https://github.com/Auburns/FastNoiseSIMD
A similarly named library inspired by this one that is more user friendly and even faster. 

FastNoise.h / cpp
-----------------
Contains SIMD constants, constant lookup tables, and SIMD intrinsic helper #defines
which allow you to switch between SSE2, SSE4, and AVX2 builds by adjusting the #defines
at the top.  The SIMD typedef allows us to abstract the __m128 and __m256 types for each
case.  It should not be too hard to adapt this to AVX512 or other instruction sets, just add
a new set of #defines for the instructions in question, and a new typedef for SIMD. Please 
feel free to add other SIMD platforms and pull request!

FastNoise3d.h / cpp
-------------------
The base Perlin and Simplex noise functions, provided in both SIMD and non SIMD form.
 

FractalNoise3d.h / cpp
----------------------
Various fractal noise variants, in SIMD and non SIMD form.These methods iterate overthe noise 
functions at different scales, providing very detailed and interesting patterns.


NoiseUtility.h / cpp
--------------------
Utility functions to grab large chunks of noise at a time. The Sphere methods will create noise
that can be texture mapped to a sphere. Methods to return 2d noise for flat textures and methods
that accept a set of coordinates and return the noise would be next up. Feel free to pull request that!

