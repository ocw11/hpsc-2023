#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>



int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N], a[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    a[i] = i;
  }
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 avec = _mm256_load_ps(a);
  __m256 bvec = _mm256_setzero_ps();
  for(int i=0; i<N; i++) {
    __m256 ivec = _mm256_set1_ps(i);
    __m256 mask = _mm256_cmp_ps(avec, ivec, _CMP_NEQ_UQ);

    __m256 xivec = _mm256_set1_ps(x[i]);
    __m256 yivec = _mm256_set1_ps(y[i]);

    __m256 rxvec = _mm256_sub_ps(xivec, xvec);
    __m256 ryvec = _mm256_sub_ps(yivec, yvec);
    __m256 rr = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec)));
    __m256 fxvec = _mm256_mul_ps(rxvec, _mm256_mul_ps(mvec, _mm256_mul_ps(rr, _mm256_mul_ps(rr, rr)))); 
    __m256 fyvec = _mm256_mul_ps(ryvec, _mm256_mul_ps(mvec, _mm256_mul_ps(rr, _mm256_mul_ps(rr, rr))));
    fxvec = _mm256_blendv_ps(bvec, fxvec, mask);
    fyvec = _mm256_blendv_ps(bvec, fyvec, mask);
    
    __m256 fxivec = _mm256_permute2f128_ps(fxvec, fxvec, 1);
    fxivec = _mm256_add_ps(fxivec, fxvec);
    fxivec = _mm256_hadd_ps(fxivec, fxivec);
    fxivec = _mm256_hadd_ps(fxivec, fxivec);
    __m256 fyivec = _mm256_permute2f128_ps(fyvec, fyvec, 1);
    fyivec = _mm256_add_ps(fyivec, fyvec);
    fyivec = _mm256_hadd_ps(fyivec, fyivec);
    fyivec = _mm256_hadd_ps(fyivec, fyivec);
    
    _mm256_store_ps(fx, _mm256_sub_ps(bvec, fxivec));
    _mm256_store_ps(fy, _mm256_sub_ps(bvec, fyivec));
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
