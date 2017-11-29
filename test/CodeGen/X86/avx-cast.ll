; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+avx  | FileCheck %s --check-prefix=AVX --check-prefix=AVX1
; RUN: llc < %s -mtriple=x86_64-apple-darwin -mattr=+avx2 | FileCheck %s --check-prefix=AVX --check-prefix=AVX2

; Prefer a blend instruction to a vinsert128 instruction because blends
; are simpler (no lane changes) and therefore will have equal or better
; performance.

define <8 x float> @castA(<4 x float> %m) nounwind uwtable readnone ssp {
; AVX-LABEL: castA:
; AVX:       ## BB#0:
; AVX-NEXT:    ## kill: %xmm0<def> %xmm0<kill> %ymm0<def>
; AVX-NEXT:    vxorps %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; AVX-NEXT:    retq
  %shuffle.i = shufflevector <4 x float> %m, <4 x float> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 4, i32 4, i32 4>
  ret <8 x float> %shuffle.i
}

define <4 x double> @castB(<2 x double> %m) nounwind uwtable readnone ssp {
; AVX-LABEL: castB:
; AVX:       ## BB#0:
; AVX-NEXT:    ## kill: %xmm0<def> %xmm0<kill> %ymm0<def>
; AVX-NEXT:    vxorpd %xmm1, %xmm1, %xmm1
; AVX-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3]
; AVX-NEXT:    retq
  %shuffle.i = shufflevector <2 x double> %m, <2 x double> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  ret <4 x double> %shuffle.i
}

; AVX2 is needed for integer types.

define <4 x i64> @castC(<2 x i64> %m) nounwind uwtable readnone ssp {
; AVX1-LABEL: castC:
; AVX1:       ## BB#0:
; AVX1-NEXT:    ## kill: %xmm0<def> %xmm0<kill> %ymm0<def>
; AVX1-NEXT:    vxorpd %xmm1, %xmm1, %xmm1
; AVX1-NEXT:    vblendpd {{.*#+}} ymm0 = ymm0[0,1],ymm1[2,3]
; AVX1-NEXT:    retq
;
; AVX2-LABEL: castC:
; AVX2:       ## BB#0:
; AVX2-NEXT:    ## kill: %xmm0<def> %xmm0<kill> %ymm0<def>
; AVX2-NEXT:    vxorps %xmm1, %xmm1, %xmm1
; AVX2-NEXT:    vblendps {{.*#+}} ymm0 = ymm0[0,1,2,3],ymm1[4,5,6,7]
; AVX2-NEXT:    retq
  %shuffle.i = shufflevector <2 x i64> %m, <2 x i64> zeroinitializer, <4 x i32> <i32 0, i32 1, i32 2, i32 2>
  ret <4 x i64> %shuffle.i
}

; The next three tests don't need any shuffling. There may or may not be a
; vzeroupper before the return, so just check for the absence of shuffles.

define <4 x float> @castD(<8 x float> %m) nounwind uwtable readnone ssp {
; AVX-LABEL: castD:
; AVX:       ## BB#0:
; AVX-NEXT:    ## kill: %xmm0<def> %xmm0<kill> %ymm0<kill>
; AVX-NEXT:    vzeroupper
; AVX-NEXT:    retq
  %shuffle.i = shufflevector <8 x float> %m, <8 x float> %m, <4 x i32> <i32 0, i32 1, i32 2, i32 3>
  ret <4 x float> %shuffle.i
}

define <2 x i64> @castE(<4 x i64> %m) nounwind uwtable readnone ssp {
; AVX-LABEL: castE:
; AVX:       ## BB#0:
; AVX-NEXT:    ## kill: %xmm0<def> %xmm0<kill> %ymm0<kill>
; AVX-NEXT:    vzeroupper
; AVX-NEXT:    retq
  %shuffle.i = shufflevector <4 x i64> %m, <4 x i64> %m, <2 x i32> <i32 0, i32 1>
  ret <2 x i64> %shuffle.i
}

define <2 x double> @castF(<4 x double> %m) nounwind uwtable readnone ssp {
; AVX-LABEL: castF:
; AVX:       ## BB#0:
; AVX-NEXT:    ## kill: %xmm0<def> %xmm0<kill> %ymm0<kill>
; AVX-NEXT:    vzeroupper
; AVX-NEXT:    retq
  %shuffle.i = shufflevector <4 x double> %m, <4 x double> %m, <2 x i32> <i32 0, i32 1>
  ret <2 x double> %shuffle.i
}

