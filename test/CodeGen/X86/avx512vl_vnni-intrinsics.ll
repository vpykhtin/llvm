; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
; RUN: llc < %s -mtriple=i686-unknown-unknown -mattr=+avx512vnni,+avx512vl --show-mc-encoding | FileCheck %s --check-prefixes=CHECK,X86
; RUN: llc < %s -mtriple=x86_64-unknown-unknown -mattr=+avx512vnni,+avx512vl --show-mc-encoding | FileCheck %s --check-prefixes=CHECK,X64

declare <8 x i32> @llvm.x86.avx512.mask.vpdpbusd.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)
declare <8 x i32> @llvm.x86.avx512.maskz.vpdpbusd.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vpdpbusd_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32>* %x2p, <8 x i32> %x4, i8 %x3) {
; X86-LABEL: test_int_x86_avx512_mask_vpdpbusd_256:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax # encoding: [0x0f,0xb6,0x44,0x24,0x08]
; X86-NEXT:    kmovw %eax, %k1 # encoding: [0xc5,0xf8,0x92,0xc8]
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax # encoding: [0x8b,0x44,0x24,0x04]
; X86-NEXT:    vmovaps %ymm0, %ymm3 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xd8]
; X86-NEXT:    vpdpbusd (%eax), %ymm1, %ymm3 {%k1} # encoding: [0x62,0xf2,0x75,0x29,0x50,0x18]
; X86-NEXT:    vmovaps %ymm0, %ymm4 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xe0]
; X86-NEXT:    vpdpbusd %ymm2, %ymm1, %ymm4 {%k1} {z} # encoding: [0x62,0xf2,0x75,0xa9,0x50,0xe2]
; X86-NEXT:    vpdpbusd %ymm2, %ymm1, %ymm0 # encoding: [0x62,0xf2,0x75,0x28,0x50,0xc2]
; X86-NEXT:    vpaddd %ymm4, %ymm0, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xfd,0xfe,0xc4]
; X86-NEXT:    vpaddd %ymm0, %ymm3, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xe5,0xfe,0xc0]
; X86-NEXT:    retl # encoding: [0xc3]
;
; X64-LABEL: test_int_x86_avx512_mask_vpdpbusd_256:
; X64:       # %bb.0:
; X64-NEXT:    kmovw %esi, %k1 # encoding: [0xc5,0xf8,0x92,0xce]
; X64-NEXT:    vmovaps %ymm0, %ymm3 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xd8]
; X64-NEXT:    vpdpbusd (%rdi), %ymm1, %ymm3 {%k1} # encoding: [0x62,0xf2,0x75,0x29,0x50,0x1f]
; X64-NEXT:    vmovaps %ymm0, %ymm4 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xe0]
; X64-NEXT:    vpdpbusd %ymm2, %ymm1, %ymm4 # encoding: [0x62,0xf2,0x75,0x28,0x50,0xe2]
; X64-NEXT:    vpdpbusd %ymm2, %ymm1, %ymm0 {%k1} {z} # encoding: [0x62,0xf2,0x75,0xa9,0x50,0xc2]
; X64-NEXT:    vpaddd %ymm0, %ymm4, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xdd,0xfe,0xc0]
; X64-NEXT:    vpaddd %ymm0, %ymm3, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xe5,0xfe,0xc0]
; X64-NEXT:    retq # encoding: [0xc3]
  %x2 = load <8 x i32>, <8 x i32>* %x2p
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpdpbusd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.vpdpbusd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.maskz.vpdpbusd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8  %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.vpdpbusd.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)
declare <4 x i32> @llvm.x86.avx512.maskz.vpdpbusd.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vpdpbusd_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32>* %x2p, <4 x i32> %x4, i8 %x3) {
; X86-LABEL: test_int_x86_avx512_mask_vpdpbusd_128:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax # encoding: [0x0f,0xb6,0x44,0x24,0x08]
; X86-NEXT:    kmovw %eax, %k1 # encoding: [0xc5,0xf8,0x92,0xc8]
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax # encoding: [0x8b,0x44,0x24,0x04]
; X86-NEXT:    vmovaps %xmm0, %xmm3 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xd8]
; X86-NEXT:    vpdpbusd (%eax), %xmm1, %xmm3 {%k1} # encoding: [0x62,0xf2,0x75,0x09,0x50,0x18]
; X86-NEXT:    vmovaps %xmm0, %xmm4 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xe0]
; X86-NEXT:    vpdpbusd %xmm2, %xmm1, %xmm4 {%k1} {z} # encoding: [0x62,0xf2,0x75,0x89,0x50,0xe2]
; X86-NEXT:    vpdpbusd %xmm2, %xmm1, %xmm0 # encoding: [0x62,0xf2,0x75,0x08,0x50,0xc2]
; X86-NEXT:    vpaddd %xmm4, %xmm0, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xf9,0xfe,0xc4]
; X86-NEXT:    vpaddd %xmm0, %xmm3, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xe1,0xfe,0xc0]
; X86-NEXT:    retl # encoding: [0xc3]
;
; X64-LABEL: test_int_x86_avx512_mask_vpdpbusd_128:
; X64:       # %bb.0:
; X64-NEXT:    kmovw %esi, %k1 # encoding: [0xc5,0xf8,0x92,0xce]
; X64-NEXT:    vmovaps %xmm0, %xmm3 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xd8]
; X64-NEXT:    vpdpbusd (%rdi), %xmm1, %xmm3 {%k1} # encoding: [0x62,0xf2,0x75,0x09,0x50,0x1f]
; X64-NEXT:    vmovaps %xmm0, %xmm4 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xe0]
; X64-NEXT:    vpdpbusd %xmm2, %xmm1, %xmm4 # encoding: [0x62,0xf2,0x75,0x08,0x50,0xe2]
; X64-NEXT:    vpdpbusd %xmm2, %xmm1, %xmm0 {%k1} {z} # encoding: [0x62,0xf2,0x75,0x89,0x50,0xc2]
; X64-NEXT:    vpaddd %xmm0, %xmm4, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xd9,0xfe,0xc0]
; X64-NEXT:    vpaddd %xmm0, %xmm3, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xe1,0xfe,0xc0]
; X64-NEXT:    retq # encoding: [0xc3]
  %x2 = load <4 x i32>, <4 x i32>* %x2p
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpdpbusd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.vpdpbusd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.maskz.vpdpbusd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8  %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.vpdpbusds.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)
declare <8 x i32> @llvm.x86.avx512.maskz.vpdpbusds.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vpdpbusds_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32>* %x2p, <8 x i32> %x4, i8 %x3) {
; X86-LABEL: test_int_x86_avx512_mask_vpdpbusds_256:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax # encoding: [0x0f,0xb6,0x44,0x24,0x08]
; X86-NEXT:    kmovw %eax, %k1 # encoding: [0xc5,0xf8,0x92,0xc8]
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax # encoding: [0x8b,0x44,0x24,0x04]
; X86-NEXT:    vmovaps %ymm0, %ymm3 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xd8]
; X86-NEXT:    vpdpbusds (%eax), %ymm1, %ymm3 {%k1} # encoding: [0x62,0xf2,0x75,0x29,0x51,0x18]
; X86-NEXT:    vmovaps %ymm0, %ymm4 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xe0]
; X86-NEXT:    vpdpbusds %ymm2, %ymm1, %ymm4 {%k1} {z} # encoding: [0x62,0xf2,0x75,0xa9,0x51,0xe2]
; X86-NEXT:    vpdpbusds %ymm2, %ymm1, %ymm0 # encoding: [0x62,0xf2,0x75,0x28,0x51,0xc2]
; X86-NEXT:    vpaddd %ymm4, %ymm0, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xfd,0xfe,0xc4]
; X86-NEXT:    vpaddd %ymm0, %ymm3, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xe5,0xfe,0xc0]
; X86-NEXT:    retl # encoding: [0xc3]
;
; X64-LABEL: test_int_x86_avx512_mask_vpdpbusds_256:
; X64:       # %bb.0:
; X64-NEXT:    kmovw %esi, %k1 # encoding: [0xc5,0xf8,0x92,0xce]
; X64-NEXT:    vmovaps %ymm0, %ymm3 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xd8]
; X64-NEXT:    vpdpbusds (%rdi), %ymm1, %ymm3 {%k1} # encoding: [0x62,0xf2,0x75,0x29,0x51,0x1f]
; X64-NEXT:    vmovaps %ymm0, %ymm4 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xe0]
; X64-NEXT:    vpdpbusds %ymm2, %ymm1, %ymm4 # encoding: [0x62,0xf2,0x75,0x28,0x51,0xe2]
; X64-NEXT:    vpdpbusds %ymm2, %ymm1, %ymm0 {%k1} {z} # encoding: [0x62,0xf2,0x75,0xa9,0x51,0xc2]
; X64-NEXT:    vpaddd %ymm0, %ymm4, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xdd,0xfe,0xc0]
; X64-NEXT:    vpaddd %ymm0, %ymm3, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xe5,0xfe,0xc0]
; X64-NEXT:    retq # encoding: [0xc3]
  %x2 = load <8 x i32>, <8 x i32>* %x2p
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpdpbusds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.vpdpbusds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.maskz.vpdpbusds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8  %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.vpdpbusds.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)
declare <4 x i32> @llvm.x86.avx512.maskz.vpdpbusds.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vpdpbusds_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32>* %x2p, <4 x i32> %x4, i8 %x3) {
; X86-LABEL: test_int_x86_avx512_mask_vpdpbusds_128:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax # encoding: [0x0f,0xb6,0x44,0x24,0x08]
; X86-NEXT:    kmovw %eax, %k1 # encoding: [0xc5,0xf8,0x92,0xc8]
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax # encoding: [0x8b,0x44,0x24,0x04]
; X86-NEXT:    vmovaps %xmm0, %xmm3 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xd8]
; X86-NEXT:    vpdpbusds (%eax), %xmm1, %xmm3 {%k1} # encoding: [0x62,0xf2,0x75,0x09,0x51,0x18]
; X86-NEXT:    vmovaps %xmm0, %xmm4 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xe0]
; X86-NEXT:    vpdpbusds %xmm2, %xmm1, %xmm4 {%k1} {z} # encoding: [0x62,0xf2,0x75,0x89,0x51,0xe2]
; X86-NEXT:    vpdpbusds %xmm2, %xmm1, %xmm0 # encoding: [0x62,0xf2,0x75,0x08,0x51,0xc2]
; X86-NEXT:    vpaddd %xmm4, %xmm0, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xf9,0xfe,0xc4]
; X86-NEXT:    vpaddd %xmm0, %xmm3, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xe1,0xfe,0xc0]
; X86-NEXT:    retl # encoding: [0xc3]
;
; X64-LABEL: test_int_x86_avx512_mask_vpdpbusds_128:
; X64:       # %bb.0:
; X64-NEXT:    kmovw %esi, %k1 # encoding: [0xc5,0xf8,0x92,0xce]
; X64-NEXT:    vmovaps %xmm0, %xmm3 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xd8]
; X64-NEXT:    vpdpbusds (%rdi), %xmm1, %xmm3 {%k1} # encoding: [0x62,0xf2,0x75,0x09,0x51,0x1f]
; X64-NEXT:    vmovaps %xmm0, %xmm4 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xe0]
; X64-NEXT:    vpdpbusds %xmm2, %xmm1, %xmm4 # encoding: [0x62,0xf2,0x75,0x08,0x51,0xe2]
; X64-NEXT:    vpdpbusds %xmm2, %xmm1, %xmm0 {%k1} {z} # encoding: [0x62,0xf2,0x75,0x89,0x51,0xc2]
; X64-NEXT:    vpaddd %xmm0, %xmm4, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xd9,0xfe,0xc0]
; X64-NEXT:    vpaddd %xmm0, %xmm3, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xe1,0xfe,0xc0]
; X64-NEXT:    retq # encoding: [0xc3]
  %x2 = load <4 x i32>, <4 x i32>* %x2p
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpdpbusds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.vpdpbusds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.maskz.vpdpbusds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8  %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

declare <8 x i32> @llvm.x86.avx512.mask.vpdpwssd.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)
declare <8 x i32> @llvm.x86.avx512.maskz.vpdpwssd.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vpdpwssd_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32>* %x2p, <8 x i32> %x4, i8 %x3) {
; X86-LABEL: test_int_x86_avx512_mask_vpdpwssd_256:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax # encoding: [0x0f,0xb6,0x44,0x24,0x08]
; X86-NEXT:    kmovw %eax, %k1 # encoding: [0xc5,0xf8,0x92,0xc8]
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax # encoding: [0x8b,0x44,0x24,0x04]
; X86-NEXT:    vmovaps %ymm0, %ymm3 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xd8]
; X86-NEXT:    vpdpwssd (%eax), %ymm1, %ymm3 {%k1} # encoding: [0x62,0xf2,0x75,0x29,0x52,0x18]
; X86-NEXT:    vmovaps %ymm0, %ymm4 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xe0]
; X86-NEXT:    vpdpwssd %ymm2, %ymm1, %ymm4 {%k1} {z} # encoding: [0x62,0xf2,0x75,0xa9,0x52,0xe2]
; X86-NEXT:    vpdpwssd %ymm2, %ymm1, %ymm0 # encoding: [0x62,0xf2,0x75,0x28,0x52,0xc2]
; X86-NEXT:    vpaddd %ymm4, %ymm0, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xfd,0xfe,0xc4]
; X86-NEXT:    vpaddd %ymm0, %ymm3, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xe5,0xfe,0xc0]
; X86-NEXT:    retl # encoding: [0xc3]
;
; X64-LABEL: test_int_x86_avx512_mask_vpdpwssd_256:
; X64:       # %bb.0:
; X64-NEXT:    kmovw %esi, %k1 # encoding: [0xc5,0xf8,0x92,0xce]
; X64-NEXT:    vmovaps %ymm0, %ymm3 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xd8]
; X64-NEXT:    vpdpwssd (%rdi), %ymm1, %ymm3 {%k1} # encoding: [0x62,0xf2,0x75,0x29,0x52,0x1f]
; X64-NEXT:    vmovaps %ymm0, %ymm4 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xe0]
; X64-NEXT:    vpdpwssd %ymm2, %ymm1, %ymm4 # encoding: [0x62,0xf2,0x75,0x28,0x52,0xe2]
; X64-NEXT:    vpdpwssd %ymm2, %ymm1, %ymm0 {%k1} {z} # encoding: [0x62,0xf2,0x75,0xa9,0x52,0xc2]
; X64-NEXT:    vpaddd %ymm0, %ymm4, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xdd,0xfe,0xc0]
; X64-NEXT:    vpaddd %ymm0, %ymm3, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xe5,0xfe,0xc0]
; X64-NEXT:    retq # encoding: [0xc3]
  %x2 = load <8 x i32>, <8 x i32>* %x2p
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpdpwssd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.vpdpwssd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.maskz.vpdpwssd.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8  %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.vpdpwssd.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)
declare <4 x i32> @llvm.x86.avx512.maskz.vpdpwssd.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vpdpwssd_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32>* %x2p, <4 x i32> %x4, i8 %x3) {
; X86-LABEL: test_int_x86_avx512_mask_vpdpwssd_128:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax # encoding: [0x0f,0xb6,0x44,0x24,0x08]
; X86-NEXT:    kmovw %eax, %k1 # encoding: [0xc5,0xf8,0x92,0xc8]
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax # encoding: [0x8b,0x44,0x24,0x04]
; X86-NEXT:    vmovaps %xmm0, %xmm3 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xd8]
; X86-NEXT:    vpdpwssd (%eax), %xmm1, %xmm3 {%k1} # encoding: [0x62,0xf2,0x75,0x09,0x52,0x18]
; X86-NEXT:    vmovaps %xmm0, %xmm4 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xe0]
; X86-NEXT:    vpdpwssd %xmm2, %xmm1, %xmm4 {%k1} {z} # encoding: [0x62,0xf2,0x75,0x89,0x52,0xe2]
; X86-NEXT:    vpdpwssd %xmm2, %xmm1, %xmm0 # encoding: [0x62,0xf2,0x75,0x08,0x52,0xc2]
; X86-NEXT:    vpaddd %xmm4, %xmm0, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xf9,0xfe,0xc4]
; X86-NEXT:    vpaddd %xmm0, %xmm3, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xe1,0xfe,0xc0]
; X86-NEXT:    retl # encoding: [0xc3]
;
; X64-LABEL: test_int_x86_avx512_mask_vpdpwssd_128:
; X64:       # %bb.0:
; X64-NEXT:    kmovw %esi, %k1 # encoding: [0xc5,0xf8,0x92,0xce]
; X64-NEXT:    vmovaps %xmm0, %xmm3 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xd8]
; X64-NEXT:    vpdpwssd (%rdi), %xmm1, %xmm3 {%k1} # encoding: [0x62,0xf2,0x75,0x09,0x52,0x1f]
; X64-NEXT:    vmovaps %xmm0, %xmm4 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xe0]
; X64-NEXT:    vpdpwssd %xmm2, %xmm1, %xmm4 # encoding: [0x62,0xf2,0x75,0x08,0x52,0xe2]
; X64-NEXT:    vpdpwssd %xmm2, %xmm1, %xmm0 {%k1} {z} # encoding: [0x62,0xf2,0x75,0x89,0x52,0xc2]
; X64-NEXT:    vpaddd %xmm0, %xmm4, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xd9,0xfe,0xc0]
; X64-NEXT:    vpaddd %xmm0, %xmm3, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xe1,0xfe,0xc0]
; X64-NEXT:    retq # encoding: [0xc3]
  %x2 = load <4 x i32>, <4 x i32>* %x2p
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpdpwssd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.vpdpwssd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.maskz.vpdpwssd.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8  %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}


declare <8 x i32> @llvm.x86.avx512.mask.vpdpwssds.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)
declare <8 x i32> @llvm.x86.avx512.maskz.vpdpwssds.256(<8 x i32>, <8 x i32>, <8 x i32>, i8)

define <8 x i32>@test_int_x86_avx512_mask_vpdpwssds_256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32>* %x2p, <8 x i32> %x4, i8 %x3) {
; X86-LABEL: test_int_x86_avx512_mask_vpdpwssds_256:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax # encoding: [0x0f,0xb6,0x44,0x24,0x08]
; X86-NEXT:    kmovw %eax, %k1 # encoding: [0xc5,0xf8,0x92,0xc8]
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax # encoding: [0x8b,0x44,0x24,0x04]
; X86-NEXT:    vmovaps %ymm0, %ymm3 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xd8]
; X86-NEXT:    vpdpwssds (%eax), %ymm1, %ymm3 {%k1} # encoding: [0x62,0xf2,0x75,0x29,0x53,0x18]
; X86-NEXT:    vmovaps %ymm0, %ymm4 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xe0]
; X86-NEXT:    vpdpwssds %ymm2, %ymm1, %ymm4 {%k1} {z} # encoding: [0x62,0xf2,0x75,0xa9,0x53,0xe2]
; X86-NEXT:    vpdpwssds %ymm2, %ymm1, %ymm0 # encoding: [0x62,0xf2,0x75,0x28,0x53,0xc2]
; X86-NEXT:    vpaddd %ymm4, %ymm0, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xfd,0xfe,0xc4]
; X86-NEXT:    vpaddd %ymm0, %ymm3, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xe5,0xfe,0xc0]
; X86-NEXT:    retl # encoding: [0xc3]
;
; X64-LABEL: test_int_x86_avx512_mask_vpdpwssds_256:
; X64:       # %bb.0:
; X64-NEXT:    kmovw %esi, %k1 # encoding: [0xc5,0xf8,0x92,0xce]
; X64-NEXT:    vmovaps %ymm0, %ymm3 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xd8]
; X64-NEXT:    vpdpwssds (%rdi), %ymm1, %ymm3 {%k1} # encoding: [0x62,0xf2,0x75,0x29,0x53,0x1f]
; X64-NEXT:    vmovaps %ymm0, %ymm4 # EVEX TO VEX Compression encoding: [0xc5,0xfc,0x28,0xe0]
; X64-NEXT:    vpdpwssds %ymm2, %ymm1, %ymm4 # encoding: [0x62,0xf2,0x75,0x28,0x53,0xe2]
; X64-NEXT:    vpdpwssds %ymm2, %ymm1, %ymm0 {%k1} {z} # encoding: [0x62,0xf2,0x75,0xa9,0x53,0xc2]
; X64-NEXT:    vpaddd %ymm0, %ymm4, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xdd,0xfe,0xc0]
; X64-NEXT:    vpaddd %ymm0, %ymm3, %ymm0 # EVEX TO VEX Compression encoding: [0xc5,0xe5,0xfe,0xc0]
; X64-NEXT:    retq # encoding: [0xc3]
  %x2 = load <8 x i32>, <8 x i32>* %x2p
  %res = call <8 x i32> @llvm.x86.avx512.mask.vpdpwssds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x2, i8 %x3)
  %res1 = call <8 x i32> @llvm.x86.avx512.mask.vpdpwssds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8 -1)
  %res2 = call <8 x i32> @llvm.x86.avx512.maskz.vpdpwssds.256(<8 x i32> %x0, <8 x i32> %x1, <8 x i32> %x4, i8  %x3)
  %res3 = add <8 x i32> %res, %res1
  %res4 = add <8 x i32> %res2, %res3
  ret <8 x i32> %res4
}

declare <4 x i32> @llvm.x86.avx512.mask.vpdpwssds.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)
declare <4 x i32> @llvm.x86.avx512.maskz.vpdpwssds.128(<4 x i32>, <4 x i32>, <4 x i32>, i8)

define <4 x i32>@test_int_x86_avx512_mask_vpdpwssds_128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32>* %x2p, <4 x i32> %x4, i8 %x3) {
; X86-LABEL: test_int_x86_avx512_mask_vpdpwssds_128:
; X86:       # %bb.0:
; X86-NEXT:    movzbl {{[0-9]+}}(%esp), %eax # encoding: [0x0f,0xb6,0x44,0x24,0x08]
; X86-NEXT:    kmovw %eax, %k1 # encoding: [0xc5,0xf8,0x92,0xc8]
; X86-NEXT:    movl {{[0-9]+}}(%esp), %eax # encoding: [0x8b,0x44,0x24,0x04]
; X86-NEXT:    vmovaps %xmm0, %xmm3 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xd8]
; X86-NEXT:    vpdpwssds (%eax), %xmm1, %xmm3 {%k1} # encoding: [0x62,0xf2,0x75,0x09,0x53,0x18]
; X86-NEXT:    vmovaps %xmm0, %xmm4 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xe0]
; X86-NEXT:    vpdpwssds %xmm2, %xmm1, %xmm4 {%k1} {z} # encoding: [0x62,0xf2,0x75,0x89,0x53,0xe2]
; X86-NEXT:    vpdpwssds %xmm2, %xmm1, %xmm0 # encoding: [0x62,0xf2,0x75,0x08,0x53,0xc2]
; X86-NEXT:    vpaddd %xmm4, %xmm0, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xf9,0xfe,0xc4]
; X86-NEXT:    vpaddd %xmm0, %xmm3, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xe1,0xfe,0xc0]
; X86-NEXT:    retl # encoding: [0xc3]
;
; X64-LABEL: test_int_x86_avx512_mask_vpdpwssds_128:
; X64:       # %bb.0:
; X64-NEXT:    kmovw %esi, %k1 # encoding: [0xc5,0xf8,0x92,0xce]
; X64-NEXT:    vmovaps %xmm0, %xmm3 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xd8]
; X64-NEXT:    vpdpwssds (%rdi), %xmm1, %xmm3 {%k1} # encoding: [0x62,0xf2,0x75,0x09,0x53,0x1f]
; X64-NEXT:    vmovaps %xmm0, %xmm4 # EVEX TO VEX Compression encoding: [0xc5,0xf8,0x28,0xe0]
; X64-NEXT:    vpdpwssds %xmm2, %xmm1, %xmm4 # encoding: [0x62,0xf2,0x75,0x08,0x53,0xe2]
; X64-NEXT:    vpdpwssds %xmm2, %xmm1, %xmm0 {%k1} {z} # encoding: [0x62,0xf2,0x75,0x89,0x53,0xc2]
; X64-NEXT:    vpaddd %xmm0, %xmm4, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xd9,0xfe,0xc0]
; X64-NEXT:    vpaddd %xmm0, %xmm3, %xmm0 # EVEX TO VEX Compression encoding: [0xc5,0xe1,0xfe,0xc0]
; X64-NEXT:    retq # encoding: [0xc3]
  %x2 = load <4 x i32>, <4 x i32>* %x2p
  %res = call <4 x i32> @llvm.x86.avx512.mask.vpdpwssds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x2, i8 %x3)
  %res1 = call <4 x i32> @llvm.x86.avx512.mask.vpdpwssds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8 -1)
  %res2 = call <4 x i32> @llvm.x86.avx512.maskz.vpdpwssds.128(<4 x i32> %x0, <4 x i32> %x1, <4 x i32> %x4, i8  %x3)
  %res3 = add <4 x i32> %res, %res1
  %res4 = add <4 x i32> %res2, %res3
  ret <4 x i32> %res4
}

