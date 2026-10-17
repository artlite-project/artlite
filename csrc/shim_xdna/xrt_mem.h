// SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
// Copyright (C) 2019-2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

#ifndef _SHIM_MEM_H_
#define _SHIM_MEM_H_

#ifdef _WIN32
#pragma warning(push)
#pragma warning(disable : 4201)
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

#ifdef __cplusplus
#include <cstdint>
extern "C" {
#else
#if defined(__KERNEL__)
#include <linux/types.h>
#else
#include <stdint.h>
#endif
#endif

/**
 * Encoding of flags passed to xcl buffer allocation APIs
 */
struct shim_xcl_bo_flags {
  union {
    uint64_t all;  // [63-0]

    struct {
      uint32_t flags;      // [31-0]
      uint32_t extension;  // [63-32]
    };

    struct {
      uint16_t bank;    // [15-0]
      uint8_t slot;     // [23-16]
      uint8_t boflags;  // [31-24]

      // extension
      uint32_t access : 2;   // [33-32]
      uint32_t dir : 2;      // [35-34]
      uint32_t use : 1;      // [36]
      uint32_t unused : 27;  // [63-35]
    };
  };
};

/**
 * XCL BO Flags bits layout
 *
 * bits  0 ~ 15: DDR BANK index
 * bits 24 ~ 31: BO flags
 */
#define XRT_BO_FLAGS_MEMIDX_MASK (0xFFFFFFUL)
#define XCL_BO_FLAGS_NONE (0)
#define XCL_BO_FLAGS_CACHEABLE (1U << 24)
#define XCL_BO_FLAGS_KERNBUF (1U << 25)
#define XCL_BO_FLAGS_SGL (1U << 26)
#define XCL_BO_FLAGS_SVM (1U << 27)
#define XCL_BO_FLAGS_DEV_ONLY (1U << 28)
#define XCL_BO_FLAGS_HOST_ONLY (1U << 29)
#define XCL_BO_FLAGS_P2P (1U << 30)
#define XCL_BO_FLAGS_EXECBUF (1U << 31)

/**
 * Shim level BO Flags for extension
 */
#define XRT_BO_ACCESS_LOCAL 0
#define XRT_BO_ACCESS_SHARED 1
#define XRT_BO_ACCESS_PROCESS 2
#define XRT_BO_ACCESS_HYBRID 3

/**
 * Shim level BO Flags for direction of data transfer
 * as seen from device.
 */
#define XRT_BO_ACCESS_READ (1U << 0)
#define XRT_BO_ACCESS_WRITE (1U << 1)
#define XRT_BO_ACCESS_READ_WRITE (XRT_BO_ACCESS_READ | XRT_BO_ACCESS_WRITE)

/**
 * Shim level BO Flags to distinguish use of BO
 *
 * The use flag is for internal use only. A debug BO
 * is supported only on some platforms to communicate
 * data from driver / firmware back to user space.
 */
#define XRT_BO_USE_NORMAL 0
#define XRT_BO_USE_DEBUG 1

/**
 * XRT Native BO flags
 *
 * These flags are simple aliases for use with XRT native BO APIs.
 */
#define XRT_BO_FLAGS_NONE XCL_BO_FLAGS_NONE
#define XRT_BO_FLAGS_CACHEABLE XCL_BO_FLAGS_CACHEABLE
#define XRT_BO_FLAGS_DEV_ONLY XCL_BO_FLAGS_DEV_ONLY
#define XRT_BO_FLAGS_HOST_ONLY XCL_BO_FLAGS_HOST_ONLY
#define XRT_BO_FLAGS_P2P XCL_BO_FLAGS_P2P
#define XRT_BO_FLAGS_SVM XCL_BO_FLAGS_SVM

#ifdef __cplusplus
}
#endif

#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

#ifdef _WIN32
#pragma warning(pop)
#endif

#endif
