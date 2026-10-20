#pragma once

// The NPU shim of this platform: the amdxdna driver's ioctls on Linux, XRT on Windows. Both define
// shim_xdna::{device, bo, hw_ctx}.
#ifdef _WIN32
#include "shim_xrt/bo.h"
#include "shim_xrt/device.h"
#include "shim_xrt/hwctx.h"
#else
#include "shim_xdna/bo.h"
#include "shim_xdna/device.h"
#include "shim_xdna/hwctx.h"
#endif
#include "shim_xdna/ert.h"
