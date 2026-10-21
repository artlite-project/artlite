#pragma once

// The NPU shim of this build's backend (ARTLITE_SHIM_XRT): the amdxdna driver's ioctls (Linux) or XRT's API (Windows;
// Linux where XRT is installed). Both define shim_xdna::{device, bo, hw_ctx}.
#ifdef ARTLITE_SHIM_XRT
#include "shim_xrt/bo.h"
#include "shim_xrt/device.h"
#include "shim_xrt/hwctx.h"
#else
#include "shim_xdna/bo.h"
#include "shim_xdna/device.h"
#include "shim_xdna/hwctx.h"
#endif
#include "shim_xdna/ert.h"
