// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2022-2026, Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

#ifndef SHIM_ERR_H
#define SHIM_ERR_H

#include <format>
#include <string>
#include <system_error>

namespace shim_xdna {

// Throws std::system_error (a std::runtime_error): what() ends with strerror(err).
template <typename... Args>
[[noreturn]] void shim_err(int err, std::format_string<Args...> fmt, Args&&... args) {
  throw std::system_error(err, std::generic_category(),
                          "shim_xdna: " + std::format(fmt, std::forward<Args>(args)...));
}

}  // namespace shim_xdna

#endif  // SHIM_ERR_H
