// SPDX-License-Identifier: Apache-2.0 OR GPL-2.0-only
// Copyright (C) 2019-2022 Xilinx, Inc. All rights reserved.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.
// Modified for Artlite: see LICENSE in this directory.

// The parts of XRT's ert.h Artlite uses: the exec-buffer packet header, the command states and
// opcodes, and the health record the driver writes over a timed-out command.

#ifndef SHIM_XDNA_ERT_H
#define SHIM_XDNA_ERT_H

#include <cstdint>
#include <string_view>

namespace shim_xdna {

// Packet header word: state [3:0], extra CU masks [11:10], payload words [22:12], opcode [27:23],
// type [31:28]. The payload starts with the CU mask.
constexpr uint32_t ert_header(uint32_t state, uint32_t count, uint32_t opcode, uint32_t type) {
  return state | (count << 12) | (opcode << 23) | (type << 28);
}
constexpr uint32_t ERT_STATE_MASK = 0xf;

enum ert_cmd_state : uint32_t {
  ERT_CMD_STATE_NEW = 1,
  ERT_CMD_STATE_QUEUED = 2,
  ERT_CMD_STATE_RUNNING = 3,
  ERT_CMD_STATE_COMPLETED = 4,
  ERT_CMD_STATE_ERROR = 5,
  ERT_CMD_STATE_ABORT = 6,
  ERT_CMD_STATE_SUBMITTED = 7,
  ERT_CMD_STATE_TIMEOUT = 8,
  ERT_CMD_STATE_NORESPONSE = 9,
  ERT_CMD_STATE_SKERROR = 10,
  ERT_CMD_STATE_SKCRASHED = 11,
};

enum ert_cmd_opcode : uint32_t {
  ERT_START_CU = 0,
  ERT_START_NPU_PREEMPT_ELF = 22,
};

constexpr uint32_t ERT_CU = 3;  // command type

// Whether a command in this state is finished. Enumerated, not `>= COMPLETED`: SUBMITTED (7) is
// an in-flight state. The driver updates the header with a plain read-modify-write, so a poller
// may see 0, which counts as pending; unknown values count as finished (nobody would end them).
constexpr bool ert_cmd_state_is_terminal(uint32_t state) {
  switch (state) {
    case 0:
    case ERT_CMD_STATE_NEW:
    case ERT_CMD_STATE_QUEUED:
    case ERT_CMD_STATE_RUNNING:
    case ERT_CMD_STATE_SUBMITTED:
      return false;
    default:
      return true;
  }
}

constexpr std::string_view ert_cmd_state_name(uint32_t state) {
  switch (state) {
    case ERT_CMD_STATE_NEW: return "NEW";
    case ERT_CMD_STATE_QUEUED: return "QUEUED";
    case ERT_CMD_STATE_RUNNING: return "RUNNING";
    case ERT_CMD_STATE_COMPLETED: return "COMPLETED";
    case ERT_CMD_STATE_ERROR: return "ERROR";
    case ERT_CMD_STATE_ABORT: return "ABORT";
    case ERT_CMD_STATE_SUBMITTED: return "SUBMITTED";
    case ERT_CMD_STATE_TIMEOUT: return "TIMEOUT";
    case ERT_CMD_STATE_NORESPONSE: return "NORESPONSE";
    case ERT_CMD_STATE_SKERROR: return "SKERROR";
    case ERT_CMD_STATE_SKCRASHED: return "SKCRASHED";
    default: return "UNKNOWN";
  }
}

// After ERT_CMD_STATE_TIMEOUT the driver copies this record over the packet words after the
// header (ert_ctx_health_data_v1, AIE2 member; the rest of the packet reads 0xff). Version is
// not 1 when the firmware gave no report. ERROR and ABORT carry no data.
struct ert_ctx_health_aie2 {
  uint32_t version;  // 1
  uint32_t npu_gen;  // 0: AIE2 / AIE2P
  uint32_t txn_op_idx;  // last control-code operation the firmware ran
  uint32_t ctx_pc;
  uint32_t fatal_error_type;
  uint32_t fatal_error_exception_type;
  uint32_t fatal_error_exception_pc;
  uint32_t fatal_error_app_module;
};

}  // namespace shim_xdna

#endif  // SHIM_XDNA_ERT_H
