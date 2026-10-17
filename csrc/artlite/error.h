#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>

namespace artlite {

struct Error : std::runtime_error {
  using std::runtime_error::runtime_error;
};

struct ExecutableError : Error {
  using Error::Error;
};

/// The firmware's report on a command that timed out (ERT_CMD_STATE_TIMEOUT).
struct CommandHealth {
  /// Index of the last control-code operation the firmware ran: for a host-built
  /// transaction, the operation index of artlite.txn.decode(words).
  uint32_t txn_op_idx;
  uint32_t ctx_pc;
  uint32_t fatal_error_type;
  uint32_t fatal_error_exception_type;
  uint32_t fatal_error_exception_pc;
  uint32_t fatal_error_app_module;
};

struct DispatchError : Error {
  using Error::Error;
  DispatchError(const std::string& what, uint32_t state, std::optional<CommandHealth> health)
      : Error(what), state(state), health(health) {}

  /// ert_cmd_state of the failed command (TIMEOUT 8 = in flight when the driver detected a
  /// stall, ABORT 6 = cancelled by a stall recovery, ERROR 5), or its pending state when a
  /// wait timed out; 0 when no command state applies.
  uint32_t state = 0;
  std::optional<CommandHealth> health;
};

}  // namespace artlite
