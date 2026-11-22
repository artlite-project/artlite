#pragma once

// A ParoQuant rotation's tan form (artlite.paro.Rotation._rounds): its Givens rounds as u <- u + T u[partner] on storage
// positions, the scales' paths through them.

#include <cstdint>

namespace artlite {

/// The rounds of groups of ``group`` channels (``groups`` of them; ``rounds`` rounds; per round and group group / 2
/// pairs): ``pairs`` [rounds][groups][group] the pairs' channels (group-local, pair k = entries 2 k, 2 k + 1), ``cs`` and
/// ``sn`` [rounds][groups][group / 2] their angles' cosines and sines, ``d`` [groups][group] the initial scales (updated
/// in place to the final ones per channel). Out: ``partner`` [rounds][groups][group], ``coef`` [rounds][groups][group]
/// per storage position; ``pos`` [groups][group] the final storage position of each channel, ``origin``
/// [groups][group] per channel the channel whose initial scale its scale carries. The arithmetic is _rounds' numpy
/// expressions term by term (fp64, no contractions): the same bits.
void paro_rounds(int rounds, int groups, int group, const int64_t* pairs, const double* cs, const double* sn,
                 double* d, uint8_t* partner, double* coef, int64_t* pos, int64_t* origin);

}  // namespace artlite
