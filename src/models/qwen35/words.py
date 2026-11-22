"""qwen35's X metadata (``decode.protocol.XMETA``): the flags of its GDN cores' objects and the words past the protocol's
header that they read, as kinds of object (their state words are the cores' own: ``gdn``; the attention core's objects
and words: ``decode.attn256``).

- the gdn_in phase's metadata object (``GDN_IN``: a phase object, its words extend the phase's) tells the GDN core the
  step's replay count (``M_NREP``), the conv shift (``M_CSHIFT``: the accepted tokens of the previous step), the commit
  flag (``M_COMMIT``) and, in the chunkwise cores, the k / v histories' base row (``M_KSHIFT``); a design of 8 slots
  also the helper's requests per step (``M_HPAN``); a PARO checkpoint's two phases each their part of the message
  (``M_MPART``);
- a step whose last slots branch off an earlier one: its word ``M_BRANCH`` (its parent slot + 1 | its slots << 8, or
  0), the attention's kind ``decode.attn256.BRANCH``, which the F_ATT objects extend too.
"""

from artlite.decode.attn256 import BRANCH
from artlite.decode.protocol import M_MODEL, PHASE, XMETA

FLAGS = XMETA.kind("qwen35 flags")
F_GDN = FLAGS.flag("F_GDN", 8)  # M_FLAGS of the first GDN object of a step (the other cores skip it and M_SKIP objects after it)
F_GIN = FLAGS.flag("F_GIN", 9)  # (8 slots) M_FLAGS of the gdn_in phase's metadata object: the GDN core takes the step's words from it

GDN_IN = XMETA.kind("qwen35 gdn_in", extends=(PHASE, BRANCH))
M_NREP, M_CSHIFT, M_COMMIT, M_KSHIFT = GDN_IN.words("M_NREP M_CSHIFT M_COMMIT M_KSHIFT", at=M_MODEL)
M_HPAN = GDN_IN.word("M_HPAN", at=35)
# (a PARO checkpoint: gdn_in in two phases, each on its own rotated input) the phase's part of the step's message: its
# objects (all slots') | MP_LAST (the last part: the step starts after it) | its first object per slot << MP_J0 (the
# earlier parts' objects); 0: the whole message. Every part's metadata carries the step's words.
M_MPART = GDN_IN.word("M_MPART")
MP_LAST, MP_J0 = 1 << 16, 17
