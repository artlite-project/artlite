"""qwen35's X metadata (``decode.protocol.XMETA``): the flags of its cores' objects and the words past the protocol's
header that the GDN and attention cores read, as kinds of object (their state words are the cores' own: ``gdn``, ``attn``).

- the gdn_in phase's metadata object (``GDN_IN``: a phase object, its words extend the phase's) tells the GDN core the
  step's replay count (``M_NREP``), the conv shift (``M_CSHIFT``: the accepted tokens of the previous step), the commit
  flag (``M_COMMIT``) and, in the chunkwise cores, the k / v histories' base row (``M_KSHIFT``); a design of 8 slots
  also the helper's requests per step (``M_HPAN``);
- an F_ATT object (``ATT``) the attention core's: the sub-step's first position; split: the KV head's chain divides the
  cache's groups (1); the core's first group, the end of its attention groups, the first group past them (a chain's first
  core's: the sub-step's group, for its records only); its first slot's row of angles; the tail objects; (one pass) the
  core's A objects through its held ones, the rows' masks of the step's first two records and the slots' K lanes;
- a step whose last slots branch off an earlier one: both objects' word ``M_BRANCH`` (its parent slot + 1 | its slots
  << 8, or 0), a kind of its own that the two extend.
"""

from artlite.decode.protocol import M_MODEL, PHASE, XMETA

FLAGS = XMETA.kind("qwen35 flags")
F_GDN = FLAGS.flag("F_GDN", 8)  # M_FLAGS of the first GDN object of a step (the other cores skip it and M_SKIP objects after it)
F_GIN = FLAGS.flag("F_GIN", 9)  # (8 slots) M_FLAGS of the gdn_in phase's metadata object: the GDN core takes the step's words from it
F_ATT = FLAGS.flag("F_ATT", 10)  # M_FLAGS of an attention core's object

BRANCH = XMETA.kind("qwen35 branch")
M_BRANCH = BRANCH.word("M_BRANCH", at=37)

GDN_IN = XMETA.kind("qwen35 gdn_in", extends=(PHASE, BRANCH))
M_NREP, M_CSHIFT, M_COMMIT, M_KSHIFT = GDN_IN.words("M_NREP M_CSHIFT M_COMMIT M_KSHIFT", at=M_MODEL)
M_HPAN = GDN_IN.word("M_HPAN", at=35)

ATT = XMETA.kind("qwen35 attention", extends=(BRANCH,))
M_POS, M_SPLIT, M_G0, M_GATT, M_GJ, M_SA, M_NTAIL = ATT.words("M_POS M_SPLIT M_G0 M_GATT M_GJ M_SA M_NTAIL", at=M_MODEL + 3)
M_ANA = ATT.word("M_ANA", at=36)  # (one pass) the core's A objects through its held ones
M_AMASK = ATT.block("M_AMASK", 4, at=38)  # (one pass) the rows' masks of the step's first two records (``attn``'s A_MK)
M_ALANE = ATT.word("M_ALANE", at=42)  # (one pass) the slots' K lanes (``attn``'s A_LN)
