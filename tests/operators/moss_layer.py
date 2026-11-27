"""The MOSS encoder layer's program: its home is ``models.moss_tr.chain`` (the app's 24-layer chain and its program),
re-exported here for the operator tests (``test_moss_layer.py``: the pages, the weight images, the host arithmetic
against the CPU reference; ``test_moss_layer_weights.py``: the same images packed from a checkpoint's tensors).
"""

from models.moss_tr.chain import (  # noqa: F401 (the names the tests import)
    A_PAGE_BUDGET,
    FFN,
    FFN_REAL,
    HD,
    HEADS,
    HP,
    KEYS,
    LV,
    NAME,
    PAGES,
    QSCALE,
    C,
    Chain,
    H,
    Layer,
    block,
    block_reference,
    compiled,
    front,
    layer_front,
    layer_weights,
    layer_wt,
    page_sizes,
    program_memory,
    reference,
    sequence_rows,
    valid_keys,
)
