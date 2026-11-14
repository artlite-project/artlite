"""The host's account of the GDN layers' lazy commit and chunkwise blocks: what the next decode step must do to the
states and conv histories the last steps left.

A step of drafts leaves the GDN states uncommitted: the next step replays the accepted tokens (``nrep``) and shifts the
conv histories by them (``cshift``). A chunkwise core (``gdn``'s "Chunkwise": 4 or 8 slots) instead writes its states at the
ends of blocks of 8 positions, double-buffered (``half``): the block of position ``gp`` starts in GSTATE's half ``gh``;
``gpos`` is the last step's position, ``gnext`` the next one's, ``gcur`` the half a prefill's next position starts from,
``kshift`` the k / v histories' and gates' first row (8 past a block's end). ``conv_in_ghdr``: a prefill's conv states wait in
GHDR (``conv_state``). ``committed`` / ``accepted``: the last step committed its tokens; its accepted ones are known.
"""

import dataclasses


@dataclasses.dataclass
class GdnCursor:
    nrep: int = 0
    cshift: int = 0
    kshift: int = 0
    gp: int = 0
    gh: int = 0
    gpos: int = 0
    gnext: int = 0
    gcur: int = 0
    conv_in_ghdr: bool = False
    committed: bool = True
    accepted: bool = True

    def reset(self) -> None:
        """A new sequence: nothing to replay, the blocks from position 0 (the last step's commit state stays)."""
        for f in dataclasses.fields(self):
            if f.name not in ("committed", "accepted"):
                setattr(self, f.name, f.default)

    def half(self, p: int) -> int:
        """The GSTATE half that holds the state at the start of position p's block: the half ``gh`` of ``gp``'s block, the
        other at each block after."""
        return (self.gh + p // 8 - self.gp // 8) & 1

    def advance(self, m: int) -> None:
        """(chunkwise) The last step's first m tokens accepted: the next step's histories' rows (past its block's end: the
        next block's, 8 rows on) and q base row."""
        self.kshift = 8 if self.gpos % 8 + m >= 8 else 0
        self.cshift = m
        self.gnext = self.gpos + m

    def copy(self) -> "GdnCursor":
        return dataclasses.replace(self)

    def assign(self, other: "GdnCursor") -> None:
        for f in dataclasses.fields(self):
            setattr(self, f.name, getattr(other, f.name))
