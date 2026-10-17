"""artlite-cli with D draft tokens per dispatch (S = D + 1 tokens per dispatch) for a model with a drafter
(its ``NpuModel.DRAFTS``; 0: the steps of one token, the drafter loaded but unused):

    uv run python .claude/skills/artlite-bench/drafts.py D <artlite-cli arguments>
"""

import sys

from models import cli
from models.common import Model, NpuModel


def main() -> None:
    drafts = int(sys.argv[1])
    open_model = cli.open_model

    def opened(*args: object, **kwargs: object) -> Model:
        model = open_model(*args, **kwargs)
        load = model.load

        def loaded(**kw: object) -> NpuModel:
            npu = load(**kw)
            if not hasattr(npu, "DRAFTS"):
                sys.exit(f"drafts.py: {model.name} has no drafter (NpuModel.DRAFTS)")
            if not 0 <= drafts <= type(npu).DRAFTS:
                sys.exit(f"drafts.py: {model.name} takes 0 .. {type(npu).DRAFTS} draft tokens per dispatch")
            npu.DRAFTS = drafts
            return npu

        model.load = loaded
        return model

    cli.open_model = opened
    cli.main(sys.argv[2:])


if __name__ == "__main__":
    main()
