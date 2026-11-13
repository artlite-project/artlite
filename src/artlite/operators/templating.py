"""Kernel templates: the operators' C++ kernel sources, Jinja2 files next to the modules that render them.

A kernel template is a file of the artlite or models package next to its module, ``<module>.cc.jinja`` (a module of
several: ``<module>.<part>.cc.jinja``), named by its path from the packages' directory (``"artlite/decode/attn.cc.jinja"``)
in Python and in ``{% include %}`` alike:

- ``load(name)``: the file's template (Jinja drops a template's last newline);
- ``source(name)``: the file's text, for a template composed in Python (``JINJA_ENV.from_string(a + b)``: pieces that
  share ``{% set %}`` names) or rendered by another environment (the format slots' fragments, ``formats.kernel_template``);
- ``{% include "<name>" %}`` on a line of its own splices the file's lines whole, as concatenating its text would (the
  loader serves a file with one newline more: the one Jinja drops; the newline after the tag goes with trim_blocks).

``JINJA_ENV``: trim_blocks removes the newline after a block tag, lstrip_blocks the whitespace before one that starts a
line (both keep the generated C++ clean); StrictUndefined: a variable the render does not pass is an error, not an empty
string; ``km``: the math functions generated per contract (``kmath``), in every template.
"""

import functools
import os
from collections.abc import Callable

from jinja2 import Environment, FileSystemLoader, StrictUndefined, Template

from . import kmath

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # (the artlite and models packages')


class _Loader(FileSystemLoader):
    """The packages' files by their path from ``ROOT``, each with one newline more (``{% include %}``: whole lines)."""

    def get_source(self, environment: Environment, template: str) -> tuple[str, str, Callable[[], bool] | None]:
        text, filename, uptodate = super().get_source(environment, template)
        return text + "\n", filename, uptodate


JINJA_ENV = Environment(loader=_Loader(ROOT), trim_blocks=True, lstrip_blocks=True, undefined=StrictUndefined)
JINJA_ENV.globals["km"] = kmath


@functools.cache
def source(name: str) -> str:
    """The text of the template file ``name`` (its path from the packages' directory)."""
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return f.read()


@functools.cache
def load(name: str) -> Template:
    """The template of the file ``name`` (its path from the packages' directory)."""
    return JINJA_ENV.from_string(source(name))
