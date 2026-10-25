"""The router's models (``router``): the GGUF files of a models directory that Artlite runs, each with the vision
encoder of its directory that it takes, and the options and models of a preset file.

A model's id is ``<directory>:<quantization>`` - ``lmstudio-community/Llama-3.2-1B-Instruct-GGUF:Q4_0``, as
llama-server names the files of its Hugging Face cache, which the web UI groups by repository - with the file name's
last ``-`` / ``.`` part as the quantization when it names one, else the file's type (``general.file_type``:
``Occamy-1.0.APEX-I-MiniPlus-V2.1-Abliterated.gguf`` is ``:IQ3_XXS``); ``<directory>:<file name>`` when the directory
has two of one quantization, the file name alone at the directory's top. Its path in the directory (what ``-m`` takes)
and its file name (when no other file has it) resolve to it too.

A file is served when Artlite has its architecture (``models.ARCHITECTURES``) and a tile format for the type of each
of its matrices (``formats.has_tile_format``); a file Artlite cannot run is skipped with a log line that says why.
Drafters (``mtp-``, ``dspark-``, ``dflash-``, ``eagle3-``: their models find them themselves) and vision encoders
(architecture ``clip``) are not models of their own; split files are not read. A vision encoder goes with a model of
its directory whose class takes its projector type (``Model.projectors``) and its width; of several, the one of the
shortest name (``mmproj-model-f16.gguf``, not ``mmproj-model-f16-4x.gguf``). Reading a header takes ~0.1 s (its
vocabulary's strings are skipped one by one): the headers are cached in ``<cache>/server/models.json`` by path, size
and modification time, with each model's last load time (the web UI's loading bar).

The preset file is llama-server's (``--models-preset``): an INI file whose ``[*]`` section holds options of every
model, a section named by a model's id or alias the options of that model, and any other section a model of its own,
named by the section, whose ``model`` key is its file (absolute, else relative to the working directory or the models
directory). Keys are the options' long names without the dashes (``ctx-size``), their short ones (``c``) or their
environment variables (``ARTLITE_ARG_CTX_SIZE``), with ``true`` / ``false`` for switches; ``mmproj`` chooses the vision
encoder (empty: none), ``alias`` adds names (comma-separated), ``load-on-startup`` loads the model when the router
starts. A model's options: the command line's, else its section's, else ``[*]``'s."""

import argparse
import configparser
import json
import logging
import math
import os
import re
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Any

from artlite.gguf_model import GGUFFile
from artlite.operators.base import CACHE_DIR
from artlite.operators.formats import has_tile_format

from .. import ARCHITECTURES, model_class
from .main import ENV, MODEL, ROUTER, env_name

log = logging.getLogger("artlite.server")

VERSION = 1  # of the cache file
DRAFTERS = ("mtp-", "dspark-", "dflash-", "eagle3-")
SPLIT = re.compile(r"-(\d{5})-of-\d{5}$")
TAG = re.compile(r"[-.]([A-Za-z0-9_]+)$")
QUANT = re.compile(r"[IPT]?Q\d|B?F(16|32|64)|MXFP\d|NVFP\d", re.IGNORECASE)  # (a quantization's name)
PRESET_KEYS = ("model", "alias", "load-on-startup")  # a preset's keys that are no options of a model's server
TRUE, FALSE = ("1", "true", "on", "yes", "enabled"), ("0", "false", "off", "no", "disabled")


@dataclass
class Entry:
    """A model the router serves: its ``id`` in the API, its file and what the router reads of it (``header``), its
    vision encoder's file (None: text only), the other names that resolve to it (``names``, ``aliases``), its server's
    options (dest -> value) and where it comes from (``models_dir``, ``preset``)."""

    id: str
    path: str
    header: dict[str, Any]
    mmproj: str | None = None
    names: list[str] = field(default_factory=list)  # its path in the models directory, its file name
    aliases: list[str] = field(default_factory=list)  # a preset's (the API lists them)
    options: dict[str, Any] = field(default_factory=dict)
    source: str = "models_dir"
    load_on_startup: bool = False
    load_s: float = 0.0  # its last load's seconds (0: none yet)

    @property
    def name(self) -> str:
        """The file's name without .gguf (the log's)."""
        return os.path.basename(self.path).removesuffix(".gguf")

    @property
    def tag(self) -> str:
        """The quantization of the id: the file name's, else the file's type, else ""."""
        m = TAG.search(self.name)
        if m and QUANT.match(m[1]):
            return m[1].upper()
        ftype = self.header.get("ftype", "unknown")
        return "" if ftype == "unknown" else re.sub(r"_K - ([SML])[a-z]*$", r"_K_\1", ftype)


def header(path: str) -> dict[str, Any]:
    """What the router reads of a GGUF file: its architecture; a vision encoder's projector type and width; a model's
    context, sizes and file type, and the types of its matrices that have no tile format (``unsupported``)."""
    from ..bench import file_type_name

    g = GGUFFile(path)
    meta, arch = g.meta, g.arch
    if arch == "clip":
        return {"arch": arch, "projector": str(meta.get("clip.projector_type", "")),
                "projection_dim": int(meta.get("clip.vision.projection_dim", 0))}
    tensors = g.tensor_info.values()
    return {"arch": arch, "n_ctx_train": int(meta.get(f"{arch}.context_length", 0)),
            "n_embd": int(meta.get(f"{arch}.embedding_length", 0)),
            "n_vocab": g.string_count("tokenizer.ggml.tokens") if "tokenizer.ggml.tokens" in g._string_arrays else 0,
            "n_params": sum(math.prod(dims) for dims, _, _, _ in tensors), "size": sum(n or 0 for _, _, _, n in tensors),
            "ftype": file_type_name(meta.get("general.file_type")),
            "size_label": str(meta.get("general.size_label", "")),
            "unsupported": sorted({t for dims, t, _, _ in tensors if len(dims) >= 2 and t != "F32"
                                   and not has_tile_format(t)})}


def skip_reason(h: dict[str, Any]) -> str | None:
    """Why Artlite cannot run a model file of this header, or None."""
    if h["arch"] not in ARCHITECTURES:
        return f"architecture {h['arch']!r} (Artlite runs {', '.join(ARCHITECTURES)})"
    if h["unsupported"]:
        return f"{', '.join(h['unsupported'])} tensors: no tile format"
    return None


def preset_keys(parser: argparse.ArgumentParser) -> dict[str, tuple[argparse.Action, bool]]:
    """A preset's keys of model options: key -> (the option, whether the key negates it: ``no-warmup``)."""
    keys: dict[str, tuple[argparse.Action, bool]] = {}
    for a in parser._actions:
        if a.dest == "help" or a.dest in ROUTER or a.dest in MODEL - {"mmproj"}:
            continue
        for s in a.option_strings:
            keys[s.lstrip("-")] = (a, isinstance(a, argparse.BooleanOptionalAction) and s.startswith("--no-"))
        keys[a.dest] = (a, False)
        if (name := env_name(a)) is not None:
            keys[name] = (a, False)
    return keys


def boolean(text: str) -> bool:
    if text.lower() in TRUE:
        return True
    if text.lower() in FALSE:
        return False
    raise ValueError(f"{text!r} is not true or false")


def read_preset(path: str, parser: argparse.ArgumentParser) -> dict[str, dict[str, Any]]:
    """A preset file's sections: name -> {dest: value}, with ``model`` (a path), ``alias`` (a list of names) and
    ``load_on_startup`` as they are (``Catalog`` applies them). ValueError for what it cannot read."""
    keys = preset_keys(parser)
    routers = {k.lstrip("-") for a in parser._actions if a.dest in ROUTER for k in a.option_strings}
    cp = configparser.ConfigParser(interpolation=None, default_section="\0", comment_prefixes=("#", ";"),
                                   inline_comment_prefixes=None, strict=True)
    cp.optionxform = str  # type: ignore[assignment,method-assign] (keys keep their case: ARTLITE_ARG_...)
    try:
        with open(path) as f:
            cp.read_string("[\0top]\n" + f.read(), source=path)  # (a version = 1 line before the first section)
    except configparser.Error as e:
        raise ValueError(f"{path}: {e}") from None
    out: dict[str, dict[str, Any]] = {}
    for section in cp.sections():
        if section == "\0top":
            if set(cp[section]) - {"version"}:
                raise ValueError(f"{path}: keys before the first section: {', '.join(set(cp[section]) - {'version'})}")
            continue
        opts: dict[str, Any] = {}
        for key, text in cp[section].items():
            where = f"{path}: [{section}] {key}"
            if key in PRESET_KEYS:
                opts[key.replace("-", "_")] = (boolean(text) if key == "load-on-startup" else
                                               [n.strip() for n in text.split(",") if n.strip()] if key == "alias" else
                                               text.strip())
                continue
            if key in routers or key.startswith(ENV) and key[len(ENV):].lower() in ROUTER:
                raise ValueError(f"{where}: the router's option, not a model's")
            if key not in keys:
                names = sorted({k for k in keys if not k.startswith(ENV) and "_" not in k})
                raise ValueError(f"{where}: no such option (a model's: {', '.join(names)})")
            action, negated = keys[key]
            try:
                if action.nargs == 0:  # (a switch)
                    value: Any = boolean(text) != negated
                elif action.dest == "mmproj":
                    value = text.strip() or None  # (a path as ``model``'s; empty: no vision encoder)
                else:
                    value = action.type(text) if action.type is not None else text
                    if action.choices is not None and value not in action.choices:
                        raise ValueError(f"{text!r} (one of {', '.join(action.choices)})")
            except (ValueError, TypeError) as e:
                raise ValueError(f"{where}: {e}") from None
            opts[action.dest] = value
        out[section] = opts
    return out


def render(options: dict[str, Any], parser: argparse.ArgumentParser) -> list[str]:
    """The command line of options (dest -> value) for a model's server."""
    actions = {a.dest: a for a in parser._actions}
    argv: list[str] = []
    for dest, value in options.items():
        a = actions[dest]
        long = max((s for s in a.option_strings if s.startswith("--") and not s.startswith("--no-")), key=len,
                   default=a.option_strings[-1])
        if isinstance(a, argparse.BooleanOptionalAction):
            argv.append(long if value else "--no-" + long[2:])
        elif a.nargs == 0:
            if value:
                argv.append(long)
        elif value is not None:
            argv += [long, json.dumps(value) if isinstance(value, dict | list) else str(value)]
    return argv


class Catalog:
    """The models of a directory (``models_dir``) and a preset file (``preset``): ``scan`` reads them (a thread's
    work: it reads files), ``find`` resolves a name. ``options``: the command line's options of the models' servers
    (dest -> value), over a preset's; ``cache``: the cache directory."""

    def __init__(self, models_dir: str, *, parser: argparse.ArgumentParser, preset: str | None = None,
                 options: dict[str, Any] | None = None, cache: str = CACHE_DIR):
        self.models_dir = os.path.abspath(models_dir)
        self.parser, self.preset, self.options = parser, preset, dict(options or {})
        self.cache_file = os.path.join(cache, "server", "models.json")
        self.entries: dict[str, Entry] = {}
        self._names: dict[str, Entry] = {}
        self._cache: dict[str, Any] = {}
        self._lock = threading.Lock()  # (the cache: a scan's thread, a load's record)
        self._dirty = False
        self._told: set[str] = set()  # (files whose skipping was logged)

    # ------------------------------------------------------------------
    # The header cache
    # ------------------------------------------------------------------

    def _read_cache(self) -> None:
        try:
            with open(self.cache_file) as f:
                data = json.load(f)
            files = data["files"] if data.get("version") == VERSION else {}
        except (OSError, ValueError, KeyError, TypeError):
            files = {}
        with self._lock:
            self._cache = files

    def _write_cache(self) -> None:
        with self._lock:
            text = json.dumps({"version": VERSION, "files": self._cache})
            self._dirty = False
        try:
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.cache_file), prefix=".models.")
            with os.fdopen(fd, "w") as f:
                f.write(text)
            os.replace(tmp, self.cache_file)
        except OSError as e:
            log.debug("the models cache %s: %s", self.cache_file, e)

    def _header(self, path: str) -> dict[str, Any]:
        """A file's header, from the cache when its size and modification time are the cached ones."""
        st = os.stat(path)
        key = os.path.realpath(path)
        c = self._cache.get(key)
        if c is None or c.get("size") != st.st_size or c.get("mtime_ns") != st.st_mtime_ns:
            c = {"size": st.st_size, "mtime_ns": st.st_mtime_ns, "header": header(path)}
            with self._lock:
                self._cache[key] = c
                self._dirty = True
        return c["header"]

    def record_load(self, entry: Entry, seconds: float) -> None:
        """A model's load time (its next load's loading bar)."""
        entry.load_s = seconds
        with self._lock:
            c = self._cache.get(os.path.realpath(entry.path))
            if c is not None:
                c["load_s"] = round(seconds, 2)
        if c is not None:
            self._write_cache()

    def _skip(self, rel: str, reason: object) -> None:
        if rel not in self._told:
            self._told.add(rel)
            log.info("not served: %s (%s)", rel, reason)

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def files(self) -> list[str]:
        """The GGUF files under the models directory (symbolic links followed, each directory once)."""
        out, seen = [], set()
        for root, dirs, names in os.walk(self.models_dir, followlinks=True):
            real = os.path.realpath(root)
            if real in seen:
                dirs[:] = []
                continue
            seen.add(real)
            dirs.sort()
            out += [os.path.join(root, n) for n in sorted(names) if n.endswith(".gguf")]
        return out

    def scan(self) -> dict[str, Entry]:
        """Read the directory and the preset file (again): ``entries`` by id. ValueError: the preset file's."""
        self._read_cache()
        models: list[tuple[str, dict[str, Any]]] = []  # (path, header)
        encoders: dict[str, list[tuple[str, dict[str, Any]]]] = {}  # directory -> its vision encoders
        for path in self.files() if os.path.isdir(self.models_dir) else []:
            rel = self.rel(path)
            stem = os.path.basename(path).removesuffix(".gguf")
            if stem.startswith(DRAFTERS) or "-mtp-" in stem:  # (an uploader's prefixed drafter: occamy-1.0-mtp-Q8_0)
                continue
            if m := SPLIT.search(stem):
                if m[1] == "00001":
                    self._skip(rel, "a split file")
                continue
            try:
                h = self._header(path)
            except (OSError, ValueError, AssertionError, KeyError, NotImplementedError) as e:
                self._skip(rel, e)
                continue
            if h["arch"] == "clip":
                encoders.setdefault(os.path.dirname(path), []).append((path, h))
            elif (reason := skip_reason(h)) is not None:
                self._skip(rel, reason)
            else:
                models.append((path, h))
        entries = self._ids(models)
        sections = read_preset(self.preset, self.parser) if self.preset else {}
        for opts in sections.values():
            if opts.get("mmproj"):
                opts["mmproj"] = self.path(opts["mmproj"])
        common = sections.pop("*", {})
        for name, opts in sections.items():
            if "model" in opts:
                path = self.path(opts["model"])
                try:
                    h = self._header(path)
                except (OSError, ValueError, AssertionError, KeyError) as e:
                    log.warning("preset [%s]: %s: %s", name, path, e)
                    continue
                if h["arch"] == "clip" or (reason := skip_reason(h)) is not None:
                    log.warning("preset [%s]: %s: %s", name, path, "a vision encoder" if h["arch"] == "clip" else
                                reason)
                    continue
                entries[name] = Entry(id=name, path=path, header=h, source="preset")
            elif (entry := self._lookup(entries, name)) is None:
                log.warning("preset [%s]: no such model (a section of another model needs a 'model' key)", name)
                continue
            else:
                name = entry.id
            entry = entries[name]
            entry.aliases += opts.pop("alias", [])
            entry.load_on_startup = opts.pop("load_on_startup", False)
            opts.pop("model", None)
            entry.options = opts
        for entry in entries.values():
            opts = {**common, **entry.options, **self.options}
            if "mmproj" in opts:
                entry.mmproj = opts.pop("mmproj")
            else:
                entry.mmproj = self._encoder(entry, encoders.get(os.path.dirname(entry.path), []))
            entry.options = opts
            entry.load_s = float(self._cache.get(os.path.realpath(entry.path), {}).get("load_s", 0.0))
        if self._dirty:
            self._write_cache()
        names: dict[str, Entry] = {}
        for entry in entries.values():
            for n in [entry.id, *entry.aliases, *entry.names]:
                if n in names and names[n] is not entry:
                    log.warning("%s names both %s and %s: it resolves to the first", n, names[n].id, entry.id)
                else:
                    names[n] = entry
        entries = dict(sorted(entries.items()))
        self.entries, self._names = entries, names  # (the event loop's ``find`` meanwhile: the old ones)
        return entries

    def path(self, path: str) -> str:
        """A preset's file: absolute, else relative to the working directory or the models directory."""
        return os.path.abspath(path if os.path.isabs(path) or os.path.exists(path) else
                               os.path.join(self.models_dir, path))

    def rel(self, path: str) -> str:
        """A path relative to the models directory, with / (as ``-m`` takes it), or the path outside it."""
        rel = os.path.relpath(path, self.models_dir)
        return path if rel.startswith("..") else rel.replace(os.sep, "/")

    def _ids(self, models: list[tuple[str, dict[str, Any]]]) -> dict[str, Entry]:
        """The entries of the directory's model files, by id (module docstring)."""
        entries = [Entry(id="", path=path, header=h) for path, h in models]
        tags: dict[tuple[str, str], int] = {}
        names: dict[str, int] = {}
        for e in entries:
            d = os.path.dirname(self.rel(e.path))
            tags[d, e.tag] = tags.get((d, e.tag), 0) + 1
            names[os.path.basename(e.path)] = names.get(os.path.basename(e.path), 0) + 1
        out: dict[str, Entry] = {}
        for e in entries:
            rel = self.rel(e.path)
            d = os.path.dirname(rel)
            e.id = (f"{d}:{e.tag}" if e.tag and tags[d, e.tag] == 1 else f"{d}:{e.name}") if d else e.name
            e.names = [rel] + ([os.path.basename(rel)] if names[os.path.basename(rel)] == 1 and d else [])
            out[e.id] = e
        return out

    @staticmethod
    def _encoder(entry: Entry, encoders: list[tuple[str, dict[str, Any]]]) -> str | None:
        """The vision encoder of the model's directory that its class takes (module docstring), or None."""
        if not encoders:
            return None
        projectors = model_class(entry.header["arch"]).projectors
        fit = [p for p, h in encoders if h["projector"] in projectors
               and h["projection_dim"] in (0, entry.header["n_embd"])]
        return min(fit, key=lambda p: (len(os.path.basename(p)), p)) if fit else None

    # ------------------------------------------------------------------
    # Names
    # ------------------------------------------------------------------

    @staticmethod
    def _lookup(entries: dict[str, Entry], name: str) -> Entry | None:
        return entries.get(name) or next((e for e in entries.values() if name in e.aliases + e.names), None)

    def find(self, name: str) -> Entry | None:
        """The model of an id, an alias or its file's path."""
        if name in self._names:
            return self._names[name]
        real = os.path.realpath(name if os.path.isabs(name) else os.path.join(self.models_dir, name))
        return next((e for e in self.entries.values() if os.path.realpath(e.path) == real), None)

    def argv(self, entry: Entry) -> list[str]:
        """The command line of a model's server (``main``'s), but the router's own options (``Router``'s)."""
        argv = ["-m", entry.path, "-a", entry.id]
        if entry.mmproj:
            argv += ["--mmproj", entry.mmproj]
        return argv + render(entry.options, self.parser)
