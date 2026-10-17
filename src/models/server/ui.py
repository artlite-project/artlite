"""The web UI: llama.cpp's own (``tools/ui`` of https://github.com/ggml-org/llama.cpp, MIT license, copyright The
ggml authors), prebuilt and unmodified. ``fetch`` gets the build pinned here - the ``dist.tar.gz`` of llama.cpp build
``TAG`` from the Hugging Face bucket that llama.cpp's CMake downloads it from, checked against ``SHA256`` - into
``<cache>/ui/<TAG>/`` on the first start (later starts need no network); ``mount`` serves a UI directory at ``/``.

The UI is a static single-page app (hash routes, relative paths: it works at ``/``) that calls ``GET ./props``
(retried every second on a 503 while the model loads; a 401 shows its API-key screen), ``GET /v1/models`` and a
streamed ``POST ./v1/chat/completions``; its other calls (``/tools``, ``/v1/streams/lookup``, ``DELETE /v1/stream``
for Stop) tolerate the answers ``api`` gives. Bumping the pin: diff the UI's ``src/lib/constants/
api-endpoints.constants.ts`` and ``src/lib/services/chat.service.ts`` (request fields, the stream fields it reads)
between the two builds against what ``api`` implements, then set the new tag and the sha256 of its archive (the
bucket's ``dist.tar.gz.sha256``)."""

import hashlib
import mimetypes
import os
import shutil
import tarfile
import tempfile
import urllib.request

from fastapi import FastAPI
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

from artlite.operators.base import CACHE_DIR

TAG = "b11699"
SHA256 = "bb99387bab7e55aaad19495e2e617e82bc8ff7ce547f967c4f1e0927fe7764be"
URL = "https://huggingface.co/buckets/ggml-org/llama-ui/resolve/{tag}/dist.tar.gz?download=true"
UNCACHED = ("index.html", "sw.js", "build.json", "_app/version.json")  # (the UI checks them for updates)

mimetypes.add_type("application/manifest+json", ".webmanifest")


def fetch(*, tag: str = TAG, sha256: str = SHA256, cache: str = CACHE_DIR, timeout: float = 60.0) -> str:
    """The directory of the pinned UI build: the cached copy, else downloaded, verified and unpacked. OSError (or
    ValueError: a wrong checksum) when it cannot be fetched."""
    out = os.path.join(cache, "ui", tag)
    if os.path.isfile(os.path.join(out, "index.html")):
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with tempfile.TemporaryDirectory(dir=os.path.dirname(out)) as tmp:
        archive = os.path.join(tmp, "dist.tar.gz")
        with urllib.request.urlopen(URL.format(tag=tag), timeout=timeout) as r, open(archive, "wb") as f:
            shutil.copyfileobj(r, f)
        with open(archive, "rb") as f:
            digest = hashlib.file_digest(f, "sha256").hexdigest()
        if digest != sha256:
            raise ValueError(f"the UI archive of {tag} has sha256 {digest}, not {sha256}")
        files = os.path.join(tmp, "files")
        with tarfile.open(archive) as t:
            t.extractall(files, filter="data")
        if not os.path.isfile(os.path.join(files, "index.html")):
            raise ValueError(f"the UI archive of {tag} has no index.html")
        shutil.rmtree(out, ignore_errors=True)
        os.replace(files, out)
    return out


class UiFiles(StaticFiles):
    """The UI's files; the ones it checks for a new version are revalidated on every request."""

    async def get_response(self, path: str, scope: Scope) -> Response:
        response = await super().get_response(path, scope)
        if path in ("", ".", *UNCACHED):
            response.headers["Cache-Control"] = "no-cache"
        return response


def mount(app: FastAPI, directory: str) -> None:
    """Serve a UI build (its ``index.html`` at ``/``) after the app's routes."""
    app.mount("/", UiFiles(directory=directory, html=True), name="ui")
