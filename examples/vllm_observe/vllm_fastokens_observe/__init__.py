"""vLLM general plugin: serve the fastokens observe dashboard from every vLLM
process that tokenizes.

With ``FASTOKENS_OBSERVE_PORT=<port>`` set (and ``VLLM_USE_FASTOKENS=1``), each
process that creates a fastokens tokenizer (the API servers, and the tokenizer
copies of their renderer pools) serves ``fastokens.observe.serve()`` on the
first free port from ``<port>`` up: ``/`` (dashboard), ``/json`` and
``/metrics``. The port each process got is logged ("fastokens observe:
pid=... serving http://127.0.0.1:PORT"); ``/json`` also carries the ``pid``.
Every tokenizer of the process is listed (``tok0``, ``tok1``, ...), each with
its own caches. ``FASTOKENS_OBSERVE_HOST`` changes the bind address (default
127.0.0.1; the server has no authentication).

Without ``FASTOKENS_OBSERVE_PORT`` the plugin does nothing.
"""

import logging
import os
import threading

_PORTS_TRIED = 64

_lock = threading.Lock()
_tokenizers = []          # every native tokenizer created in this process, in order
_server = None
_port = None


def _log():
    try:
        from vllm.logger import init_logger

        # Under "vllm." so that vLLM's log handler (and its process prefix) applies.
        return init_logger("vllm.fastokens_observe")
    except Exception:  # outside vLLM
        return logging.getLogger(__name__)


def _serve_all():
    """(Re)start this process's observe server so it lists every tokenizer
    created so far. Called with ``_lock`` held, once per new tokenizer (a few
    times at startup)."""
    global _server, _port
    import fastokens.observe as obs

    toks = {f"tok{i}": t for i, t in enumerate(_tokenizers)}
    host = os.environ.get("FASTOKENS_OBSERVE_HOST", "127.0.0.1")
    if _server is not None:
        _server.shutdown()
        _server.server_close()
        _server = obs.serve(toks, port=_port, host=host)
        return
    base = int(os.environ["FASTOKENS_OBSERVE_PORT"])
    for port in range(base, base + _PORTS_TRIED):
        try:
            _server = obs.serve(toks, port=port, host=host)
        except OSError:
            continue
        _port = port
        _log().info("fastokens observe: pid=%d serving http://%s:%d (observe build: %s)",
                    os.getpid(), host, port, obs.enabled())
        return
    _log().warning("fastokens observe: no free port in %d-%d", base, base + _PORTS_TRIED - 1)


def _add(shim):
    with _lock:
        _tokenizers.append(shim._fast)
        try:
            _serve_all()
        except Exception as e:  # never break tokenizer creation
            _log().warning("fastokens observe: could not serve: %r", e)


def register():
    if not os.environ.get("FASTOKENS_OBSERVE_PORT"):
        return
    try:
        from fastokens import _compat
    except ImportError:
        _log().warning("fastokens observe: fastokens is not installed")
        return
    shim = _compat._TokenizerShim
    if getattr(shim, "_observe_plugin", False):
        return
    shim._observe_plugin = True
    init, deepcopy = shim.__init__, shim.__deepcopy__

    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        _add(self)

    def __deepcopy__(self, memo):
        new = deepcopy(self, memo)
        _add(new)
        return new

    shim.__init__, shim.__deepcopy__ = __init__, __deepcopy__
