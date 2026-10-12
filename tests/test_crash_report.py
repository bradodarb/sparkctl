"""Crash-cause extraction — must surface the real exception, not the torch teardown noise."""
from sparkctl.cli.resource import _extract_cause

# The exact shape that masked muse-glimmer's failure: a real KeyError, then the interpreter-teardown
# UnicodeDecodeError (markerless exception line; torch finalizer frames sit above it).
MUSE_LOG = """\
(APIServer pid=1) ERROR abstract_tool_parser.py:376] FileNotFoundError: [Errno 2] No such file or directory: '/opt/sparkctl-plugins/atem_tool_parser.py'
(APIServer pid=1) Traceback (most recent call last):
(APIServer pid=1)     raise KeyError(
(APIServer pid=1) KeyError: 'invalid tool call parser: atem (chose from { hermes,qwen3_xml })'
(APIServer pid=1) Traceback (most recent call last):
(APIServer pid=1)   File "/usr/lib/python3.12/weakref.py", line 590, in __call__
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/torch/library.py", line 628, in _clear_torch_ops_cache
(APIServer pid=1)   File "/usr/local/lib/python3.12/dist-packages/torch/_ops.py", line 1412, in _get_packet
(APIServer pid=1)     op, overload_names = torch._C._jit_get_operation(qualname)
(APIServer pid=1) UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89 in position 180: invalid start byte""".splitlines()


def test_teardown_unicode_error_is_skipped_for_real_cause():
    cause = _extract_cause(MUSE_LOG)
    assert "invalid tool call parser: atem" in cause      # the real, actionable error
    assert "UnicodeDecodeError" not in cause               # not the teardown ghost


def test_core_engine_error_preferred():
    lines = [
        "(EngineCore pid=2) ERROR core.py:1165] ValueError: model type `muse_glimmer` unrecognized",
        "(APIServer pid=1) Traceback (most recent call last):",
        "(APIServer pid=1)   File weakref.py, in __call__",
        "(APIServer pid=1)   torch._C._jit_get_operation(qualname)",
        "(APIServer pid=1) UnicodeDecodeError: 'utf-8' codec can't decode byte 0xa1",
    ]
    assert "muse_glimmer" in _extract_cause(lines)


def test_no_exception_returns_empty():
    assert _extract_cause(["INFO starting up", "INFO loading weights"]) == ""
