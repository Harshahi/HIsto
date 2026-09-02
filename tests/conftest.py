"""Shared fixtures: load a notebook's definitions into a plain namespace.

The notebooks are the source of truth in this project -- there is no importable
module -- so the tests exec the notebooks' own code cells. That way a test can
never drift from the notebook it is testing.
"""
import contextlib
import io
import os
import pathlib
import re

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")     # before any matplotlib import

ROOT = pathlib.Path(__file__).resolve().parents[1]

_MAGIC = re.compile(r"^(\s*)([%!].*)$", re.MULTILINE)
# Cells that draw: slow, and they write into figures/. The end-to-end test covers them.
_FIGURE_MARKERS = ("fig.savefig", "plt.subplots")


def strip_magics(src):
    """Comment out IPython magics (%matplotlib inline, !pip ...) so exec() accepts them."""
    return _MAGIC.sub(r"\1# [stripped for exec] \2", src)


def load_notebook(name, skip_figures=True):
    """exec a notebook's code cells in order, returning (namespace, skipped).

    Cells that fail to compile or raise are recorded in `skipped` rather than
    aborting the load, so a notebook that is mid-edit still yields its working
    definitions. stdout is swallowed to keep test output readable.
    """
    import json

    nb = json.loads((ROOT / name).read_text())
    ns, skipped = {"__name__": f"<{name}>"}, []
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = strip_magics("".join(cell["source"]))
        cid = cell.get("id")
        if not src.strip():
            continue
        if skip_figures and any(k in src for k in _FIGURE_MARKERS):
            continue
        try:
            compiled = compile(src, f"{name}[{cid}]", "exec")
        except SyntaxError as e:
            skipped.append((cid, f"SyntaxError: {e.msg}"))
            continue
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                exec(compiled, ns)
        except Exception as e:                     # noqa: BLE001 - recorded, not raised
            skipped.append((cid, f"{type(e).__name__}: {e}"))
    return ns, skipped


def code_cells(name):
    """[(cell_id, source)] for every non-empty code cell."""
    import json

    nb = json.loads((ROOT / name).read_text())
    return [(c.get("id"), "".join(c["source"])) for c in nb["cells"]
            if c["cell_type"] == "code" and "".join(c["source"]).strip()]


@pytest.fixture(scope="session")
def encode_ns():
    ns, skipped = load_notebook("encode.ipynb")
    assert "histo_encode" in ns, f"encode.ipynb did not define histo_encode; skipped={skipped}"
    return ns


@pytest.fixture(scope="session")
def encode_m_ns():
    ns, skipped = load_notebook("encode_m.ipynb")
    assert "histo_encode" in ns, f"encode_m.ipynb did not define histo_encode; {skipped}"
    return ns
