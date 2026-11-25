"""The app must survive every (p, m, K) the sidebar can reach, not just 3,3,3.

This caught a real one: all_d_layers globbed every grid file on disk without
checking which code it was for, so changing p, m or K left the Gaussian tab
showing p=3, m=3, K=3 numbers under the new labels. No exception, no warning --
the heatmap just described a different code than the sidebar did.
"""
import pytest

APP = "/Users/gamma/CodeBase/Histogram/streamlit_app/app.py"
CASES = [(3, 3, 3), (2, 3, 2), (2, 4, 3), (3, 2, 2),
         (5, 2, 3), (5, 2, 4), (7, 1, 2), (3, 3, 5), (2, 3, 4)]


@pytest.mark.parametrize("p,m,K", CASES, ids=lambda v: str(v))
def test_app_runs_at_these_parameters(p, m, K):
    AppTest = pytest.importorskip("streamlit.testing.v1",
                                  reason="needs the app group").AppTest
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    assert not at.exception, f"load failed: {at.exception}"
    at.session_state["p"], at.session_state["m"], at.session_state["K"] = p, m, K
    at.run()
    assert not at.exception, \
        f"p={p} m={m} K={K}: " + " | ".join(e.value.split(chr(10))[0]
                                            for e in at.exception)
