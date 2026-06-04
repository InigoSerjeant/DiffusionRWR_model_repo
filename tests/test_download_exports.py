import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from Dissertation.streamlit_app import _build_export_zip, _json_safe


def test_json_safe_converts_path_numpy_and_collections():
    payload = {
        "path": Path("a/b/c"),
        "arr": np.array([1, 2, 3]),
        "float": np.float64(1.5),
        "int": np.int64(7),
        "idx": pd.Index(["x", "y"]),
        "nested": {"tuple": (1, 2)},
    }

    converted = _json_safe(payload)

    assert converted["path"] == "a/b/c"
    assert converted["arr"] == [1, 2, 3]
    assert converted["float"] == 1.5
    assert converted["int"] == 7
    assert converted["idx"] == ["x", "y"]
    assert converted["nested"]["tuple"] == [1, 2]


def test_export_zip_contains_parameters_json():
    zip_bytes = _build_export_zip(
        base_name="example",
        params={"alpha": 0.1, "n_simulations": 1000},
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        names = set(zf.namelist())
        assert "parameters.json" in names
        params = json.loads(zf.read("parameters.json").decode("utf-8"))
        assert params["alpha"] == 0.1
        assert params["n_simulations"] == 1000


def test_export_zip_contains_plotly_html_when_figure_is_passed():
    fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
    zip_bytes = _build_export_zip(base_name="plot_test", params={"k": 1}, plotly_fig=fig)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        assert "plot_test.html" in zf.namelist()
        html = zf.read("plot_test.html").decode("utf-8")
        assert "plotly" in html.lower()


def test_export_zip_contains_raw_html_when_content_passed():
    raw_html = "<html><body><h1>Visualization</h1></body></html>"
    zip_bytes = _build_export_zip(base_name="raw_html", params={"x": 1}, html_content=raw_html)

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        assert "raw_html.html" in zf.namelist()
        html = zf.read("raw_html.html").decode("utf-8")
        assert "Visualization" in html


def test_export_zip_contains_csv_frames():
    df = pd.DataFrame({"node": ["a", "b"], "visit": [0.2, 0.3]})
    zip_bytes = _build_export_zip(
        base_name="csv_test",
        params={"method": "corr_power"},
        csv_frames={"top_visits": df},
    )

    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        assert "top_visits.csv" in zf.namelist()
        csv_text = zf.read("top_visits.csv").decode("utf-8")
        assert "node,visit" in csv_text
        assert "a,0.2" in csv_text


def test_export_zip_is_valid_zip_file():
    zip_bytes = _build_export_zip(base_name="valid", params={"ok": True})
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        assert zf.testzip() is None
