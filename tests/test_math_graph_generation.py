import numpy as np
import pandas as pd

from Dissertation.streamlit_app import build_adjacency
from DiffusionRWR_model_package.graph_generation.edge_weight_functions import (
    cor_exponential_abs_inter,
    cor_gaussian_abs_inter,
    inter_layer_corr_power,
    intra_layer_corr_exponential_shifted,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "0": [0.0, 1.0, 2.0],
            "1": [1.0, 1.0, 0.0],
            "2": [2.0, 1.0, -1.0],
            "3": [3.0, 1.0, -2.0],
        },
        index=["g1", "g2", "g3"],
    )


def test_build_adjacency_corr_power_matches_formula():
    df = _sample_df()
    out = build_adjacency(df, method="corr_power", n_power=4, sigma=0.05)

    corr = df.T.corr().fillna(0.0)
    expected = np.abs(0.5 * (corr.values + 1.0)) ** 4
    np.fill_diagonal(expected, 0.0)

    assert np.allclose(out.values, expected)


def test_build_adjacency_gaussian_matches_formula():
    df = _sample_df()
    sigma = 0.2
    out = build_adjacency(df, method="cor_gaussian_abs", n_power=20, sigma=sigma)

    corr = df.T.corr().fillna(0.0)
    expected = np.exp(-0.5 * ((1.0 - np.abs(corr.values)) / sigma) ** 2)
    np.fill_diagonal(expected, 0.0)

    assert np.allclose(out.values, expected)


def test_build_adjacency_exponential_matches_formula():
    df = _sample_df()
    sigma = 0.2
    out = build_adjacency(df, method="cor_exponential_abs", n_power=20, sigma=sigma)

    corr = df.T.corr().fillna(0.0)
    expected = np.exp(-(1.0 - np.abs(corr.values)) / sigma)
    np.fill_diagonal(expected, 0.0)

    assert np.allclose(out.values, expected)


def test_build_adjacency_has_zero_diagonal_for_all_methods():
    df = _sample_df()
    methods = ["corr_power", "cor_gaussian_abs", "cor_exponential_abs"]
    for method in methods:
        out = build_adjacency(df, method=method, n_power=20, sigma=0.05)
        assert np.allclose(np.diag(out.values), 0.0)


def test_build_adjacency_is_symmetric():
    df = _sample_df()
    out = build_adjacency(df, method="cor_exponential_abs", n_power=20, sigma=0.05)
    assert np.allclose(out.values, out.values.T)


def test_inter_layer_corr_power_changes_with_n_power():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.1, 0.9, 1.8, 2.7])

    w2 = inter_layer_corr_power(a, b, n_power=2)
    w20 = inter_layer_corr_power(a, b, n_power=20)

    assert w2 != w20


def test_cor_exponential_abs_inter_changes_with_sigma():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([3.0, 2.1, 1.2, 0.1])

    w_small = cor_exponential_abs_inter(a, b, sigma=0.01)
    w_large = cor_exponential_abs_inter(a, b, sigma=0.5)

    assert w_small != w_large


def test_cor_gaussian_abs_inter_changes_with_sigma():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([3.0, 2.1, 1.2, 0.1])

    w_small = cor_gaussian_abs_inter(a, b, sigma=0.01)
    w_large = cor_gaussian_abs_inter(a, b, sigma=0.5)

    assert w_small != w_large


def test_intra_layer_corr_exponential_shifted_changes_with_sigma():
    a = np.array([0.0, 1.0, 2.0, 3.0])
    b = np.array([0.0, 1.0, 1.8, 2.6])

    w_small = intra_layer_corr_exponential_shifted(a, b, sigma=0.1)
    w_large = intra_layer_corr_exponential_shifted(a, b, sigma=1.0)

    assert w_small != w_large
