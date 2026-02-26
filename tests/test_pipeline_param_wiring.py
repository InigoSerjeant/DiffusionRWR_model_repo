import inspect

import DiffusionRWR_model_package.diffusion_functions.start_to_end as start_to_end
import DiffusionRWR_model_package.diffusion_functions.histone_cluster_analysis as histone


MODULES = [start_to_end, histone]


def _fn_with_sigma(a, sigma=0.1):
    return (a, sigma)


def _fn_with_n_power(a, n_power=5):
    return (a, n_power)


def _fn_with_both(a, b=None, sigma=0.1, n_power=5):
    return (a, b, sigma, n_power)


def _fn_with_neither(a):
    return a


def test_call_edge_fn_with_params_passes_sigma_when_supported():
    for module in MODULES:
        out = module._call_edge_fn_with_params(_fn_with_sigma, 10, sigma_value=0.77, n_power_value=13)
        assert out[1] == 0.77


def test_call_edge_fn_with_params_passes_n_power_when_supported():
    for module in MODULES:
        out = module._call_edge_fn_with_params(_fn_with_n_power, 10, sigma_value=0.77, n_power_value=13)
        assert out[1] == 13


def test_call_edge_fn_with_params_passes_both_when_supported():
    for module in MODULES:
        out = module._call_edge_fn_with_params(_fn_with_both, 10, 20, sigma_value=0.33, n_power_value=17)
        assert out[2] == 0.33
        assert out[3] == 17


def test_call_edge_fn_with_params_does_not_break_without_supported_kwargs():
    for module in MODULES:
        out = module._call_edge_fn_with_params(_fn_with_neither, 10, sigma_value=0.33, n_power_value=17)
        assert out == 10


def test_start_to_end_source_uses_param_aware_intra_and_inter_wrappers():
    source = inspect.getsource(start_to_end.start_to_end)
    assert "def edge_fn_intra_param" in source
    assert "sigma_value=sigma" in source
    assert "n_power_value=n_power" in source
    assert "edge_fn_inter=edge_fn_inter_param" in source


def test_histone_source_uses_param_aware_intra_and_inter_wrappers():
    source = inspect.getsource(histone.histone_cluster_analysis)
    assert "def edge_fn_intra_param" in source
    assert "sigma_value=sigma" in source
    assert "n_power_value=n_power" in source
    assert "edge_fn_inter=edge_fn_inter_param" in source
