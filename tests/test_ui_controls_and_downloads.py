from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "streamlit_app.py"


def _app_text() -> str:
    return APP_PATH.read_text(encoding="utf-8", errors="ignore")


def test_s2e_has_edge_weight_sliders():
    text = _app_text()
    assert 'key="s2e_n_power"' in text
    assert 'key="s2e_sigma"' in text


def test_histone_has_edge_weight_sliders():
    text = _app_text()
    assert 'key="hist_n_power"' in text
    assert 'key="hist_sigma"' in text


def test_diffusion_simulator_has_edge_weight_sliders():
    text = _app_text()
    assert 'key="diffusion_edge_power"' in text
    assert 'key="diffusion_edge_sigma"' in text


def test_download_buttons_exist_for_all_visualizations():
    text = _app_text()
    required_keys = [
        'key="download_diffusion_centrality"',
        'key="download_diffusion_pca3d"',
        'key="download_diffusion_visits"',
        'key="download_start_to_end_bundle"',
        'key="download_histone_bundle"',
    ]
    for key in required_keys:
        assert key in text


def test_pipeline_sliders_are_assigned_to_module_globals():
    text = _app_text()
    assert "start_to_end_module.n_power = int(n_power_s2e)" in text
    assert "start_to_end_module.sigma = float(sigma_s2e)" in text
    assert "histone_module.n_power = int(n_power_hist)" in text
    assert "histone_module.sigma = float(sigma_hist)" in text


def test_download_bundles_write_parameters_json():
    text = _app_text()
    assert 'zf.writestr("parameters.json"' in text


def test_start_to_end_download_label_present():
    text = _app_text()
    assert "Download start-to-end visualization + parameters" in text


def test_histone_download_label_present():
    text = _app_text()
    assert "Download histone visualization + parameters" in text


def test_diffusion_download_labels_present():
    text = _app_text()
    assert "Download centrality visualization + parameters" in text
    assert "Download 3D PCA visualization + parameters" in text
    assert "Download visit-frequency chart + parameters" in text
