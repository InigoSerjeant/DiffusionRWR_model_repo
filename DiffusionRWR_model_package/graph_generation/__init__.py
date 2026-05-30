from ..data_sorting import load_data
from .edge_weight_functions import corr_power, cor_gaussian_shifted, cor_gaussian_abs, cor_abs_inter
from .generate_graph_internal import generate_single_layer_graphs, lasso_single_graph
from .generate_multi_graph import create_multigraph, create_shadow_network_multigraph_lasso

__all__ = [
    'load_data',
    'corr_power',
    'cor_gaussian_shifted',
    'cor_gaussian_abs',
    'cor_abs_inter',
    'generate_single_layer_graphs',
    'lasso_single_graph',
    'create_multigraph',
    'create_shadow_network_multigraph_lasso',
    'wgcna_scale_invariance_r2',
    'wgcna_scale_invariance_from_adjacency',
]


def __getattr__(name):
    if name in {'wgcna_scale_invariance_r2', 'wgcna_scale_invariance_from_adjacency'}:
        from .tune_params import wgcna_scale_invariance_r2, wgcna_scale_invariance_from_adjacency

        exports = {
            'wgcna_scale_invariance_r2': wgcna_scale_invariance_r2,
            'wgcna_scale_invariance_from_adjacency': wgcna_scale_invariance_from_adjacency,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
