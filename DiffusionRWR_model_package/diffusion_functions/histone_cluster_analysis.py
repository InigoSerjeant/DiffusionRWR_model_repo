import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from pathlib import Path
import inspect
import plotly.graph_objects as go

# Import subpackages
from DiffusionRWR_model_package.graph_generation.preprocess_data import load_data
from DiffusionRWR_model_package.graph_generation.generate_graph_internal import generate_single_layer_graphs, generate_negative_correlation_graphs
from DiffusionRWR_model_package.graph_generation.generate_multi_graph import create_multigraph, create_multigraph_with_layer_transitions, create_shadow_network_multigraph
from DiffusionRWR_model_package.graph_generation.edge_weight_functions import corr_power, negative_correlation_weight, negative_correlation_weight_exponential
from DiffusionRWR_model_package.run_RWR.slow_RWR import simulate_random_walks
from DiffusionRWR_model_package.run_RWR.numba_RWR import simulate_walks_to_rna_FAST
from DiffusionRWR_model_package.model_analysis.PCA_frequency_plot import plot_trajectory_visit_frequencies, plot_shortest_trajectories, plot_ideal_trajectory
from DiffusionRWR_model_package.graph_generation.edge_weight_functions import cor_gaussian_abs_inter, inter_layer_corr_power, intra_layer_corr_gaussian_shifted, cor_gaussian_abs, corr_power, cor_gaussian_shifted, cor_exponential_abs, cor_exponential_abs_inter, cor_exponential_shifted, intra_layer_corr_exponential_shifted
import matplotlib.pyplot as plt

# Configuration parameters
USE_FAST_RWR = True  # Set to False to use slow RWR (more detailed output)
USE_SHADOW_NETWORK = True  # Set to True to use shadow network with negative correlations
fix_transition_prob = True  # Whether to fix transition probabilities between layers (ignored if USE_SHADOW_NETWORK=True)
edge_fn_intra = cor_exponential_abs
edge_fn_inter = cor_exponential_abs_inter
edge_fn_negative = negative_correlation_weight_exponential  # Function for negative correlations
gamma = 0.1  # Probability of negative correlation jumps
alpha = 0.001
n_genes = 60  # Number of top RNA genes to report
start='AK9*17_k20me3'
end='e5'
restart_prob=0.0
n_simulations=10000
n_power=20
sigma=0.05
folder_path = r"C:\Users\inigo\OneDrive\Documents\Fourth_Year\Computations\Dissertation\DiffusionRWR_model_repo\DiffusionRWR_model_package\data\Modelled"


def _call_edge_fn_with_params(edge_fn, *args, sigma_value=None, n_power_value=None):
    kwargs = {}
    try:
        params = inspect.signature(edge_fn).parameters
        if sigma_value is not None and "sigma" in params:
            kwargs["sigma"] = sigma_value
        if n_power_value is not None and "n_power" in params:
            kwargs["n_power"] = n_power_value
    except (TypeError, ValueError):
        pass
    return edge_fn(*args, **kwargs)


def histone_cluster_analysis():
    # Placeholder for histone cluster analysis functions
    print("=" * 80)
    print("DIFFUSION RWR PIPELINE")
    print("=" * 80)
    
    # Step 1: Load data from modelled folder
    print("\n[STEP 1] Loading data...")
    data_dict = load_data(folder_path)
    for name, df in list(data_dict.items()):
        if 'k9me2' in name:
            data_dict.pop(name)
    
    # Clean up dataset names to remove prefixes and suffixes
    cleaned_data_dict = {}
    for key, df in data_dict.items():
        # Extract dataset type: k20me3 or rna_ai (k9me2 already removed)
        if 'k20me3' in key:
            clean_name = 'k20me3'
        elif 'rna' in key:
            clean_name = 'rna_ai'
        else:
            clean_name = key
        cleaned_data_dict[clean_name] = df
    
    print(f"\nCleaned dataset names: {list(cleaned_data_dict.keys())}")

    def edge_fn_intra_param(data_with_basis):
        result = _call_edge_fn_with_params(
            edge_fn_intra,
            data_with_basis,
            sigma_value=sigma,
            n_power_value=n_power,
        )
        if isinstance(result, tuple):
            return result[0]
        return result

    def edge_fn_intra_with_signs(data_with_basis):
        corr_matrix = data_with_basis.T.corr()
        sign_matrix = corr_matrix.apply(np.sign)
        adjacency = edge_fn_intra_param(data_with_basis)
        if isinstance(adjacency, np.ndarray):
            adjacency = pd.DataFrame(adjacency, index=data_with_basis.index, columns=data_with_basis.index)
        return adjacency, sign_matrix

    def edge_fn_inter_param(vec_a, vec_b):
        return _call_edge_fn_with_params(
            edge_fn_inter,
            vec_a,
            vec_b,
            sigma_value=sigma,
            n_power_value=n_power,
        )
    
    # Step 2: Generate intra-layer graphs with sign matrices
    print("\n[STEP 2] Generating intra-layer graphs...")
    if USE_SHADOW_NETWORK:
        # Use selected intra-layer edge function with explicit sign matrix
        intra_layer_graphs, sign_matrices = generate_single_layer_graphs(
            cleaned_data_dict, 
            edge_fn=edge_fn_intra_with_signs,
            start=start,
            end=end,
            return_signs=True
        )
        print(f"Generated {len(intra_layer_graphs)} intra-layer graphs with sign matrices")
    else:
        # Use regular edge function without sign matrices
        intra_layer_graphs = generate_single_layer_graphs(
            cleaned_data_dict, 
            edge_fn=edge_fn_intra_param,
            start=start,
            end=end
        )
    
    print("\n[STEP 3] Creating multi-layer graph...")
    if USE_SHADOW_NETWORK:
        print("\nUsing SHADOW NETWORK with negative correlations (OPTIMIZED)...")
        print(f"Gamma (negative jump probability): {gamma}")
        
        # Create shadow network multi-graph using sign matrices
        multi_graph = create_shadow_network_multigraph(
            std_data_dict=cleaned_data_dict,
            intra_graphs=intra_layer_graphs,
            sign_matrices=sign_matrices,
            edge_fn_inter=edge_fn_inter_param,
            gamma=gamma,
            alpha=alpha,
            start=start,
            end=end
        )
    elif fix_transition_prob:
        print("\nFixing transition probabilities between layers...")
        multi_graph = create_multigraph_with_layer_transitions(
            std_data_dict=cleaned_data_dict,
            intra_layer_graphs=intra_layer_graphs,
            edge_fn_inter=edge_fn_inter_param,
            alpha=alpha,
            start=start,
            end=end
        )
    else:
        # Step 3: Create multi-layer graph
        
        multi_graph = create_multigraph(
            std_data_dict=cleaned_data_dict,
            intra_layer_graphs=intra_layer_graphs,
            edge_fn_inter=edge_fn_inter_param,
            start=start,
            end=end
        )
    
    # Visualize the multi-layer graph to diagnose issues
    print("\n[STEP 3.5] Visualizing graph structure to diagnose connectivity...")
    
    # Check connectivity around start and end nodes
    print(f"\n  Analyzing connectivity of start node '{start}':")
    if start in multi_graph.index:
        start_out_edges = multi_graph.loc[start, :].sum()
        start_out_nonzero = (multi_graph.loc[start, :] > 0).sum()
        start_in_edges = multi_graph.loc[:, start].sum()
        start_in_nonzero = (multi_graph.loc[:, start] > 0).sum()
        print(f"    Outgoing edge weight sum: {start_out_edges:.6f}")
        print(f"    Number of outgoing edges: {start_out_nonzero}")
        print(f"    Incoming edge weight sum: {start_in_edges:.6f}")
        print(f"    Number of incoming edges: {start_in_nonzero}")
        if start_out_nonzero > 0:
            top_targets = multi_graph.loc[start, :].nlargest(5)
            print(f"    Top 5 outgoing connections:")
            for node, weight in top_targets.items():
                if weight > 0:
                    print(f"      -> {node}: {weight:.6f}")
    else:
        print(f"    ⚠ WARNING: Start node '{start}' not found in graph!")

    # Step 4: Run random walks from histone mark to RNA genes
    print("\n[STEP 4] Running random walks to RNA genes...")
    print(f"Start: {start}")
    print(f"Target: Any RNA node")
    print(f"Restart probability: {restart_prob}")
    print(f"Number of simulations: {n_simulations}")
    
    results, n_success, n_restart, n_total, success_list = simulate_walks_to_rna_FAST(
        adj_matrix=multi_graph,
        start=start,
        restart_prob=restart_prob,
        n_sims=n_simulations
    )
    
    # Step 5: Analyze RNA endpoints
    print("\n[STEP 5] Analyzing RNA gene endpoints...")
    
    # First, count how many histone marks correspond to each RNA gene
    # This is needed to normalize visit counts (genes with more histones are more likely to be reached)
    print("  Counting histone marks per RNA gene...")
    rna_to_histone_count = {}
    
    # Get all histone layer names (exclude RNA)
    histone_layers = [name for name in cleaned_data_dict.keys() if 'rna' not in name]
    
    # Get all RNA gene names
    rna_genes = list(cleaned_data_dict['rna_ai'].index)
    
    for rna_gene in rna_genes:
        count = 0
        # Count how many histone genes contain this RNA gene name
        for histone_layer in histone_layers:
            for histone_gene in cleaned_data_dict[histone_layer].index:
                if rna_gene in histone_gene:
                    count += 1
        rna_to_histone_count[rna_gene] = count if count > 0 else 1  # Minimum 1 to avoid division by zero
    
    total_histones = sum(rna_to_histone_count.values())
    genes_with_multiple_histones = sum(1 for c in rna_to_histone_count.values() if c > 1)
    print(f"  Total RNA genes: {len(rna_genes)}")
    print(f"  Total histone associations: {total_histones}")
    print(f"  RNA genes with multiple histone marks: {genes_with_multiple_histones}")
    print(f"  Average histones per RNA gene: {total_histones / len(rna_genes):.2f}")
    
    # Count visits to each RNA endpoint (raw counts)
    rna_endpoint_counts_raw = {}
    for result in success_list:
        end_node = result['end_node']
        # Extract gene name from node (remove layer suffix if present)
        gene_name = end_node.split('_rna')[0] if '_rna' in end_node else end_node
        rna_endpoint_counts_raw[gene_name] = rna_endpoint_counts_raw.get(gene_name, 0) + 1
    
    # Normalize by number of corresponding histone marks
    rna_endpoint_counts_normalized = {}
    for gene, raw_count in rna_endpoint_counts_raw.items():
        histone_count = rna_to_histone_count.get(gene, 1)
        rna_endpoint_counts_normalized[gene] = raw_count / histone_count
    
    # Sort by normalized count
    sorted_rna = sorted(rna_endpoint_counts_normalized.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop {n_genes} RNA genes reached from {start} (normalized by histone count):")
    print("=" * 80)
    for i, (gene, norm_count) in enumerate(sorted_rna[:n_genes], 1):
        raw_count = rna_endpoint_counts_raw[gene]
        histone_count = rna_to_histone_count[gene]  # Use pre-computed value
        percentage = 100 * raw_count / n_success
        print(f"{i:2d}. {gene:30s}: {raw_count:4d} walks / {histone_count} histones = {norm_count:7.2f} ({percentage:5.1f}%)")
    
    print(f"\nTotal unique RNA genes reached: {len(rna_endpoint_counts_raw)}")
    
    # Keep both raw and normalized counts for downstream analysis
    rna_endpoint_counts = rna_endpoint_counts_normalized  # Use normalized for visualization
    
    # Export RNA endpoint analysis to CSV
    print("\n[STEP 5b] Exporting RNA endpoint analysis to CSV...")
    
    # Create DataFrame with RNA endpoint statistics
    export_data = []
    rna_expression_df = cleaned_data_dict['rna_ai']  # Get RNA expression data
    
    for gene, norm_count in sorted_rna[:n_genes]:
        raw_count = rna_endpoint_counts_raw[gene]
        histone_count = rna_to_histone_count[gene]
        percentage = 100 * raw_count / n_success
        
        # Get RNA expression values for this gene across time stages
        if gene in rna_expression_df.index:
            gene_expression = rna_expression_df.loc[gene]
            
            # Sort time stages by expression level (highest to lowest)
            sorted_stages = gene_expression.sort_values(ascending=False)
            
            # Create a row with basic statistics and ordered time stages
            row = {
                'gene_name': gene,
                'walk_count': raw_count,
                'percentage': percentage,
                'histone_count': histone_count,
                'normalized_count': norm_count
            }
            
            # Add sorted time stages and their expression values
            for rank, (stage, expr_value) in enumerate(sorted_stages.items(), 1):
                row[f'stage_rank_{rank}'] = stage
                row[f'stage_rank_{rank}_expression'] = expr_value
            
            export_data.append(row)
    
    # Create DataFrame and export to CSV
    export_df = pd.DataFrame(export_data)
    csv_path = r"C:\Users\inigo\OneDrive\Documents\Fourth_Year\Computations\Dissertation\DiffusionRWR_model_repo\DiffusionRWR_model_package\data\rna_endpoint_analysis.csv"
    export_df.to_csv(csv_path, index=False)
    print(f"  Exported RNA endpoint analysis to: {csv_path}")
    print(f"  Rows: {len(export_df)}, Columns: {len(export_df.columns)}")
    
    # Step 6: Create PCA visualization
    print("\n[STEP 6] Creating PCA visualization...")
    
    # Perform PCA on combined data
    print("  Running PCA on all genes...")
    all_data = pd.concat([cleaned_data_dict[name] for name in cleaned_data_dict.keys()])
    pca = PCA(n_components=3)
    pca.fit(all_data)
    
    print(f"  PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.1%}, "
          f"PC2={pca.explained_variance_ratio_[1]:.1%}, PC3={pca.explained_variance_ratio_[2]:.1%}")
    
    # Create embeddings dictionary for each category
    embeddings_dict = {}
    for dataset_name, data in cleaned_data_dict.items():
        embeddings_dict[dataset_name] = pca.transform(data)
    
    # Convert success_list to trajectory format
    print("\n  Creating trajectories for visualization...")
    successful_trajectories = []
    for result in success_list:
        successful_trajectories.append({
            'path': result['path'],
            'steps': len(result['path']) - 1,
            'unique_nodes': len(set(result['path']))
        })
    
    # Create a custom data_dict with normalized visit counts for RNA genes
    # We'll modify the RNA layer data to scale by normalization factors
    print("  Preparing normalized data for visualization...")
    plot_data_dict = cleaned_data_dict.copy()
    
    # Create custom trajectories that account for normalization
    # by weighting paths inversely by histone count
    normalized_trajectories = []
    for result in success_list:
        end_node = result['end_node']
        gene_name = end_node.split('_rna')[0] if '_rna' in end_node else end_node
        norm_factor = rna_to_histone_count.get(gene_name, 1)
        
        # Replicate this trajectory fractionally (use probability sampling)
        # For simplicity, we'll weight by including the path multiple times or fractionally
        if norm_factor <= 1:
            # Include all
            normalized_trajectories.append({
                'path': result['path'],
                'steps': len(result['path']) - 1,
                'unique_nodes': len(set(result['path']))
            })
        else:
            # Include with probability 1/norm_factor
            import random
            if random.random() < (1.0 / norm_factor):
                normalized_trajectories.append({
                    'path': result['path'],
                    'steps': len(result['path']) - 1,
                    'unique_nodes': len(set(result['path']))
                })
    
    print(f"  Normalized {len(success_list)} trajectories to {len(normalized_trajectories)} weighted trajectories")
    
    # Use the plot_trajectory_visit_frequencies function
    categories = list(cleaned_data_dict.keys())
    fig, node_counts = plot_trajectory_visit_frequencies(
        data_dict=plot_data_dict,
        embeddings_dict=embeddings_dict,
        successful_trajectories=normalized_trajectories,  # Use normalized trajectories
        categories=categories,
        pca_model=pca,
        start=start,
        end=None,  # No specific end node (any RNA is valid)
        title=f'Histone to RNA Walk Visit Frequencies (Normalized by Histone Count)<br>Starting from {start}',
        min_marker_size=2,
        max_marker_size=30
    )
    
    # Add shortest trajectories to the plot
    print("\n  Adding sample trajectories to visualization...")
    # Sort trajectories by length and select a few shortest ones
    sorted_trajectories = sorted(success_list, key=lambda x: len(x['path']))
    n_trajectories_to_plot = min(10, len(sorted_trajectories))  # Plot up to 5 shortest trajectories
    
    fig = plot_shortest_trajectories(
        fig=fig,
        successful_trajectories=sorted_trajectories,  # Pass all sorted trajectories
        data_dict=cleaned_data_dict,
        embeddings_dict=embeddings_dict,
        pca_model=pca,
        basis_coords=None,  # No basis vectors in this analysis
        n_trajectories=n_trajectories_to_plot
    )
    
    # Save to file
    # Sanitize filename (remove invalid characters like *)
    start_name_safe = start.replace('*', '_').replace('/', '_').replace('\\', '_').replace(':', '_')
    output_file = str(Path(__file__).resolve().parent.parent.parent / f"histone_rna_pca_{start_name_safe}.html")
    fig.write_html(output_file)
    print(f"\n✓ PCA visualization saved to: {output_file}")
    
    return results, rna_endpoint_counts, sorted_rna, fig


    