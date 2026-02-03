"""
Main script to run the complete DiffusionRWR pipeline:
1. Load and preprocess data from data/Unmodelled
2. Generate multi-layer graph
3. Run random walk simulations
4. Visualize results with visit frequency plot
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# Import subpackages
from DiffusionRWR_model_package.graph_generation.preprocess_data import load_data
from DiffusionRWR_model_package.graph_generation.generate_graph_internal import generate_single_layer_graphs, generate_negative_correlation_graphs
from DiffusionRWR_model_package.graph_generation.generate_multi_graph import create_multigraph, create_multigraph_with_layer_transitions, create_shadow_network_multigraph
from DiffusionRWR_model_package.graph_generation.edge_weight_functions import corr_power, negative_correlation_weight
from DiffusionRWR_model_package.run_RWR.slow_RWR import simulate_random_walks
from DiffusionRWR_model_package.run_RWR.numba_RWR import simulate_walks_FAST
from DiffusionRWR_model_package.model_analysis.PCA_frequency_plot import plot_trajectory_visit_frequencies, plot_shortest_trajectories, plot_ideal_trajectory
from DiffusionRWR_model_package.graph_generation.edge_weight_functions import inter_layer_corr_power, intra_layer_corr_gaussian_shifted, cor_gaussian_abs, corr_power, cor_gaussian_shifted
import matplotlib.pyplot as plt


# Configuration parameters
USE_FAST_RWR = True  # Set to False to use slow RWR (more detailed output)
USE_SHADOW_NETWORK = True  # Set to True to use shadow network with negative correlations
fix_transition_prob = True  # Whether to fix transition probabilities between layers (ignored if USE_SHADOW_NETWORK=True)
edge_fn_intra = cor_gaussian_shifted
edge_fn_inter = intra_layer_corr_gaussian_shifted
edge_fn_negative = negative_correlation_weight  # Function for negative correlations
gamma = 0.5  # Probability of negative correlation jumps
start='e3'
end='e5'
restart_prob=0.02
n_simulations=1000
folder_path = r"C:\Users\inigo\OneDrive\Documents\Fourth_Year\Computations\Dissertation\DiffusionRWR_model_repo\DiffusionRWR_model_package\data\Modelled"



def start_to_end():
    """Main pipeline execution."""
    
    print("=" * 80)
    print("DIFFUSION RWR PIPELINE")
    print("=" * 80)
    
    # Step 1: Load data from modelled folder
    print("\n[STEP 1] Loading data...")
    data_dict = load_data(folder_path)
    
    # Clean up dataset names to remove prefixes and suffixes
    cleaned_data_dict = {}
    for key, df in data_dict.items():
        # Extract dataset type: k9me2, k20me3, or rna_ai
        if 'k9me2' in key:
            clean_name = 'k9me2'
        elif 'k20me3' in key:
            clean_name = 'k20me3'
        elif 'rna' in key:
            clean_name = 'rna_ai'
        else:
            clean_name = key
        cleaned_data_dict[clean_name] = df
    
    print(f"\nCleaned dataset names: {list(cleaned_data_dict.keys())}")
    
    # Step 2: Generate intra-layer graphs with sign matrices
    print("\n[STEP 2] Generating intra-layer graphs...")
    if USE_SHADOW_NETWORK:
        # Use absolute correlation with sign matrices
        intra_layer_graphs, sign_matrices = generate_single_layer_graphs(
            cleaned_data_dict, 
            edge_fn=cor_gaussian_abs,  # Uses |correlation|
            start=start,
            end=end,
            return_signs=True
        )
        print(f"Generated {len(intra_layer_graphs)} intra-layer graphs with sign matrices")
    else:
        # Use regular edge function without sign matrices
        intra_layer_graphs = generate_single_layer_graphs(
            cleaned_data_dict, 
            edge_fn=edge_fn_intra,
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
            edge_fn_inter=edge_fn_inter,
            gamma=gamma,
            start=start,
            end=end
        )
    elif fix_transition_prob:
        print("\nFixing transition probabilities between layers...")
        multi_graph = create_multigraph_with_layer_transitions(
            std_data_dict=cleaned_data_dict,
            intra_layer_graphs=intra_layer_graphs,
            edge_fn_inter=edge_fn_inter,
            alpha=0.1,
            start=start,
            end=end
        )
    else:
        # Step 3: Create multi-layer graph
        
        multi_graph = create_multigraph(
            std_data_dict=cleaned_data_dict,
            intra_layer_graphs=intra_layer_graphs,
            edge_fn_inter=edge_fn_inter,
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
    
    print(f"\n  Analyzing connectivity of end node '{end}':")
    if end in multi_graph.index:
        end_out_edges = multi_graph.loc[end, :].sum()
        end_out_nonzero = (multi_graph.loc[end, :] > 0).sum()
        end_in_edges = multi_graph.loc[:, end].sum()
        end_in_nonzero = (multi_graph.loc[:, end] > 0).sum()
        print(f"    Outgoing edge weight sum: {end_out_edges:.6f}")
        print(f"    Number of outgoing edges: {end_out_nonzero}")
        print(f"    Incoming edge weight sum: {end_in_edges:.6f}")
        print(f"    Number of incoming edges: {end_in_nonzero}")
        if end_in_nonzero > 0:
            top_sources = multi_graph.loc[:, end].nlargest(5)
            print(f"    Top 5 incoming connections:")
            for node, weight in top_sources.items():
                if weight > 0:
                    print(f"      {node} -> : {weight:.6f}")
    else:
        print(f"    ⚠ WARNING: End node '{end}' not found in graph!")
    
    
    # Step 4: Run random walk simulations
    print("\n[STEP 4] Running random walk simulations...")
    
    if USE_FAST_RWR:
        print("Using FAST RWR (Numba JIT-compiled)")
        results, successful_walks, restarted_walks, total_attempts, successful_results = simulate_walks_FAST(
            adj_matrix=multi_graph,
            start=start,  # Shared basis vector (e.g., 'e3')
            target=end,  # Shared basis vector (e.g., 'e5')
            restart_prob=restart_prob,
            n_sims=n_simulations
        )
    else:
        print("Using SLOW RWR (more detailed output)")
        results, successful_walks, restarted_walks, total_attempts, successful_results = simulate_random_walks(
            adjacency_matrix=multi_graph,
            start_node=start,  # Shared basis vector (e.g., 'e3')
            target_node=end,  # Shared basis vector (e.g., 'e5')
            restart_prob=restart_prob,
            n_simulations=n_simulations
        )
    
    # Step 5: Prepare data for visualization
    print("\n[STEP 5] Preparing data for visualization...")
    
    # Perform PCA on combined data
    combined_data = pd.concat(list(cleaned_data_dict.values()), axis=0)
    pca_model = PCA(n_components=3, random_state=42)
    combined_embedding = pca_model.fit_transform(combined_data)
    
    # Split embeddings by category
    embeddings_dict = {}
    current_idx = 0
    for name, df in cleaned_data_dict.items():
        n_samples = len(df)
        embeddings_dict[name] = combined_embedding[current_idx:current_idx + n_samples]
        current_idx += n_samples
    
    # Compute basis vector embeddings
    n_features = 5  # 5 time points
    standard_basis = np.eye(n_features)
    standard_basis_std = np.zeros_like(standard_basis)
    for i in range(n_features):
        row_mean = standard_basis[i].mean()
        row_std = standard_basis[i].std()
        standard_basis_std[i] = (standard_basis[i] - row_mean) / row_std
    
    basis_centered = standard_basis_std - pca_model.mean_
    basis_pca = basis_centered @ pca_model.components_.T
    
    # Shared basis vectors (not layer-specific)
    basis_coords = {
        'e1': basis_pca[0],
        'e2': basis_pca[1],
        'e3': basis_pca[2],
        'e4': basis_pca[3],
        'e5': basis_pca[4]
    }
    
    # Step 6: Plot results
    print("\n[STEP 6] Plotting visit frequency visualization...")
    if successful_walks > 0:
        fig, node_counts = plot_trajectory_visit_frequencies(
            data_dict=cleaned_data_dict,
            embeddings_dict=embeddings_dict,
            successful_trajectories=successful_results,
            categories=['k9me2', 'k20me3', 'rna_ai'],
            pca_model=pca_model,
            basis_coords=basis_coords,
            start=start,
            end=end,
            title="Random Walk Visit Frequency on Multi-Layer Graph PCA"
        )
        
        # Add shortest trajectories to the plot
        fig = plot_shortest_trajectories(
            fig=fig,
            successful_trajectories=successful_results,
            data_dict=cleaned_data_dict,
            embeddings_dict=embeddings_dict,
            pca_model=pca_model,
            basis_coords=basis_coords,
            n_trajectories=5 # Change this number to plot more or fewer trajectories
        )
        
        # Add ideal developmental trajectory
        fig = plot_ideal_trajectory(
            fig=fig,
            pca_model=pca_model,
            n_points_per_segment=50
        )
        
        # Save to HTML file instead of showing in browser
        output_file = "trajectory_visualization.html"
        fig.write_html(output_file)
        print(f"\n✓ Visualization saved to: {output_file}")
        print("  Open this file in your browser to view the plot.")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    else:
        print("\n⚠ No successful walks found. Cannot generate visualization.")
        print("Consider adjusting parameters or running more simulations.")


