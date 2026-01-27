import plotly.graph_objects as go
import numpy as np
import pandas as pd


def plot_trajectory_visit_frequencies(
    data_dict,
    embeddings_dict,
    successful_trajectories,
    categories,
    pca_model,
    basis_coords=None,
    title="Random Walk Visit Frequency Visualization",
    min_marker_size=2,
    max_marker_size=30
):
    """
    Plot visit frequencies of nodes across successful trajectories, colored by data category.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary mapping category names to their dataframes (for getting gene names).
        Example: {'k9me2': k9me2_std, 'k20me3': k20me3_std, 'rna': rna_std}
    
    embeddings_dict : dict
        Dictionary mapping category names to their PCA embeddings (nx3 arrays).
        Example: {'k9me2': k9me2_embedding, 'k20me3': k20me3_embedding, 'rna': rna_embedding}
    
    successful_trajectories : list
        List of successful trajectory dictionaries, each containing a 'path' key with node names.
        
    categories : list
        List of category names (prefixes) to identify in node names.
        Example: ['k9me2', 'k20me3', 'rna_ai']
        
    pca_model : PCA object
        Fitted PCA model used for the embeddings (for axis labels).
        
    basis_coords : dict, optional
        Dictionary mapping basis vector names to their PCA coordinates for plotting reference points.
        
    title : str
        Title for the plot.
        
    min_marker_size : float
        Minimum marker size for unvisited/rarely visited nodes.
        
    max_marker_size : float
        Maximum marker size for most visited nodes.
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The generated figure object.
    node_counts : pd.Series
        Visit counts for all nodes.
    """
    print(f"=== Creating Visit Frequency Visualization ===\n")
    
    # Calculate visit counts for all nodes across ALL successful walks
    all_nodes_visited = []
    for traj in successful_trajectories:
        all_nodes_visited.extend(traj['path'])
    
    node_counts = pd.Series(all_nodes_visited).value_counts()
    
    print(f"Total node visits across {len(successful_trajectories)} walks: {len(all_nodes_visited)}")
    print(f"Unique nodes visited: {len(node_counts)}")
    
    # Define color palette for categories
    category_colors = [
        '#1f77b4',  # blue
        '#ff7f0e',  # orange
        '#2ca02c',  # green
        '#d62728',  # red
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#7f7f7f',  # gray
        '#bcbd22',  # olive
        '#17becf'   # cyan
    ]
    
    # Create the figure
    fig = go.Figure()
    
    # Process each category
    for idx, (cat_name, cat_data) in enumerate(data_dict.items()):
        cat_embedding = embeddings_dict[cat_name]
        gene_names = list(cat_data.index)
        
        # Get visit counts for genes in this category
        # Node names in the graph have the format: gene_layer (e.g., "GENE123_rna_ai")
        visit_counts = []
        for gene in gene_names:
            # Append layer suffix to match node names in the graph
            node_name = f"{gene}_{cat_name}"
            if node_name in node_counts.index:
                visit_counts.append(node_counts[node_name])
            else:
                visit_counts.append(0)
        
        visit_counts = np.array(visit_counts)
        
        print(f"\n{cat_name.upper()} statistics:")
        print(f"  Genes visited at least once: {np.sum(visit_counts > 0)}/{len(gene_names)}")
        if np.sum(visit_counts > 0) > 0:
            print(f"  Max visits to a single gene: {visit_counts.max()}")
            print(f"  Mean visits per visited gene: {visit_counts[visit_counts > 0].mean():.1f}")
            print(f"  Median visits per visited gene: {np.median(visit_counts[visit_counts > 0]):.1f}")
        
        # Scale visit counts to marker sizes
        visit_counts_scaled = visit_counts.copy().astype(float)
        visit_counts_scaled[visit_counts_scaled == 0] = 0.1  # Small size for unvisited
        
        # Apply sqrt scaling for better visual distribution
        sizes_scaled = visit_counts_scaled
        sizes_scaled = min_marker_size + (max_marker_size - min_marker_size) * \
                      (sizes_scaled - sizes_scaled.min()) / (sizes_scaled.max() - sizes_scaled.min() + 1e-10)
        
        # Get color for this category
        cat_color = category_colors[idx % len(category_colors)]
        
        # Add trace for this category
        fig.add_trace(go.Scatter3d(
            x=cat_embedding[:, 0],
            y=cat_embedding[:, 1],
            z=cat_embedding[:, 2],
            mode='markers',
            name=cat_name,
            marker=dict(
                size=sizes_scaled,
                color=cat_color,
                line=dict(color='black', width=0.3)
            ),
            text=[f"{gene}<br>Category: {cat_name}<br>Visits: {count}" 
                  for gene, count in zip(gene_names, visit_counts)],
            hovertemplate='%{text}<extra></extra>',
            showlegend=True
        ))
    
    # Add basis vectors if provided
    if basis_coords is not None:
        basis_colors_plot = ['orange', 'cyan', 'red', 'purple', 'green']
        basis_names_plot = ['e1', 'e2', 'e3 (Start/Restart)', 'e4', 'e5 (Target)']
        
        for i, (basis_key, coords) in enumerate(basis_coords.items()):
            # Get visit count for this basis vector
            basis_visits = node_counts[basis_key] if basis_key in node_counts.index else 0
            
            fig.add_trace(go.Scatter3d(
                x=[coords[0]],
                y=[coords[1]],
                z=[coords[2]],
                mode='markers+text',
                name=basis_names_plot[i],
                marker=dict(
                    size=20,
                    color=basis_colors_plot[i],
                    symbol='diamond',
                    line=dict(color='black', width=2)
                ),
                text=[f"{basis_key}<br>({basis_visits} visits)"],
                textposition='top center',
                textfont=dict(size=12, color=basis_colors_plot[i]),
                hovertemplate=f'{basis_key}<br>Visits: {basis_visits}<extra></extra>',
                showlegend=True
            ))
    
    # Update layout
    variance_ratios = pca_model.explained_variance_ratio_
    fig.update_layout(
        title=f'{title}<br>(Marker size ∝ visit count, {len(successful_trajectories)} successful walks)',
        scene=dict(
            xaxis_title=f'PC1 ({variance_ratios[0]:.2%} variance)',
            yaxis_title=f'PC2 ({variance_ratios[1]:.2%} variance)',
            zaxis_title=f'PC3 ({variance_ratios[2]:.2%} variance)',
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray'),
            zaxis=dict(showgrid=True, gridcolor='lightgray'),
        ),
        width=1000,
        height=800,
        showlegend=True,
        legend=dict(x=0.02, y=0.98)
    )
    
    print(f"\n✓ Visualized gene visit frequencies with {len(data_dict)} data categories")
    
    # Show top visited genes across all categories
    print(f"\n=== Top 10 Most Visited Genes (All Categories) ===")
    all_gene_names = []
    for cat_data in data_dict.values():
        all_gene_names.extend(cat_data.index)
    
    top_genes = node_counts[node_counts.index.isin(all_gene_names)].head(10)
    for gene, count in top_genes.items():
        percentage = 100 * count / len(all_nodes_visited)
        # Identify category from gene name
        gene_cat = "unknown"
        for cat in categories:
            if gene.startswith(cat):
                gene_cat = cat
                break
        print(f"  {gene} [{gene_cat}]: {count} visits ({percentage:.2f}%)")
    
    return fig, node_counts


def plot_shortest_trajectories(
    fig,
    successful_trajectories,
    data_dict,
    embeddings_dict,
    pca_model,
    basis_coords,
    n_trajectories=5
):
    """
    Add N randomly sampled trajectories to an existing 3D PCA plot.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Existing figure to add trajectories to.
    
    successful_trajectories : list
        List of successful trajectory dictionaries with 'path' and 'steps' keys.
    
    data_dict : dict
        Dictionary mapping category names to their dataframes.
    
    embeddings_dict : dict
        Dictionary mapping category names to their PCA embeddings.
    
    pca_model : PCA object
        Fitted PCA model used for the embeddings.
    
    basis_coords : dict
        Dictionary mapping basis vector names to their PCA coordinates.
    
    n_trajectories : int
        Number of trajectories to randomly sample and plot (default: 5).
    
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Updated figure with trajectories.
    """
    print(f"\n=== Adding {n_trajectories} Randomly Sampled Trajectories ===")
    
    # Randomly sample trajectories
    import random
    n_to_sample = min(n_trajectories, len(successful_trajectories))
    sampled_trajectories = random.sample(successful_trajectories, n_to_sample)
    
    # Create a mapping from node names to PCA coordinates
    node_to_coords = {}
    
    # Add gene coordinates
    for cat_name, cat_embedding in embeddings_dict.items():
        gene_names = list(data_dict[cat_name].index)
        for gene, coords in zip(gene_names, cat_embedding):
            node_name = f"{gene}_{cat_name}"
            node_to_coords[node_name] = coords
    
    # Add basis vector coordinates
    if basis_coords is not None:
        for basis_name, coords in basis_coords.items():
            node_to_coords[basis_name] = coords
    
    # Plot each trajectory
    trajectory_colors = [
        'red', 'yellow', 'magenta', 'cyan', 'lime',
        'orange', 'pink', 'purple', 'gold', 'turquoise',
        'coral', 'salmon', 'violet', 'chartreuse', 'crimson',
        'deeppink', 'dodgerblue', 'fuchsia', 'hotpink', 'indigo'
    ]
    
    for i, traj in enumerate(sampled_trajectories):
        path = traj['path']
        steps = traj['steps']
        
        # Get coordinates for each node in the path
        x_coords = []
        y_coords = []
        z_coords = []
        
        for node in path:
            if node in node_to_coords:
                coords = node_to_coords[node]
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                z_coords.append(coords[2])
            else:
                print(f"  Warning: Node '{node}' not found in coordinates")
        
        # Add trajectory as a line
        color = trajectory_colors[i % len(trajectory_colors)]
        fig.add_trace(go.Scatter3d(
            x=x_coords,
            y=y_coords,
            z=z_coords,
            mode='lines+markers',
            name=f'Trajectory {i+1} ({steps} steps)',
            line=dict(
                color=color,
                width=4
            ),
            marker=dict(
                size=4,
                color=color,
                opacity=0.8
            ),
            hovertext=[f"Step {j}: {node}" for j, node in enumerate(path[:len(x_coords)])],
            hovertemplate='%{hovertext}<extra></extra>',
            showlegend=True
        ))
        
        print(f"  Trajectory {i+1}: {steps} steps, {traj['unique_nodes']} unique nodes")
    
    print(f"✓ Added {len(sampled_trajectories)} randomly sampled trajectories to plot")
    
    return fig
