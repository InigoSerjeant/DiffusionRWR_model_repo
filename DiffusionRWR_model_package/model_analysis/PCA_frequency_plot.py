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
    start=None,
    end=None,
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
    
    start : str, optional
        Starting node name for the random walks.
    
    end : str, optional
        Ending node name for the random walks.
        
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
    
    # Check if start/end are basis vectors
    basis_vector_names = ['e1', 'e2', 'e3', 'e4', 'e5']
    start_is_basis = start in basis_vector_names if start else True
    end_is_basis = end in basis_vector_names if end else True
    
    # Calculate visit counts for all nodes across ALL successful walks
    all_nodes_visited = []
    for traj in successful_trajectories:
        all_nodes_visited.extend(traj['path'])
    
    node_counts = pd.Series(all_nodes_visited).value_counts()
    
    # Separate shadow nodes from regular nodes
    shadow_node_counts = node_counts[node_counts.index.str.contains('_neg')]
    regular_node_counts = node_counts[~node_counts.index.str.contains('_neg')]
    
    print(f"Total node visits across {len(successful_trajectories)} walks: {len(all_nodes_visited)}")
    print(f"Unique nodes visited: {len(node_counts)}")
    print(f"  Regular nodes: {len(regular_node_counts)}")
    print(f"  Shadow nodes: {len(shadow_node_counts)}")
    if len(shadow_node_counts) > 0:
        print(f"  Shadow space visits: {shadow_node_counts.sum()} ({100*shadow_node_counts.sum()/len(all_nodes_visited):.1f}%)")
    
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
    
    # First pass: collect all visit counts to get global min/max
    all_visit_counts_list = []
    category_visit_data = {}
    start_end_nodes = []  # Track start/end nodes if they're not basis vectors
    
    for cat_name, cat_data in data_dict.items():
        gene_names = list(cat_data.index)
        visit_counts = []
        for gene in gene_names:
            # Count visits to both regular and shadow versions
            node_name = f"{gene}_{cat_name}"
            shadow_node_name = f"{gene}_{cat_name}_neg"
            
            regular_visits = node_counts[node_name] if node_name in node_counts.index else 0
            shadow_visits = node_counts[shadow_node_name] if shadow_node_name in node_counts.index else 0
            
            # Combine visits from both spaces for visualization
            total_visits = regular_visits + shadow_visits
            visit_counts.append(total_visits)
        
        visit_counts = np.array(visit_counts)
        category_visit_data[cat_name] = (gene_names, visit_counts)
        
        # Exclude start/end nodes from scaling if they're not basis vectors
        for i, gene in enumerate(gene_names):
            node_name = f"{gene}_{cat_name}"
            if visit_counts[i] > 0:
                # Check if this node is start or end
                is_start_or_end = False
                if not start_is_basis and start and node_name == start:
                    is_start_or_end = True
                    start_end_nodes.append((cat_name, i, 'start', visit_counts[i]))
                if not end_is_basis and end and node_name == end:
                    is_start_or_end = True
                    start_end_nodes.append((cat_name, i, 'end', visit_counts[i]))
                
                if not is_start_or_end:
                    all_visit_counts_list.append(visit_counts[i])
    
    # Calculate global statistics for scaling (excluding start/end if they're genes)
    all_visit_counts_array = np.array(all_visit_counts_list)
    global_min = 0.1  # For unvisited nodes
    global_max = all_visit_counts_array.max() if len(all_visit_counts_array) > 0 else 1
    
    print(f"\nGlobal visit statistics across all categories:")
    print(f"  Min visits: 0")
    print(f"  Max visits (excluding start/end genes): {global_max}")
    if start_end_nodes:
        print(f"  Start/end genes excluded from scaling: {len(start_end_nodes)}")
    
    # Process each category with global scaling
    for idx, (cat_name, cat_data) in enumerate(data_dict.items()):
        cat_embedding = embeddings_dict[cat_name]
        gene_names, visit_counts = category_visit_data[cat_name]
        
        print(f"\n{cat_name.upper()} statistics:")
        print(f"  Genes visited at least once: {np.sum(visit_counts > 0)}/{len(gene_names)}")
        if np.sum(visit_counts > 0) > 0:
            print(f"  Max visits to a single gene: {visit_counts.max()}")
            print(f"  Mean visits per visited gene: {visit_counts[visit_counts > 0].mean():.1f}")
            print(f"  Median visits per visited gene: {np.median(visit_counts[visit_counts > 0]):.1f}")
        
        # Scale visit counts to marker sizes using GLOBAL min/max
        visit_counts_scaled = visit_counts.copy().astype(float)
        visit_counts_scaled[visit_counts_scaled == 0] = global_min  # Small size for unvisited
        
        # Apply scaling using global range
        sizes_scaled = min_marker_size + (max_marker_size - min_marker_size) * \
                      (visit_counts_scaled - global_min) / (global_max - global_min + 1e-10)
        
        # Get color for this category
        cat_color = category_colors[idx % len(category_colors)]
        
        # Create masks for regular nodes vs start/end nodes
        regular_mask = np.ones(len(gene_names), dtype=bool)
        for cat, gene_idx, role, count in start_end_nodes:
            if cat == cat_name:
                regular_mask[gene_idx] = False
        
        # Add trace for regular nodes in this category
        if np.any(regular_mask):
            fig.add_trace(go.Scatter3d(
                x=cat_embedding[regular_mask, 0],
                y=cat_embedding[regular_mask, 1],
                z=cat_embedding[regular_mask, 2],
                mode='markers',
                name=cat_name,
                marker=dict(
                    size=sizes_scaled[regular_mask],
                    color=cat_color,
                    line=dict(color='black', width=0.3)
                ),
                text=[f"{gene}<br>Category: {cat_name}<br>Visits: {count}" 
                      for gene, count, is_regular in zip(gene_names, visit_counts, regular_mask) if is_regular],
                hovertemplate='%{text}<extra></extra>',
                showlegend=True
            ))
        
        # Add separate traces for start/end nodes with star markers
        for cat, gene_idx, role, count in start_end_nodes:
            if cat == cat_name:
                gene = gene_names[gene_idx]
                fig.add_trace(go.Scatter3d(
                    x=[cat_embedding[gene_idx, 0]],
                    y=[cat_embedding[gene_idx, 1]],
                    z=[cat_embedding[gene_idx, 2]],
                    mode='markers',
                    name=f"{gene} ({role})",
                    marker=dict(
                        size=20,  # Larger than max
                        color=cat_color,
                        symbol='diamond',
                        line=dict(color='gold', width=3)
                    ),
                    text=f"{gene}<br>Category: {cat_name}<br>Role: {role}<br>Visits: {count}",
                    hovertemplate='%{text}<extra></extra>',
                    showlegend=True
                ))
    
    # Add basis vectors if provided
    if basis_coords is not None:
        basis_colors_plot = ['orange', 'cyan', 'red', 'purple', 'green']
        basis_names_plot = ['e1', 'e2', 'e3', 'e4', 'e5']
        
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
    
    # Add gene coordinates (both regular and shadow nodes map to same coordinates)
    for cat_name, cat_embedding in embeddings_dict.items():
        gene_names = list(data_dict[cat_name].index)
        for gene, coords in zip(gene_names, cat_embedding):
            # Regular node
            node_name = f"{gene}_{cat_name}"
            node_to_coords[node_name] = coords
            
            # Shadow node maps to same PCA coordinates (same gene, different layer)
            shadow_node_name = f"{gene}_{cat_name}_neg"
            node_to_coords[shadow_node_name] = coords
    
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
        node_labels = []
        is_shadow = []
        
        for node in path:
            if node in node_to_coords:
                coords = node_to_coords[node]
                x_coords.append(coords[0])
                y_coords.append(coords[1])
                z_coords.append(coords[2])
                node_labels.append(node)
                # Mark if this is a shadow node
                is_shadow.append('_neg' in node)
            else:
                print(f"  Warning: Node '{node}' not found in coordinates")
        
        # Add trajectory as a line
        color = trajectory_colors[i % len(trajectory_colors)]
        
        # Create hover text with shadow indication
        hover_text = []
        for j, (node, shadow) in enumerate(zip(node_labels, is_shadow)):
            space_label = "SHADOW SPACE" if shadow else "Regular space"
            hover_text.append(f"Step {j}: {node}<br>{space_label}")
        
        # Identify transitions between regular and shadow space
        # We need to create separate traces for different segment types
        segments = []  # List of (start_idx, end_idx, is_transition)
        
        current_segment_start = 0
        for j in range(len(is_shadow) - 1):
            # Check if there's a transition between consecutive nodes
            is_transition = is_shadow[j] != is_shadow[j + 1]
            
            # Check if we need to start a new segment
            if j == 0:
                current_is_transition = is_transition
            elif is_transition != current_is_transition:
                # End current segment and start new one
                segments.append((current_segment_start, j + 1, current_is_transition))
                current_segment_start = j
                current_is_transition = is_transition
        
        # Add the final segment
        segments.append((current_segment_start, len(is_shadow), current_is_transition if len(is_shadow) > 1 else False))
        
        # Plot each segment with appropriate line style
        for seg_idx, (start_idx, end_idx, is_transition) in enumerate(segments):
            seg_x = x_coords[start_idx:end_idx]
            seg_y = y_coords[start_idx:end_idx]
            seg_z = z_coords[start_idx:end_idx]
            seg_hover = hover_text[start_idx:end_idx]
            seg_shadow = is_shadow[start_idx:end_idx]
            
            # Determine line style
            dash_style = 'dot' if is_transition else 'solid'
            
            # Only show legend for first segment
            show_legend = (seg_idx == 0)
            legend_name = f'Trajectory {i+1} ({steps} steps)' if show_legend else None
            
            fig.add_trace(go.Scatter3d(
                x=seg_x,
                y=seg_y,
                z=seg_z,
                mode='lines+markers',
                name=legend_name,
                line=dict(
                    color=color,
                    width=4,
                    dash=dash_style
                ),
                marker=dict(
                    size=4,
                    color=color,
                    opacity=0.8,
                    symbol=['diamond' if s else 'circle' for s in seg_shadow]
                ),
                hovertext=seg_hover,
                hovertemplate='%{hovertext}<extra></extra>',
                showlegend=show_legend,
                legendgroup=f'traj_{i}'  # Group all segments of same trajectory
            ))
        
        print(f"  Trajectory {i+1}: {steps} steps, {traj['unique_nodes']} unique nodes")
    
    print(f"✓ Added {len(sampled_trajectories)} randomly sampled trajectories to plot")
    
    return fig


def plot_ideal_trajectory(fig, pca_model, n_points_per_segment=50):
    """
    Add the ideal piecewise linear trajectory from e1 → e2 → e3 → e4 → e5 to the plot.
    
    This trajectory represents the "ground truth" path through the developmental stages,
    moving linearly through the convex space between consecutive basis vectors.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        Existing figure to add the ideal trajectory to
    pca_model : PCA object
        Fitted PCA model to transform points into PCA space
    n_points_per_segment : int
        Number of interpolation points between each pair of basis vectors
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        Updated figure with ideal trajectory
    """
    print("\n=== Adding Ideal Trajectory to Plot ===")
    
    # Define the 5 basis vectors (identity matrix)
    n_features = 5
    basis_vectors = np.eye(n_features)
    
    # Standardize each basis vector (row-wise standardization)
    standardized_basis = np.zeros_like(basis_vectors)
    for i in range(n_features):
        row_mean = basis_vectors[i].mean()
        row_std = basis_vectors[i].std()
        if row_std == 0:
            standardized_basis[i] = basis_vectors[i] - row_mean
        else:
            standardized_basis[i] = (basis_vectors[i] - row_mean) / row_std
    
    print(f"Standardized basis vectors:")
    for i, vec in enumerate(standardized_basis):
        print(f"  e{i+1}: {vec}")
    
    # Generate piecewise linear path: e1 → e2 → e3 → e4 → e5
    trajectory_points_unstd = []
    
    for i in range(n_features - 1):
        # Interpolate between e_i and e_{i+1} in ORIGINAL space (before standardization)
        start_vec = basis_vectors[i]
        end_vec = basis_vectors[i + 1]
        
        # Create convex combination: (1-t)*start + t*end for t in [0, 1]
        for t in np.linspace(0, 1, n_points_per_segment, endpoint=(i == n_features - 2)):
            interpolated = (1 - t) * start_vec + t * end_vec
            trajectory_points_unstd.append(interpolated)
    
    # Convert to array
    trajectory_points_unstd = np.array(trajectory_points_unstd)
    print(f"\nGenerated {len(trajectory_points_unstd)} interpolated points in convex space")
    
    # Standardize each interpolated point (row-wise)
    trajectory_points = np.zeros_like(trajectory_points_unstd)
    for i in range(len(trajectory_points_unstd)):
        row_mean = trajectory_points_unstd[i].mean()
        row_std = trajectory_points_unstd[i].std()
        if row_std == 0:
            trajectory_points[i] = trajectory_points_unstd[i] - row_mean
        else:
            trajectory_points[i] = (trajectory_points_unstd[i] - row_mean) / row_std
    
    print(f"Standardized all {len(trajectory_points)} interpolated points")
    
    # Transform into PCA space
    trajectory_pca = pca_model.transform(trajectory_points)
    
    print(f"Transformed to PCA space: shape {trajectory_pca.shape}")
    print(f"  PC1 range: [{trajectory_pca[:, 0].min():.3f}, {trajectory_pca[:, 0].max():.3f}]")
    print(f"  PC2 range: [{trajectory_pca[:, 1].min():.3f}, {trajectory_pca[:, 1].max():.3f}]")
    print(f"  PC3 range: [{trajectory_pca[:, 2].min():.3f}, {trajectory_pca[:, 2].max():.3f}]")
    
    # Add to plot as a thick golden line
    fig.add_trace(go.Scatter3d(
        x=trajectory_pca[:, 0],
        y=trajectory_pca[:, 1],
        z=trajectory_pca[:, 2],
        mode='lines',
        name='Ideal Trajectory (e1→e2→e3→e4→e5)',
        line=dict(
            color='gold',
            width=8,
            dash='solid'
        ),
        hovertemplate='Ideal Developmental Path<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>',
        showlegend=True,
        legendrank=1  # Show at top of legend
    ))
    
    # Add markers at the basis vector positions
    basis_pca = pca_model.transform(standardized_basis)
    basis_labels = ['e1 (Start)', 'e2', 'e3', 'e4', 'e5 (End)']
    
    fig.add_trace(go.Scatter3d(
        x=basis_pca[:, 0],
        y=basis_pca[:, 1],
        z=basis_pca[:, 2],
        mode='markers+text',
        name='Developmental Stages',
        marker=dict(
            size=12,
            color='gold',
            symbol='diamond',
            line=dict(color='black', width=2)
        ),
        text=basis_labels,
        textposition='top center',
        textfont=dict(size=12, color='gold', family='Arial Black'),
        hovertemplate='%{text}<br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>',
        showlegend=True,
        legendrank=2
    ))
    
    print("✓ Added ideal trajectory and stage markers to plot")
    
    return fig
