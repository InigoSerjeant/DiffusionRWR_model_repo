import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def preprocess_adjacency_to_sparse(adjacency_matrix, integer=False):
    """
    Convert dense adjacency matrix to sparse representation for faster random walks.
    Returns a dictionary mapping each node to (neighbors, weights) tuples.
    """
    print("Pre-processing adjacency matrix to sparse format...")
    sparse_graph = {}
    
    for node in adjacency_matrix.index:
        # Get all non-zero edges from this node
        weights = adjacency_matrix.loc[node, :]
        nonzero_mask = weights > 0.1
        
        if nonzero_mask.any():
            neighbors = weights[nonzero_mask].index.tolist()
            if integer:
                edge_weights = np.array([1 for _ in weights[nonzero_mask].values])
            else:
                edge_weights = weights[nonzero_mask].values
            sparse_graph[node] = (neighbors, edge_weights)
        else:
            # Node has no outgoing edges
            sparse_graph[node] = ([], np.array([]))
    
    total_edges = sum(len(neighbors) for neighbors, _ in sparse_graph.values())
    print(f"  ✓ Sparse graph created: {len(sparse_graph)} nodes, {total_edges} edges")
    return sparse_graph

def random_walk_with_restart(adjacency_matrix, start_node, target_node, restart_prob=0.0001, max_steps=100000, sparse_graph=None):
    """
    Perform a random walk with restart on the graph defined by adjacency_matrix.
    
    Parameters:
    - adjacency_matrix: DataFrame with edge weights (only used if sparse_graph is None)
    - start_node: node to start and restart from (e.g., 'e3')
    - target_node: node to terminate at (e.g., 'e5')
    - restart_prob: probability of restarting at each step
    - max_steps: maximum number of steps before giving up
    - sparse_graph: pre-processed sparse representation (dict) for faster walks
    
    Returns:
    - path: list of nodes visited
    - steps: number of steps taken
    - success: whether target was reached
    - restarted: whether a restart was triggered (ends this simulation)
    """
    # If sparse_graph not provided, use dense matrix (slower)
    use_sparse = sparse_graph is not None
    
    current_node = start_node
    path = [current_node]
    
    for step in range(max_steps):
        # Check if we've reached the target
        if current_node == target_node:
            return path, step + 1, True, False
        
        # With probability restart_prob, restart (this ends the simulation)
        if np.random.random() < restart_prob:
            return path, step + 1, False, True
        
        # Get neighbors and weights
        if use_sparse:
            # Fast sparse lookup
            neighbors, weights = sparse_graph[current_node]
            
            if len(neighbors) == 0:
                # If stuck (no outgoing edges), this simulation ends
                return path, step + 1, False, True
            
            # Normalize to get transition probabilities
            total_weight = weights.sum()
            transition_probs = weights / total_weight
            
            # Sample next node
            next_node = np.random.choice(neighbors, p=transition_probs)
        else:
            # Slower dense matrix lookup (original method)
            weights = adjacency_matrix.loc[current_node, :]
            
            # Normalize to get transition probabilities
            total_weight = weights.sum()
            if total_weight == 0:
                # If stuck (no outgoing edges), this simulation ends
                return path, step + 1, False, True
            
            transition_probs = weights / total_weight
            
            # Sample next node
            next_node = np.random.choice(adjacency_matrix.columns, p=transition_probs.values)
        
        current_node = next_node
        path.append(current_node)
    
    # Max steps reached without finding target
    return path, max_steps, False, False

def simulate_random_walks(adjacency_matrix, start_node, target_node, restart_prob=0.0001, n_simulations=100, integer_edges = False):
    """
    Simulate multiple random walks with restart and collect statistics.
    
    Parameters:
    - adjacency_matrix: DataFrame with edge weights
    - start_node: node to start and restart from
    - target_node: node to terminate at
    - restart_prob: probability of restarting at each step
    - n_simulations: number of successful simulations to run
    
    Returns:
    - results: list of dictionaries with simulation results
    """
    # Pre-process adjacency matrix to sparse format for efficiency
    sparse_graph = preprocess_adjacency_to_sparse(adjacency_matrix)
    
    results = []
    successful_walks = 0
    restarted_walks = 0
    total_attempts = 0
    
    print("Running simulations...")
    simulation_count = 0
    while simulation_count < n_simulations:
        total_attempts += 1
        
        if total_attempts % 20 == 0:
            print(f"  Completed {simulation_count} successful simulations ({total_attempts} total attempts)...")
        
        path, steps, success, restarted = random_walk_with_restart(
            adjacency_matrix, start_node, target_node, restart_prob, sparse_graph=sparse_graph
        )
        
        if restarted:
            restarted_walks += 1
            continue  # Don't count this as a completed simulation, try again
        
        # Only count as a completed simulation if not restarted
        simulation_count += 1
        
        results.append({
            'simulation': simulation_count,
            'path': path,
            'steps': steps,
            'success': success,
            'unique_nodes': len(set(path))
        })
        
        if success:
            successful_walks += 1
        # Analyze results
    print(f"\n✓ Completed {n_simulations} simulations")
    print(f"  Total attempts (including restarts): {total_attempts}")
    print(f"  Restarts triggered: {restarted_walks}")
    print(f"\n=== Results Summary ===")
    print(f"Successful walks (reached {target_node}): {successful_walks}/{n_simulations} ({100*successful_walks/n_simulations:.1f}%)")

    if successful_walks > 0:
        successful_results = [r for r in results if r['success']]
        steps_list = [r['steps'] for r in successful_results]
        unique_nodes_list = [r['unique_nodes'] for r in successful_results]
    
        print(f"\nFor successful walks:")
        print(f"  Steps to reach target:")
        print(f"    Min: {min(steps_list)}")
        print(f"    Max: {max(steps_list)}")
        print(f"    Mean: {np.mean(steps_list):.1f}")
        print(f"    Median: {np.median(steps_list):.1f}")
        print(f"    Std: {np.std(steps_list):.1f}")
    
        print(f"\n  Unique nodes visited:")
        print(f"    Min: {min(unique_nodes_list)}")
        print(f"    Max: {max(unique_nodes_list)}") 
        print(f"    Mean: {np.mean(unique_nodes_list):.1f}")
        print(f"    Median: {np.median(unique_nodes_list):.1f}")
    
        # Analyze node visitation frequency across all successful walks
        all_nodes_visited = []
        for r in successful_results:
            all_nodes_visited.extend(r['path'])
    
        node_counts = pd.Series(all_nodes_visited).value_counts()
    
        print(f"\n=== Most frequently visited nodes (across all successful walks) ===")
        print("Top 15:")
        for node, count in node_counts.head(15).items():
            percentage = 100 * count / sum(node_counts)
            print(f"  {node}: {count} visits ({percentage:.2f}%)")
    
    # Show basis vector visitation
        print(f"\n=== Basis vector visitation ===")
        for basis in ['e1', 'e2', 'e3', 'e4', 'e5']:
            if basis in node_counts.index:
                count = node_counts[basis]
                percentage = 100 * count / sum(node_counts)
                print(f"  {basis}: {count} visits ({percentage:.2f}%)")
    
        # Plot distribution of steps
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
        # Histogram of steps
        axes[0].hist(steps_list, bins=30, alpha=0.7, edgecolor='black')
        axes[0].axvline(np.mean(steps_list), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(steps_list):.1f}')
        axes[0].axvline(np.median(steps_list), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(steps_list):.1f}')
        axes[0].set_xlabel('Steps to reach target', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title(f'Distribution of Steps to Reach {target_node}\n({successful_walks} successful walks)', 
                         fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    
        # Bar plot of top visited nodes
        top_nodes = node_counts.head(20)
        axes[1].barh(range(len(top_nodes)), top_nodes.values, alpha=0.7, edgecolor='black')
        axes[1].set_yticks(range(len(top_nodes)))
        axes[1].set_yticklabels(top_nodes.index, fontsize=10)
        axes[1].set_xlabel('Number of visits', fontsize=12)
        axes[1].set_title('Top 20 Most Visited Nodes\n(across all successful walks)', 
                     fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='x')
        axes[1].invert_yaxis()
    
        plt.tight_layout()
        plt.show()
    
    # Show a few example paths (shortest, median, longest)
        sorted_results = sorted(successful_results, key=lambda x: x['steps'])
        shortest = sorted_results[0]
        median_idx = len(sorted_results) // 2
        median_walk = sorted_results[median_idx]
        longest = sorted_results[-1]
    
        print(f"\n=== Example Walks ===")
        print(f"\nShortest walk ({shortest['steps']} steps):")
        path_preview = shortest['path'][:20]
        print(f"  Path: {' → '.join(path_preview)}")
        if len(shortest['path']) > 20:
            print(f"  ... ({len(shortest['path']) - 20} more steps)")
    
        print(f"\nMedian walk ({median_walk['steps']} steps):")
        path_preview = median_walk['path'][:20]
        print(f"  Path: {' → '.join(path_preview)}")
        if len(median_walk['path']) > 20:
            print(f"  ... ({len(median_walk['path']) - 20} more steps)")
    
        print(f"\nLongest walk ({longest['steps']} steps):")
        path_preview = longest['path'][:20]
        print(f"  Path: {' → '.join(path_preview)}")
        if len(longest['path']) > 20:
            print(f"  ... ({len(longest['path']) - 20} more steps)")
    else:
        print("\nNo successful walks found. The target node may be unreachable or")
        print("the restart probability may be too high. Consider running more simulations")
        print("or adjusting parameters.")


    # Store results for further analysis

    rwr_results = results


    
    
    return results, successful_walks, restarted_walks, total_attempts, successful_results