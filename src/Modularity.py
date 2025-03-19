import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from kneed import KneeLocator
from collections import defaultdict
import math
import random
from tqdm import tqdm

# Try to import community detection libraries
try:
    import community as community_louvain
except ImportError:
    print("WARNING: python-louvain package not found. Installing...")
    import subprocess
    subprocess.check_call(["pip", "install", "python-louvain"])
    import community as community_louvain

def process_data(filepath):
    """Load and process the CSV data"""
    df = pd.read_csv(filepath)
    return df

def get_threshold(scores, topic_num):
    """Get threshold value based on topic number"""
    if topic_num == 7:
        return 0.8
    else:
        sorted_scores = np.sort(scores)[::-1]
        smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
        x = np.arange(len(smoothed_scores))
        knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
        if knee_locator.knee is not None:
            knee_index = knee_locator.knee
            return smoothed_scores[knee_index]
        else:
            return sorted_scores.mean() + sorted_scores.std()

def calculate_community_aic(G, partition):
    """
    Calculate AIC for a community structure using description length principle
    
    Args:
        G (networkx.Graph): The graph
        partition (dict): Mapping from node to community ID
        
    Returns:
        float: AIC value
    """
    # Get number of communities
    communities = {}
    for node, comm_id in partition.items():
        if comm_id not in communities:
            communities[comm_id] = []
        communities[comm_id].append(node)
    
    k = len(communities)  # Number of parameters (communities)
    
    # Calculate internal and external edges
    internal_edges = 0
    external_edges = 0
    
    for u, v in G.edges():
        if partition[u] == partition[v]:
            internal_edges += 1
        else:
            external_edges += 1
    
    # Calculate modularity
    modularity = community_louvain.modularity(partition, G)
    
    # Log-likelihood approximation based on modularity and description length
    log_likelihood = modularity - (external_edges / (internal_edges + 1))
    
    # Calculate AIC: 2k - 2ln(L)
    aic = 2 * k - 2 * log_likelihood
    
    return aic

def calculate_silhouette_score(G, partition):
    """
    Calculate silhouette score for the communities
    
    Args:
        G (networkx.Graph): The network graph
        partition (dict): Mapping from node to community ID
        
    Returns:
        float: Average silhouette score (-1 to 1, higher is better)
    """
    # This function uses a neighbor-based approximation that works with disconnected graphs
    print("Using neighbor-based silhouette calculation for disconnected graph")
    
    # Group nodes by community
    communities = defaultdict(list)
    for node, comm_id in partition.items():
        communities[comm_id].append(node)
    
    # Calculate silhouette score for each node
    silhouette_scores = []
    
    for node in G.nodes():
        comm_id = partition[node]
        
        # Get all neighbors of this node
        neighbors = set(G.neighbors(node))
        if not neighbors:
            continue  # Skip isolated nodes
        
        # Count neighbors in same community
        same_comm_neighbors = sum(1 for n in neighbors if partition.get(n) == comm_id)
        
        # Count neighbors in different communities
        diff_comm_neighbors = sum(1 for n in neighbors if partition.get(n) != comm_id)
        
        # Calculate a cohesion measure (how well node fits in its community)
        a_i = 1.0 - (same_comm_neighbors / len(neighbors)) if neighbors else 0
        
        # Calculate a separation measure (how well separated from other communities)
        b_i = diff_comm_neighbors / len(neighbors) if neighbors else 1
        
        # Calculate silhouette
        if len(neighbors) == 0:
            continue  # Skip nodes with no neighbors
        elif a_i < b_i:
            s_i = 1.0 - (a_i / max(b_i, 1e-10))
        else:
            s_i = (b_i / max(a_i, 1e-10)) - 1.0
            
        silhouette_scores.append(s_i)
    
    if not silhouette_scores:
        return 0.0  # Default for empty list
        
    # Return average silhouette score
    return np.mean(silhouette_scores)

def run_permutation_test(G, partition, n_permutations=100):
    """
    Run permutation test to assess statistical significance of communities
    by comparing modularity against random networks
    
    Args:
        G (networkx.Graph): The network graph
        partition (dict): Mapping from node to community ID
        n_permutations (int): Number of permutations to run
        
    Returns:
        tuple: (p-value, z-score)
    """
    print(f"Running permutation test with {n_permutations} permutations...")
    
    # Calculate actual modularity
    actual_modularity = community_louvain.modularity(partition, G)
    
    # Store modularity values from random permutations
    random_modularities = []
    
    # Run permutations (just shuffle the partition labels)
    # This is more efficient and reliable than creating new random networks
    nodes = list(G.nodes())
    
    for i in range(n_permutations):
        # Create a shuffled partition
        community_labels = list(partition.values())
        random.shuffle(community_labels)
        random_partition = {node: label for node, label in zip(nodes, community_labels)}
        
        # Calculate modularity on shuffled partition
        try:
            rand_mod = community_louvain.modularity(random_partition, G)
            random_modularities.append(rand_mod)
        except Exception as e:
            print(f"Warning: Error calculating modularity for permutation {i}: {e}")
            # Use a low modularity value as fallback
            random_modularities.append(0.0)
    
    # Calculate p-value (proportion of random networks with modularity >= actual)
    p_value = sum(1 for m in random_modularities if m >= actual_modularity) / max(1, len(random_modularities))
    
    # Calculate z-score
    if len(random_modularities) > 1 and np.std(random_modularities) > 0:
        z_score = (actual_modularity - np.mean(random_modularities)) / np.std(random_modularities)
    else:
        z_score = float('inf') if actual_modularity > 0 else float('-inf')
    
    return p_value, z_score

def detect_communities_hierarchical(G, min_cluster_size=3):
    """
    Detect communities hierarchically using modularity optimization
    and calculate AIC for each hierarchy level
    
    Args:
        G (networkx.Graph): The graph to analyze
        min_cluster_size (int): Minimum size of communities to consider
        
    Returns:
        tuple: (Communities at different levels, AIC values)
    """
    # If the graph is too small or has no edges, return empty results
    if len(G.nodes()) < min_cluster_size or len(G.edges()) == 0:
        print("Graph too small or has no edges for community detection")
        return {}, {}
    
    # Initialize results containers
    hierarchy_levels = {}
    aic_values = {}
    silhouette_scores = {}
    
    # Level 0: Original graph (each node is its own community)
    level0_partition = {node: i for i, node in enumerate(G.nodes())}
    hierarchy_levels[0] = level0_partition
    aic_values[0] = calculate_community_aic(G, level0_partition)
    print(f"Level 0 (each node as community) AIC: {aic_values[0]:.4f}")
    
    # Level 1: Initial partitioning using Louvain method
    partition = community_louvain.best_partition(G)
    hierarchy_levels[1] = partition
    
    # Calculate modularity for level 1
    modularity = community_louvain.modularity(partition, G)
    print(f"Level 1 modularity: {modularity:.4f}")
    
    # Run permutation test for level 1
    p_value, z_score = run_permutation_test(G, partition, n_permutations=1000)
    print(f"Level 1 statistical significance: p-value={p_value:.4f}, z-score={z_score:.4f}")
    print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
    
    # Calculate silhouette score for level 1
    silhouette = calculate_silhouette_score(G, partition)
    silhouette_scores[1] = silhouette
    print(f"Level 1 silhouette score: {silhouette:.4f}")
    print(f"  Interpretation: {interpret_silhouette(silhouette)}")
    
    # Count communities of size >= min_cluster_size
    communities = defaultdict(list)
    for node, community_id in partition.items():
        communities[community_id].append(node)
    
    significant_communities = {k: v for k, v in communities.items() if len(v) >= min_cluster_size}
    print(f"Level 1: Found {len(communities)} communities, {len(significant_communities)} with size >= {min_cluster_size}")
    
    # Calculate AIC for level 1
    aic_values[1] = calculate_community_aic(G, partition)
    print(f"Level 1 AIC: {aic_values[1]:.4f}")
    
    # Level 2+: Apply hierarchical subdivision
    current_level = 2
    has_subcommunities = True
    
    while has_subcommunities and current_level <= 5:  # Limit to 5 levels
        has_subcommunities = False
        new_partition = partition.copy()
        max_community_id = max(partition.values())
        
        # For each significant community at the previous level
        for comm_id, nodes in significant_communities.items():
            # Create subgraph for this community
            subgraph = G.subgraph(nodes)
            
            # Only proceed if the subgraph has enough nodes and edges
            if len(subgraph.nodes()) >= min_cluster_size * 2 and len(subgraph.edges()) >= len(subgraph.nodes()) / 2:
                try:
                    # Apply community detection to the subgraph
                    sub_partition = community_louvain.best_partition(subgraph)
                    
                    # Check if we found meaningful sub-communities
                    sub_communities = defaultdict(list)
                    for node, sub_comm_id in sub_partition.items():
                        sub_communities[sub_comm_id].append(node)
                    
                    significant_sub_communities = {k: v for k, v in sub_communities.items() if len(v) >= min_cluster_size}
                    
                    # If we found at least 2 significant sub-communities
                    if len(significant_sub_communities) >= 2:
                        has_subcommunities = True
                        
                        # Calculate modularity of the sub-partition
                        sub_modularity = community_louvain.modularity(sub_partition, subgraph)
                        
                        # Only use this subdivision if it's meaningful (positive modularity)
                        if sub_modularity > 0.05:
                            # Update the partition with new sub-community IDs
                            for sub_comm_id, sub_nodes in significant_sub_communities.items():
                                max_community_id += 1
                                for node in sub_nodes:
                                    new_partition[node] = max_community_id
                except Exception as e:
                    print(f"Error subdividing community {comm_id}: {str(e)}")
        
        # If we found sub-communities, update the hierarchy
        if has_subcommunities:
            partition = new_partition
            hierarchy_levels[current_level] = partition
            aic_values[current_level] = calculate_community_aic(G, partition)
            
            # Calculate modularity for this level
            modularity = community_louvain.modularity(partition, G)
            print(f"Level {current_level} modularity: {modularity:.4f}")
            
            # Run permutation test for this level
            p_value, z_score = run_permutation_test(G, partition, n_permutations=1000)
            print(f"Level {current_level} statistical significance: p-value={p_value:.4f}, z-score={z_score:.4f}")
            print(f"  Interpretation: {'Significant' if p_value < 0.05 else 'Not significant'} at α=0.05")
            
            # Calculate silhouette score for this level
            silhouette = calculate_silhouette_score(G, partition)
            silhouette_scores[current_level] = silhouette
            print(f"Level {current_level} silhouette score: {silhouette:.4f}")
            print(f"  Interpretation: {interpret_silhouette(silhouette)}")
            
            # Count communities
            communities = defaultdict(list)
            for node, community_id in partition.items():
                communities[community_id].append(node)
            
            significant_communities = {k: v for k, v in communities.items() if len(v) >= min_cluster_size}
            print(f"Level {current_level}: Found {len(communities)} communities, {len(significant_communities)} with size >= {min_cluster_size}")
            print(f"Level {current_level} AIC: {aic_values[current_level]:.4f}")
            
            current_level += 1
        else:
            print(f"No further subdivision possible at level {current_level}")
    
    # Find the level with the lowest AIC
    if aic_values:
        best_level = min(aic_values, key=aic_values.get)
        print(f"\nBest hierarchical level based on AIC: Level {best_level}")
        print(f"Best AIC value: {aic_values[best_level]:.4f}")
        
        # Check if the best level is also statistically significant
        if best_level in silhouette_scores:
            print(f"Silhouette score at best level: {silhouette_scores[best_level]:.4f}")
            print(f"Interpretation: {interpret_silhouette(silhouette_scores[best_level])}")
    
    return hierarchy_levels, aic_values, silhouette_scores

def interpret_silhouette(score):
    """Interpret silhouette score value"""
    if score < 0:
        return "Poor structure (negative score indicates misclassifications)"
    elif score < 0.25:
        return "No substantial structure"
    elif score < 0.5:
        return "Weak structure"
    elif score < 0.7:
        return "Reasonable structure"
    else:
        return "Strong structure"

def visualize_communities(G, partition, topic_num, level):
    """
    Visualize communities detected in the network
    
    Args:
        G (networkx.Graph): The graph to visualize
        partition (dict): Mapping from node to community ID
        topic_num (int): The topic number
        level (int): The hierarchical level
    """
    # Convert partition to list of sets for coloring
    communities = defaultdict(set)
    for node, comm_id in partition.items():
        communities[comm_id].add(node)
    
    community_list = list(communities.values())
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Set random seed for layout reproducibility
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Draw nodes with community colors
    for i, community in enumerate(community_list):
        nx.draw_networkx_nodes(G, pos, nodelist=list(community), 
                              node_color=plt.cm.tab20(i % 20),
                              node_size=[G.nodes[node]['topic_score'] * 300 for node in community], 
                              alpha=0.6)
    
    # Draw edges
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 * weight/max_weight for weight in edge_weights]
    
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          alpha=0.5, edge_color='gray')
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Get modularity
    modularity = community_louvain.modularity(partition, G)
    
    # Get number of communities
    num_communities = len(set(partition.values()))
    
    # Add title and info
    plt.title(f'Topic {topic_num} Author Network - Level {level} Communities\n'
              f'Communities: {num_communities}, Modularity: {modularity:.4f}')
    
    # Save plot
    plt.savefig(f'topic_{topic_num}_level_{level}_communities.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save community assignments to CSV
    community_df = pd.DataFrame({
        'Author': list(partition.keys()),
        'Community': list(partition.values())
    })
    community_df.to_csv(f'topic_{topic_num}_level_{level}_communities.csv', index=False)
    
    return community_list, modularity

def export_community_metrics(G, communities, topic_num, level):
    """
    Export metrics for each community
    
    Args:
        G (networkx.Graph): The graph to analyze
        communities (list): List of sets of nodes for each community
        topic_num (int): The topic number
        level (int): The hierarchical level
    """
    # Initialize data collection
    community_metrics = []
    
    for i, community in enumerate(communities):
        # Skip communities with less than 3 nodes
        if len(community) < 3:
            continue
            
        # Create subgraph for this community
        subgraph = G.subgraph(community)
        
        # Calculate community metrics
        avg_topic_score = np.mean([G.nodes[node]['topic_score'] for node in community])
        density = nx.density(subgraph)
        
        # Calculate centralization measures
        degree_centrality = nx.degree_centrality(subgraph)
        degree_centralization = np.var(list(degree_centrality.values())) * len(degree_centrality)
        
        betweenness_centrality = nx.betweenness_centrality(subgraph)
        betweenness_centralization = np.var(list(betweenness_centrality.values())) * len(betweenness_centrality)
        
        # Count internal and external edges
        internal_edges = len(subgraph.edges())
        external_edges = sum(1 for node in community for neighbor in G.neighbors(node) if neighbor not in community)
        
        # Calculate conductance: ratio of external to total edges
        total_edges = internal_edges + external_edges
        conductance = external_edges / total_edges if total_edges > 0 else 0
        
        # Add to metrics
        community_metrics.append({
            'Community': i,
            'Size': len(community),
            'Avg_Topic_Score': avg_topic_score,
            'Density': density,
            'Internal_Edges': internal_edges,
            'External_Edges': external_edges,
            'Conductance': conductance,
            'Degree_Centralization': degree_centralization,
            'Betweenness_Centralization': betweenness_centralization
        })
    
    # Create and save DataFrame
    if community_metrics:
        metrics_df = pd.DataFrame(community_metrics)
        metrics_df.to_csv(f'topic_{topic_num}_level_{level}_community_metrics.csv', index=False)
        
        # Print summary
        print(f"\nCommunity Metrics for Topic {topic_num}, Level {level}:")
        print(f"Number of significant communities: {len(metrics_df)}")
        print(f"Average community size: {metrics_df['Size'].mean():.2f}")
        print(f"Average density: {metrics_df['Density'].mean():.4f}")
        print(f"Average conductance: {metrics_df['Conductance'].mean():.4f}")

def create_collaboration_network(df, topic_num):
    """Create and visualize collaboration network for a specific topic"""
    topic_col = f'Topic {topic_num} Score'
    
    # Get threshold and filter papers
    threshold = get_threshold(df[topic_col], topic_num)
    filtered_df = df[df[topic_col] > threshold].copy()
    
    # Remove duplicates based on Title
    filtered_df = filtered_df.drop_duplicates(subset=['Title'])
    
    # Initialize network
    G = nx.Graph()
    
    # Track author topic scores and collaboration weights
    author_topic_scores = defaultdict(float)
    collaboration_weights = defaultdict(float)
    collaboration_counts = defaultdict(int)
    
    # Process each paper to calculate author topic scores
    for _, row in filtered_df.iterrows():
        submitters = row['Submitted By'].split(', ')
        topic_score = row[topic_col]
        
        # Add topic score for each submitter
        for submitter in submitters:
            author_topic_scores[submitter] += topic_score
        
        # Add collaboration edges if multiple submitters
        if len(submitters) > 1:
            for i in range(len(submitters)):
                for j in range(i + 1, len(submitters)):
                    author1, author2 = submitters[i], submitters[j]
                    # Edge weight is sum of both authors' topic scores
                    collaboration_weights[(author1, author2)] = (
                        author_topic_scores[author1] + author_topic_scores[author2]
                    )
                    collaboration_counts[(author1, author2)] += 1
    
    # Add all authors as nodes with their topic scores
    for author, score in author_topic_scores.items():
        G.add_node(author, topic_score=score)
    
    # Add weighted edges for collaborations
    for (author1, author2), weight in collaboration_weights.items():
        G.add_edge(author1, author2, weight=weight, count=collaboration_counts[(author1, author2)])
    
    # Remove isolated nodes (nodes with no edges)
    isolated_nodes = [node for node, degree in dict(G.degree()).items() if degree == 0]
    G.remove_nodes_from(isolated_nodes)
    
    print(f"Removed {len(isolated_nodes)} isolated nodes from the network")
    
    # Calculate node sizes based on topic scores
    node_sizes = [G.nodes[node]['topic_score'] * 300 for node in G.nodes()]
    
    # Calculate edge widths based on weights
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 * weight/max_weight for weight in edge_weights]
    
    # Create figure
    plt.figure(figsize=(15, 10))
    
    # Set random seed for layout reproducibility
    pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                          node_color='lightblue', alpha=0.6)
    nx.draw_networkx_edges(G, pos, width=edge_widths, 
                          alpha=0.5, edge_color='gray')
    nx.draw_networkx_labels(G, pos, font_size=8)
    
    # Add title and info
    plt.title(f'Topic {topic_num} Author Network (Without Isolated Nodes)\n'
              f'Papers above threshold: {len(filtered_df)}, '
              f'Authors: {len(G.nodes)}, '
              f'Collaborations: {len(G.edges)}')
    
    # Save plot
    plt.savefig(f'topic_{topic_num}_author_network.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print network statistics
    print(f"\nTopic {topic_num} Network Statistics:")
    print(f"Number of papers above threshold: {len(filtered_df)}")
    print(f"Number of authors (after removing isolated): {len(G.nodes)}")
    print(f"Number of collaboration edges: {len(G.edges)}")
    
    # Calculate centrality metrics
    degree_centrality = nx.degree_centrality(G)
    betweenness_centrality = nx.betweenness_centrality(G)
    closeness_centrality = nx.closeness_centrality(G)
    eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # Calculate weighted degree (sum of edge weights) and degree (count of edges)
    weighted_degree = {node: sum(d['weight'] for _, _, d in G.edges(node, data=True)) for node in G.nodes()}
    degree = {node: sum(1 for _ in G.edges(node)) for node in G.nodes()}
    
    # Create dataframe with metrics
    metrics_df = pd.DataFrame({
        'Author': list(G.nodes()),
        'Topic_Score': [G.nodes[node]['topic_score'] for node in G.nodes()],
        'Weighted_Degree': [weighted_degree.get(node, 0) for node in G.nodes()],
        'Degree': [degree.get(node, 0) for node in G.nodes()],
        'Degree_Centrality': [degree_centrality[node] for node in G.nodes()],
        'Betweenness_Centrality': [betweenness_centrality[node] for node in G.nodes()],
        'Closeness_Centrality': [closeness_centrality[node] for node in G.nodes()],
        'Eigenvector_Centrality': [eigenvector_centrality[node] for node in G.nodes()]
    })
    
    # Save metrics to CSV
    metrics_df.to_csv(f'topic_{topic_num}_network_metrics_kneed.csv', index=False)
    
    # Perform hierarchical community detection
    print("\nPerforming hierarchical community detection...")
    hierarchy_levels, aic_values, silhouette_scores = detect_communities_hierarchical(G)
    
    # Visualize communities and export metrics
    if hierarchy_levels:
        # Find best level based on AIC
        best_level = min(aic_values, key=aic_values.get) if aic_values else max(hierarchy_levels.keys())
        best_partition = hierarchy_levels[best_level]
        
        print(f"\nVisualizing best communities (Level {best_level})...")
        communities, modularity = visualize_communities(G, best_partition, topic_num, best_level)
        
        # Export community metrics
        print("\nExporting community metrics...")
        export_community_metrics(G, communities, topic_num, best_level)
        
        # Save AIC values to CSV
        results_df = pd.DataFrame({
            'Level': list(aic_values.keys()),
            'AIC': list(aic_values.values()),
            # 'Silhouette': [silhouette_scores.get(level, float('nan')) for level in aic_values.keys()]
        })
        results_df.to_csv(f'topic_{topic_num}_community_statistics.csv', index=False)
    
    return G

def main(filepath):
    df = process_data(filepath)
    green_topics = {1, 5, 7, 14}
    
    print("\nGenerating author networks with hierarchical community detection:")
    print("-" * 60)
    
    networks = {}
    for topic_num in green_topics:
        print(f"\nProcessing Topic {topic_num}...")
        networks[topic_num] = create_collaboration_network(df, topic_num)
    
    print("\nAll networks and community analyses generated successfully!")

if __name__ == "__main__":
    main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')