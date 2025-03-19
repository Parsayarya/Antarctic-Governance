# import pandas as pd
# import numpy as np
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator
# from pyvis.network import Network
# import networkx as nx
# from collections import defaultdict

# def process_data(filepath):
#     df = pd.read_csv(filepath)
#     return df

# def get_knee_point(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()

# def create_bipartite_network(df):
#     # Initialize network
#     net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='#000000')
#     net.toggle_physics(True)
    
#     # Special topics to be colored green
#     green_topics = {1, 5, 7, 14}
    
#     # Process each topic
#     topic_submitter_weights = defaultdict(float)
    
#     print("\nNumber of papers remaining after knee point filtering per topic:")
#     print("-" * 60)
#     print(f"{'Topic':^10} {'Total Papers':^15} {'After Knee':^15} {'Percentage':^15}")
#     print("-" * 60)
    
#     for topic_num in range(20):
#         topic_col = f'Topic {topic_num} Score'
        
#         # Get knee point for this topic
#         knee_value = get_knee_point(df[topic_col], topic_num)
        
#         # Filter papers above knee point
#         filtered_df = df[df[topic_col] > knee_value]
        
#         # Calculate and print statistics
#         total_papers = len(df)
#         remaining_papers = len(filtered_df)
#         percentage = (remaining_papers / total_papers) * 100
        
#         print(f"{topic_num:^10} {total_papers:^15} {remaining_papers:^15} {percentage:^15.2f}%")
        
#         # Process submitters for filtered papers
#         for _, row in filtered_df.iterrows():
#             submitters = row['Submitted By'].split(', ')
#             score = row[topic_col]
            
#             for submitter in submitters:
#                 topic_submitter_weights[(topic_num, submitter)] += score
    
#     print("-" * 60)
    
#     # Add nodes and edges
#     added_topics = set()
#     added_submitters = set()
    
#     for (topic_num, submitter), weight in topic_submitter_weights.items():
#         if topic_num not in added_topics:
#             color = '#00ff00' if topic_num in green_topics else '#1f77b4'
#             net.add_node(f'Topic {topic_num}', 
#                         label=f'Topic {topic_num}',
#                         color=color,
#                         title=f'Topic {topic_num}',
#                         size=15)
#             added_topics.add(topic_num)
        
#         if submitter not in added_submitters:
#             net.add_node(submitter,
#                         label=submitter,
#                         color='#ff7f0e',
#                         title=submitter,
#                         size=5)
#             added_submitters.add(submitter)
        
#         net.add_edge(f'Topic {topic_num}',
#                     submitter,
#                     value=weight,
#                     title=f'Weight: {weight:.2f}')
    
#     # Print total network statistics
#     print(f"\nTotal network statistics:")
#     print(f"Number of topic nodes: {len(added_topics)}")
#     print(f"Number of submitter nodes: {len(added_submitters)}")
#     print(f"Number of connections: {len(topic_submitter_weights)}")
    
#     # Set physics layout options
#     net.set_options("""
#     const options = {
#       "physics": {
#         "enabled": true,
#         "barnesHut": {
#           "gravitationalConstant": -100000,
#           "centralGravity": 5,
#           "springLength": 50,
#           "springConstant": 0.1,
#           "avoidOverlap": 1.0
#         },
#         "minVelocity": 0.1,
#         "maxVelocity": 0.1,
#         "solver": "barnesHut",
#         "stabilization": {
#           "enabled": true,
#           "iterations": 3000,
#           "updateInterval": 50,
#           "onlyDynamicEdges": false,
#           "fit": true
#         },
#         "timestep": 0.1,
#         "adaptiveTimestep": true
#       },
#       "layout": {
#         "improvedLayout": true,
#         "randomSeed": 42,
#         "hierarchical": {
#           "enabled": false
#         },
#         "interaction": {   
#           "navigationButtons": true,
#           "dragNodes": true,
#           "dragView": true,
#           "zoomView": true
#         }
#       }
#     }
#     """)
    
#     return net

# def main(filepath):
#     df = process_data(filepath)
#     net = create_bipartite_network(df)
#     net.save_graph('topic_submitter_Full_Bi_network.html')

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')
# ------------------------------------------------------------------- With NetworkX
# import pandas as pd
# import numpy as np
# import networkx as nx
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def process_data(filepath):
#     df = pd.read_csv(filepath)
#     return df

# def get_knee_point(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         # Sort scores in descending order
#         sorted_scores = np.sort(scores)[::-1]
#         # Return the score at index 73 (74th paper)
#         # This will make the threshold include exactly the top 74 papers
#         return sorted_scores[73] if len(sorted_scores) > 73 else sorted_scores[-1]
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()

# def create_bipartite_network(df):
#     B = nx.Graph()
    
#     # Special topics to be colored green
#     green_topics = {1, 5, 7, 14}
#     topic_submitter_weights = defaultdict(float)
    
#     for topic_num in green_topics:
#         topic_col = f'Topic {topic_num} Score'
#         knee_value = get_knee_point(df[topic_col], topic_num)
#         filtered_df = df[df[topic_col] > knee_value]
        
#         # Process submitters for filtered papers
#         for _, row in filtered_df.iterrows():
#             submitters = row['Submitted By'].split(', ')
#             score = row[topic_col]
#             for submitter in submitters:
#                 topic_submitter_weights[(f'Topic {topic_num}', submitter)] += score
    
#     # Add nodes and edges to the bipartite graph
#     for (topic, submitter), weight in topic_submitter_weights.items():
#         # Assign color based on topic group
#         topic_color = '#00ff00' if int(topic.split()[-1]) in green_topics else '#1f77b4'
#         B.add_node(topic, bipartite=0, color=topic_color, size=15, shape='rectangle')
#         B.add_node(submitter, bipartite=1, color='#ff7f0e', size=5, shape='circle')
#         B.add_edge(topic, submitter, weight=weight)
    
#     return B

# def visualize_bipartite_network(B):
#     plt.figure(figsize=(12, 12))
    
#     # Get the positions for the two node sets using spring layout
#     pos = nx.spring_layout(B, k=2, iterations=100)
    
#     # Separate topics and submitters for distinct styles
#     topic_nodes = [n for n in B.nodes() if B.nodes[n].get('shape') == 'rectangle']
#     submitter_nodes = [n for n in B.nodes() if B.nodes[n].get('shape') == 'circle']
    
#     # Separate sizes for each type of node
#     topic_sizes = [B.nodes[n]['size'] * 25 for n in topic_nodes]
#     submitter_sizes = [B.nodes[n]['size'] * 25 for n in submitter_nodes]
    
#     # Draw edges
#     nx.draw_networkx_edges(B, pos, 
#                            edge_color="gray", 
#                            width=[B[u][v]['weight'] / max(nx.get_edge_attributes(B, 'weight').values()) * 2 
#                                   for u, v in B.edges()], alpha=0.7)
    
#     # Draw topic nodes as rectangles
#     nx.draw_networkx_nodes(B, pos, nodelist=topic_nodes, node_shape='s', node_size=topic_sizes, 
#                            node_color=[B.nodes[n]['color'] for n in topic_nodes], alpha=0.7)
    
#     # Draw submitter nodes as circles
#     nx.draw_networkx_nodes(B, pos, nodelist=submitter_nodes, node_shape='o', node_size=submitter_sizes, 
#                            node_color=[B.nodes[n]['color'] for n in submitter_nodes], alpha=0.7)
    
#     # Draw labels with reduced font size
#     nx.draw_networkx_labels(B, pos, font_size=6)
    
#     # Bold font for topic nodes
#     topic_labels = {node: node for node in topic_nodes}
    
#     plt.title("Bipartite Topic-Submitter Network\nGreen Topics: Conservation Topics", fontsize=16, pad=20)
#     plt.axis("off")
#     plt.savefig('topic_submitter_Binetwork_Conservation.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()

# def main(filepath):
#     df = process_data(filepath)
#     B = create_bipartite_network(df)
#     visualize_bipartite_network(B)

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')


# ----------------------------------------------------------------------- JUST Conservation
# import pandas as pd
# import numpy as np
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator
# from pyvis.network import Network
# import networkx as nx
# from collections import defaultdict

# def process_data(filepath):
#     df = pd.read_csv(filepath)
#     return df

# def get_knee_point(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         # Sort scores in descending order
#         sorted_scores = np.sort(scores)[::-1]
#         # Return the score at index 73 (74th paper)
#         # This will make the threshold include exactly the top 74 papers
#         return sorted_scores[73] if len(sorted_scores) > 73 else sorted_scores[-1]
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()


# def create_bipartite_network(df):
#     # Initialize network
#     net = Network(height='750px', width='100%', bgcolor='#ffffff', font_color='#000000')
#     net.toggle_physics(True)
    
#     # Only include green topics
#     green_topics = {1, 5, 7, 14}
    
#     # Process each topic
#     topic_submitter_weights = defaultdict(float)
    
#     print("\nNumber of papers remaining after knee point filtering per topic:")
#     print("-" * 60)
#     print(f"{'Topic':^10} {'Total Papers':^15} {'After Knee':^15} {'Percentage':^15}")
#     print("-" * 60)
    
#     # Only process green topics
#     for topic_num in green_topics:
#         topic_col = f'Topic {topic_num} Score'
        
#         # Get knee point for this topic
#         knee_value = get_knee_point(df[topic_col], topic_num)
        
#         # Filter papers above knee point
#         filtered_df = df[df[topic_col] > knee_value]
        
#         # Calculate and print statistics
#         total_papers = len(df)
#         remaining_papers = len(filtered_df)
#         percentage = (remaining_papers / total_papers) * 100
        
#         print(f"{topic_num:^10} {total_papers:^15} {remaining_papers:^15} {percentage:^15.2f}%")
        
#         # Process submitters for filtered papers
#         for _, row in filtered_df.iterrows():
#             submitters = row['Submitted By'].split(', ')
#             score = row[topic_col]
            
#             for submitter in submitters:
#                 topic_submitter_weights[(topic_num, submitter)] += score
    
#     print("-" * 60)
    
#     # Add nodes and edges
#     added_topics = set()
#     added_submitters = set()
    
#     for (topic_num, submitter), weight in topic_submitter_weights.items():
#         if topic_num not in added_topics:
#             net.add_node(f'Topic {topic_num}', 
#                         label=f'Topic {topic_num}',
#                         color='#00ff00',  # All topics will be green
#                         title=f'Topic {topic_num}',
#                         size=15)
#             added_topics.add(topic_num)
        
#         if submitter not in added_submitters:
#             net.add_node(submitter,
#                         label=submitter,
#                         color='#ff7f0e',
#                         title=submitter,
#                         size=5)
#             added_submitters.add(submitter)
        
#         net.add_edge(f'Topic {topic_num}',
#                     submitter,
#                     value=weight,
#                     title=f'Weight: {weight:.2f}')
    
#     # Print total network statistics
#     print(f"\nTotal network statistics:")
#     print(f"Number of topic nodes: {len(added_topics)}")
#     print(f"Number of submitter nodes: {len(added_submitters)}")
#     print(f"Number of connections: {len(topic_submitter_weights)}")
    
#     # Set physics layout options
#     net.set_options("""
#     const options = {
#       "physics": {
#         "enabled": true,
#         "barnesHut": {
#           "gravitationalConstant": -1000000,
#           "centralGravity": 0.8,
#           "springLength": 100,
#           "springConstant": 0.9,
#           "damping": 0.8,
#           "avoidOverlap": 1.0
#         },
#         "minVelocity": 0.1,
#         "maxVelocity": 0.1,
#         "solver": "barnesHut",
#         "stabilization": {
#           "enabled": true,
#           "iterations": 3000,
#           "updateInterval": 50,
#           "onlyDynamicEdges": false,
#           "fit": true
#         },
#         "timestep": 0.1,
#         "adaptiveTimestep": true
#       },
#       "layout": {
#         "improvedLayout": true,
#         "randomSeed": 42,
#         "hierarchical": {
#           "enabled": false
#         },
#         "interaction": {   
#           "navigationButtons": true,
#           "dragNodes": true,
#           "dragView": true,
#           "zoomView": true
#         }
#       }
#     }
#     """)
    
#     return net

# def main(filepath):
#     df = process_data(filepath)
#     net = create_bipartite_network(df)
#     net.save_graph('topic_submitter_Binetwork_just_conservation_pyvis_network.html')

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')
# ----------------------------------------------------------------------- Big Co-Authorship Network
# import pandas as pd
# import numpy as np
# import seaborn as sns
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def analyze_coauthorship(df):
#     # Create a dictionary to store co-authorship counts
#     coauthor_counts = defaultdict(lambda: defaultdict(int))
#     author_total_collabs = defaultdict(int)
    
#     # Process each paper's authors
#     for authors in df['Submitted By']:
#         author_list = [author.strip() for author in authors.split(',')]
        
#         # Count co-authorships for each pair
#         for i in range(len(author_list)):
#             for j in range(i + 1, len(author_list)):
#                 author1 = author_list[i]
#                 author2 = author_list[j]
#                 coauthor_counts[author1][author2] += 1
#                 coauthor_counts[author2][author1] += 1
#                 author_total_collabs[author1] += 1
#                 author_total_collabs[author2] += 1
    
#     # Convert to DataFrame for heatmap
#     authors = sorted(author_total_collabs.keys())
#     matrix_data = []
#     for author1 in authors:
#         row = []
#         for author2 in authors:
#             if author1 == author2:
#                 row.append(0)
#             else:
#                 row.append(coauthor_counts[author1][author2])
#         matrix_data.append(row)
    
#     coauthor_matrix = pd.DataFrame(matrix_data, index=authors, columns=authors)
    
#     # Find top 5 collaborators
#     top_collaborators = sorted([(author, sum(coauthor_counts[author].values())) 
#                               for author in authors], 
#                              key=lambda x: x[1], 
#                              reverse=True)[:5]
    
#     # Get top 3 collaborators for each top 5 author
#     top_collab_details = {}
#     for author, _ in top_collaborators:
#         collaborators = sorted(coauthor_counts[author].items(), key=lambda x: x[1], reverse=True)[:3]
#         top_collab_details[author] = collaborators
    
#     return coauthor_matrix, top_collaborators, top_collab_details

# def plot_coauthorship(coauthor_matrix, top_collaborators, top_collab_details):
#     # Create figure with two subplots
#     plt.figure(figsize=(25, 10))
    
#     # Heatmap subplot
#     plt.subplot(1, 2, 1)
    
#     # Calculate appropriate font size based on matrix size
#     n_authors = len(coauthor_matrix)
#     font_size = max(8, min(10, 200 / n_authors))  # Adjust font size based on number of authors
    
#     # Create heatmap with adjusted parameters
#     sns.heatmap(coauthor_matrix, 
#                 cmap='YlOrRd',
#                 xticklabels=True,
#                 yticklabels=True)
    
#     plt.title('Co-authorship Heatmap', pad=20)
    
#     # Rotate labels and adjust them to be visible
#     plt.xticks(rotation=90, ha='center', fontsize=font_size)
#     plt.yticks(rotation=0, va='center', fontsize=font_size)
    
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
    
#     # Bar plot
#     plt.subplot(1, 2, 2)
#     authors = [author for author, count in top_collaborators]
#     counts = [count for author, count in top_collaborators]
    
#     bars = plt.bar(authors, counts)
#     plt.title('Top 5 Collaborators')
#     plt.xlabel('Author')
#     plt.ylabel('Number of Collaborations')
    
#     # Rotate x-labels for better readability
#     plt.xticks(rotation=45, ha='right')
    
#     # Add top 3 collaborators text in each bar
#     for i, bar in enumerate(bars):
#         author = authors[i]
#         collab_text = '\n'.join([f"{c[0]}: {c[1]}" for c in top_collab_details[author]])
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, height/2,
#                 collab_text,
#                 ha='center', va='center',
#                 fontsize=8, color='white',
#                 bbox=dict(facecolor='black', alpha=0.5))
    
#     # Adjust layout
#     plt.tight_layout()
    
#     # Save the figure with high resolution
#     plt.savefig('coauthorship_analysis.png', dpi=300, bbox_inches='tight')
#     plt.close()

# def main(filepath):
#     # Read data
#     df = pd.read_csv(filepath)
    
#     # Analyze co-authorship
#     coauthor_matrix, top_collaborators, top_collab_details = analyze_coauthorship(df)
    
#     # Create visualizations
#     plot_coauthorship(coauthor_matrix, top_collaborators, top_collab_details)
    
#     # Print top collaborators and their details
#     print("\nTop 5 Collaborators and their most frequent co-authors:")
#     print("-" * 60)
#     for author, count in top_collaborators:
#         print(f"\n{author} (Total collaborations: {count})")
#         print("Top 3 co-authors:")
#         for coauthor, colab_count in top_collab_details[author]:
#             print(f"  - {coauthor}: {colab_count} collaborations")

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def create_coauthor_network(df):
#     # Create a dictionary to store co-authorship counts
#     coauthor_counts = defaultdict(lambda: defaultdict(int))
    
#     # Calculate topic scores per author
#     author_topic_scores = defaultdict(lambda: defaultdict(float))
#     topic_columns = [col for col in df.columns if 'Topic' in col and 'Score' in col]
    
#     # Process each paper's authors and their topic scores
#     for _, row in df.iterrows():
#         author_list = [author.strip() for author in row['Submitted By'].split(',')]
        
#         # Count co-authorships
#         for i in range(len(author_list)):
#             for j in range(i + 1, len(author_list)):
#                 author1 = author_list[i]
#                 author2 = author_list[j]
#                 coauthor_counts[author1][author2] += 1
#                 coauthor_counts[author2][author1] += 1
                
#         # Sum topic scores for each author
#         for author in author_list:
#             for topic_col in topic_columns:
#                 author_topic_scores[author][topic_col] += row[topic_col]
    
#     # Create NetworkX graph
#     G = nx.Graph()
    
#     # Add edges with weights
#     total_collaborations = 0
#     for author1 in coauthor_counts:
#         for author2, weight in coauthor_counts[author1].items():
#             if weight > 0:
#                 G.add_edge(author1, author2, weight=weight)
#                 total_collaborations += weight
    
#     # Calculate centrality metrics
#     degree_centrality = nx.degree_centrality(G)
#     betweenness_centrality = nx.betweenness_centrality(G)
#     closeness_centrality = nx.closeness_centrality(G)
#     eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
#     # Create results DataFrame
#     results = []
#     for node in G.nodes():
#         node_data = {
#             'Author': node,
#             'Degree': G.degree(node),
#             'Degree_Centrality': degree_centrality[node],
#             'Betweenness_Centrality': betweenness_centrality[node],
#             'Closeness_Centrality': closeness_centrality[node],
#             'Eigenvector_Centrality': eigenvector_centrality[node]
#         }
        
#         # Add topic scores
#         for topic_col in topic_columns:
#             node_data[topic_col] = author_topic_scores[node][topic_col]
        
#         results.append(node_data)
    
#     metrics_df = pd.DataFrame(results)
    
#     # Get network statistics
#     network_stats = {
#         'num_nodes': G.number_of_nodes(),
#         'num_edges': G.number_of_edges(),
#         'total_collaborations': total_collaborations,
#         'density': nx.density(G),
#         'average_degree': sum(dict(G.degree()).values()) / G.number_of_nodes()
#     }
    
#     return G, metrics_df, network_stats

# def visualize_network(G, special_nodes):
#     plt.figure(figsize=(8, 8))
    
#     # Calculate layout with more space for labels
#     pos = nx.spring_layout(G, k=10, iterations=150)
    
#     # Get edge weights for scaling
#     weights = [G[u][v]['weight'] for u, v in G.edges()]
#     max_weight = max(weights)
#     scaled_weights = [0.2 + (w / max_weight) *1.5 for w in weights]
    
#     # Get node sizes based on degree
#     degrees = dict(G.degree())
#     max_degree = max(degrees.values())
#     node_sizes = [(degrees[node] / max_degree) * 50 for node in G.nodes()]  # Scaled down sizes
    
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, 
#                           width=scaled_weights,
#                           alpha=0.3,
#                           edge_color='gray')
    
#     # Draw regular nodes
#     regular_nodes = [node for node in G.nodes() if node not in special_nodes]
#     nx.draw_networkx_nodes(G, pos,
#                           nodelist=regular_nodes,
#                           node_size=[node_sizes[i] for i, node in enumerate(G.nodes()) if node in regular_nodes],
#                           node_color='lightblue',
#                           alpha=0.6)
    
#     # Draw special nodes
#     special_nodes_present = [node for node in special_nodes if node in G.nodes()]
#     if special_nodes_present:
#         nx.draw_networkx_nodes(G, pos,
#                              nodelist=special_nodes_present,
#                              node_size=[node_sizes[i] for i, node in enumerate(G.nodes()) if node in special_nodes_present],
#                              node_color='red',
#                              alpha=0.8)
    
#     # Add labels for all nodes with adjusted position
#     label_pos = {node: (coord[0], coord[1] + 0.02) for node, coord in pos.items()}  # Slightly above nodes
    
#     # Labels for special nodes (larger font)
#     special_labels = {node: node for node in G.nodes() if node in special_nodes}
#     nx.draw_networkx_labels(G, label_pos, special_labels, font_size=6, font_weight='bold')
    
#     # Labels for regular nodes (smaller font)
#     regular_labels = {node: node for node in G.nodes() if node not in special_nodes}
#     nx.draw_networkx_labels(G, label_pos, regular_labels, font_size=6)
    
#     plt.title("Co-authorship Network\nRed nodes: Top Collaborators, Node size: Number of Collaborations", 
#               pad=20, fontsize=16)
#     plt.axis('off')
#     plt.tight_layout()
    
#     # Save the figure with high resolution
#     plt.savefig('coauthor_network.png', dpi=300, bbox_inches='tight')
#     plt.show()
#     plt.close()

# def main(filepath):
#     # Read data
#     df = pd.read_csv(filepath)
    
#     # Create network and calculate metrics
#     G, metrics_df, network_stats = create_coauthor_network(df)
    
#     # Special nodes to highlight
#     special_nodes = ["United Kingdom", "United States", "Norway", "Australia", "Chile"]
    
#     # Create visualization
#     visualize_network(G, special_nodes)
    
#     # Save metrics to CSV
#     metrics_df.to_csv('coauthor_metrics.csv', index=False)
    
#     # Print summary statistics
#     print("\nNetwork Analysis Summary:")
#     print("-" * 40)
#     print(f"Number of authors (nodes): {network_stats['num_nodes']}")
#     print(f"Number of unique collaborations (edges): {network_stats['num_edges']}")
#     print(f"Total number of papers co-authored: {network_stats['total_collaborations']}")
#     print(f"Network density: {network_stats['density']:.4f}")
#     print(f"Average degree: {network_stats['average_degree']:.2f}")
    
#     print("\nTop 5 authors by different centrality measures:")
#     metrics = ['Degree', 'Degree_Centrality', 'Betweenness_Centrality', 
#                'Closeness_Centrality', 'Eigenvector_Centrality']
    
#     for metric in metrics:
#         print(f"\nTop 5 by {metric}:")
#         print(metrics_df.nlargest(5, metric)[['Author', metric]])

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')

# ------------------------------------------------------------------------------------------------ Country Contribution
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator
# from collections import defaultdict

# def get_knee_point(scores):
#     """Calculate knee point for a series of scores"""
#     sorted_scores = np.sort(scores)[::-1]
#     smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#     x = np.arange(len(smoothed_scores))
#     knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
    
#     if knee_locator.knee is not None:
#         knee_index = knee_locator.knee
#         knee_value = smoothed_scores[knee_index]
#         return knee_value
#     else:
#         return sorted_scores.mean() + sorted_scores.std()

# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8  # Fixed threshold for Topic 7 to get 74 papers
#     else:
#         return get_knee_point(scores)

# def analyze_topic_contributions(df):
#     selected_topics = [1, 5, 7, 14]
#     author_topic_contributions = defaultdict(lambda: defaultdict(float))
#     papers_per_topic = defaultdict(int)
    
#     # Process each topic
#     for topic_num in selected_topics:
#         topic_col = f'Topic {topic_num} Score'
        
#         # Get threshold and filter
#         threshold = get_threshold(df[topic_col], topic_num)
#         filtered_df = df[df[topic_col] > threshold]
#         # Remove duplicates based on Title after filtering
#         filtered_df = filtered_df.drop_duplicates(subset=['Title'])
#         papers_per_topic[topic_num] = len(filtered_df)
        
#         # Calculate contributions for filtered papers
#         for _, row in filtered_df.iterrows():
#             authors = row['Submitted By'].split(', ')
#             score = row[topic_col]
            
#             # Distribute score among authors
#             contribution_per_author = score / len(authors)
#             for author in authors:
#                 author_topic_contributions[author][topic_num] += contribution_per_author
    
#     # Create DataFrame from contributions
#     contribution_data = []
#     for author, topic_scores in author_topic_contributions.items():
#         row = {'Author': author}
#         total_contribution = 0
#         for topic_num in selected_topics:
#             score = topic_scores[topic_num]
#             row[f'Topic {topic_num}'] = score
#             total_contribution += score
#         row['Total Contribution'] = total_contribution
#         contribution_data.append(row)
    
#     # Convert to DataFrame and sort by total contribution
#     contribution_df = pd.DataFrame(contribution_data)
#     contribution_df = contribution_df.sort_values('Total Contribution', ascending=False)
    
#     return contribution_df, papers_per_topic

# def create_visualizations(contribution_df, papers_per_topic):
#     selected_topics = [1, 5, 7, 14]
#     top_contributors = contribution_df.nlargest(20, 'Total Contribution')
    
#     # Create heatmap
#     plt.figure(figsize=(15, 10))
    
#     # Prepare heatmap data
#     heatmap_data = top_contributors[[col for col in top_contributors.columns 
#                                    if col.startswith('Topic')]].values
    
#     # Create heatmap
#     sns.heatmap(heatmap_data,
#                 xticklabels=[f'Topic {num}\n({papers_per_topic[num]} papers)' 
#                             for num in selected_topics],
#                 yticklabels=top_contributors['Author'],
#                 cmap='YlOrRd',
#                 fmt='.2f',
#                 cbar_kws={'label': 'Contribution Score'})
    
#     plt.title('Top 20 Contributors to Selected Topics')
#     plt.ylabel('Authors')
#     plt.xlabel('Topics (with number of papers above threshold)')
    
#     plt.tight_layout()
#     plt.savefig('topic_contributions_heatmap.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Create summary table
#     summary_table = pd.DataFrame({
#         'Topic': [f'Topic {num}' for num in selected_topics],
#         'Papers Above Threshold': [papers_per_topic[num] for num in selected_topics],
#         'Unique Contributors': [len(contribution_df[contribution_df[f'Topic {num}'] > 0]) 
#                               for num in selected_topics],
#         'Top Contributor': [contribution_df.nlargest(1, f'Topic {num}')['Author'].iloc[0] 
#                           for num in selected_topics],
#         'Top Contribution Score': [contribution_df[f'Topic {num}'].max() 
#                                  for num in selected_topics]
#     })
    
#     return summary_table

# def main(filepath):
#     # Read data
#     df = pd.read_csv(filepath)
    
#     # Analyze contributions
#     contribution_df, papers_per_topic = analyze_topic_contributions(df)
    
#     # Create visualizations and summary table
#     summary_table = create_visualizations(contribution_df, papers_per_topic)
    
#     # Save detailed contributions
#     contribution_df.to_csv('topic_contributions_detailed.csv', index=False)
    
#     # Save summary table
#     summary_table.to_csv('topic_contributions_summary.csv', index=False)
    
#     # Print summary statistics
#     print("\nTopic Contribution Summary:")
#     print("-" * 100)
#     print(summary_table.to_string(index=False))
    
#     print("\nTop 10 Overall Contributors:")
#     print("-" * 80)
#     print(contribution_df[['Author', 'Total Contribution']].head(10).to_string(index=False))

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')

#------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from collections import defaultdict
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator

# def get_knee_point(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         # Sort scores in descending order
#         sorted_scores = np.sort(scores)[::-1]
#         # Return the score at index 73 (74th paper)
#         # This will make the threshold include exactly the top 74 papers
#         return sorted_scores[73] if len(sorted_scores) > 73 else sorted_scores[-1]
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()


# def get_conservation_papers(df):
#     """Get papers above knee point for conservation topics"""
#     selected_topics = [1, 5, 7, 14]
#     filtered_papers = set()
    
#     for topic_num in selected_topics:
#         topic_col = f'Topic {topic_num} Score'
#         knee_value = get_knee_point(df[topic_col], topic_num)
#         filtered_indices = df[df[topic_col] > knee_value].index
#         filtered_papers.update(filtered_indices)
    
#     return df.loc[list(filtered_papers)]

# def analyze_collaborations(conservation_df):
#     # Create a dictionary to store co-authorship counts
#     coauthor_counts = defaultdict(lambda: defaultdict(int))
#     author_total_collabs = defaultdict(int)
    
#     # Process each paper's authors
#     for _, row in conservation_df.iterrows():
#         authors = [author.strip() for author in row['Submitted By'].split(',')]
        
#         # Count co-authorships
#         for i in range(len(authors)):
#             for j in range(i + 1, len(authors)):
#                 author1 = authors[i]
#                 author2 = authors[j]
#                 coauthor_counts[author1][author2] += 1
#                 coauthor_counts[author2][author1] += 1
#                 author_total_collabs[author1] += 1
#                 author_total_collabs[author2] += 1
    
#     return coauthor_counts, author_total_collabs

# def create_visualizations(coauthor_counts, author_total_collabs):
#     # First visualization: Heatmap (remains the same)
#     # Get top 20 authors by total collaborations
#     top_authors = sorted(author_total_collabs.items(), key=lambda x: x[1], reverse=True)[:20]
#     top_author_names = [author for author, _ in top_authors]
    
#     # Create heatmap data
#     heatmap_data = []
#     for author1 in top_author_names:
#         row_data = []
#         for author2 in top_author_names:
#             if author1 == author2:
#                 row_data.append(0)
#             else:
#                 row_data.append(coauthor_counts[author1][author2])
#         heatmap_data.append(row_data)
    
#     # Create heatmap
#     plt.figure(figsize=(15, 12))
#     sns.heatmap(heatmap_data,
#                 xticklabels=top_author_names,
#                 yticklabels=top_author_names,
#                 cmap='YlOrRd',
#                 annot=True,
#                 fmt='d',
#                 cbar_kws={'label': 'Number of Collaborations'})
    
#     plt.title('Co-authorship Heatmap for Conservation Papers\n(Top 20 Authors)')
#     plt.xticks(rotation=45, ha='right')
#     plt.yticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig('conservation_coauthorship_heatmap.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Second visualization: Modified bar plot with collaborators inside bars
#     plt.figure(figsize=(15, 8))
#     top_10_authors = top_authors[:10]
    
#     y_pos = np.arange(len(top_10_authors))
#     total_collabs = [count for _, count in top_10_authors]
    
#     # Create bars
#     # Create vertical bar chart
#     bars = plt.bar(y_pos, total_collabs, width=0.7)

# # Add author names on x-axis
#     author_labels = [author for author, _ in top_10_authors]
#     plt.xticks(y_pos, author_labels, rotation=45, ha="right")

#     # Add collaborator information inside bars
#     for i, bar in enumerate(bars):
#         author = top_10_authors[i][0]
        
#         # Get top 3 collaborators
#         collaborators = sorted(coauthor_counts[author].items(), key=lambda x: x[1], reverse=True)[:3]
#         collab_text = '\n'.join([f"{c[0]}: {c[1]}" for c in collaborators])
        
#         # Calculate text position
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2, height * 0.5,
#                  collab_text,
#                  ha='center', va='center',
#                  color='white',
#                  fontsize=8,
#                  bbox=dict(facecolor='black', alpha=0.5))
    
#         # Add total collaboration number at the top of each bar
#         plt.text(bar.get_x() + bar.get_width()/2, height + 0.01,
#              f'{int(height)}',
#                 ha='center', va='bottom')

#     # Set labels and title
#     plt.ylabel('Total Number of Collaborations')
#     plt.title('Top 10 Authors by Number of Collaborations in Conservation Papers\nwith Their Top 3 Collaborators')

#     # Add padding to y-axis for the numbers above bars
#     plt.margins(y=0.1)

#     plt.tight_layout()
#     plt.savefig('conservation_collaboration_barplot.png', dpi=300, bbox_inches='tight')
#     plt.close()


# # Rest of the code remains the same
# def main(filepath):
#     # Read data
#     df = pd.read_csv(filepath)
    
#     # Get conservation papers
#     conservation_df = get_conservation_papers(df)
#     print(f"\nTotal conservation papers after knee filtering: {len(conservation_df)}")
    
#     # Analyze collaborations
#     coauthor_counts, author_total_collabs = analyze_collaborations(conservation_df)
    
#     # Create visualizations
#     create_visualizations(coauthor_counts, author_total_collabs)
    
#     # Print top collaboration statistics
#     print("\nTop 10 Authors by Number of Collaborations:")
#     print("-" * 60)
#     for author, collabs in sorted(author_total_collabs.items(), key=lambda x: x[1], reverse=True)[:10]:
#         print(f"\n{author} ({collabs} total collaborations)")
#         print("Top 3 collaborators:")
#         top_collaborators = sorted(coauthor_counts[author].items(), key=lambda x: x[1], reverse=True)[:3]
#         for collaborator, count in top_collaborators:
#             print(f"  - {collaborator}: {count} papers")

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')

#------------------------------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator

# def get_knee_point(scores):
#     """Calculate knee point for a series of scores"""
#     sorted_scores = np.sort(scores)[::-1]
#     smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#     x = np.arange(len(smoothed_scores))
#     knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
    
#     if knee_locator.knee is not None:
#         knee_index = knee_locator.knee
#         knee_value = smoothed_scores[knee_index]
#         return knee_value
#     else:
#         return sorted_scores.mean() + sorted_scores.std()

# def create_topic_network(df, topic_num):
#     """Create network for a specific topic using papers above knee point"""
#     topic_col = f'Topic {topic_num} Score'
    
#     # Filter papers above knee point for this topic
#     knee_value = get_knee_point(df[topic_col])
#     filtered_df = df[df[topic_col] > knee_value]
    
#     # Create a dictionary to store co-authorship topic score sums
#     coauthor_topic_sums = defaultdict(lambda: defaultdict(float))
#     author_topic_scores = defaultdict(float)
    
#     # Process each paper's authors and their topic scores
#     for _, row in filtered_df.iterrows():
#         authors = [author.strip() for author in row['Submitted By'].split(',')]
#         topic_score = row[topic_col]
        
#         # Sum topic scores for each author
#         for author in authors:
#             author_topic_scores[author] += topic_score
        
#         # Calculate edge weights as sum of topic scores for co-authored papers
#         for i in range(len(authors)):
#             for j in range(i + 1, len(authors)):
#                 author1 = authors[i]
#                 author2 = authors[j]
#                 coauthor_topic_sums[author1][author2] += topic_score
#                 coauthor_topic_sums[author2][author1] += topic_score
    
#     # Create NetworkX graph
#     G = nx.Graph()
    
#     # Add edges with weights
#     for author1 in coauthor_topic_sums:
#         for author2, weight in coauthor_topic_sums[author1].items():
#             if weight > 0:
#                 G.add_edge(author1, author2, weight=weight)
    
#     # Calculate centrality metrics
#     metrics = {}
#     if len(G.nodes()) > 0:  # Only calculate if graph has nodes
#         metrics['degree_centrality'] = nx.degree_centrality(G)
#         metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
#         metrics['closeness_centrality'] = nx.closeness_centrality(G)
#         metrics['eigenvector_centrality'] = nx.eigenvector_centrality(G, max_iter=1000)
    
#         # Create results DataFrame
#         results = []
#         for node in G.nodes():
#             results.append({
#                 'Author': node,
#                 'Topic_Score': author_topic_scores[node],
#                 'Degree': G.degree(node),
#                 'Degree_Centrality': metrics['degree_centrality'][node],
#                 'Betweenness_Centrality': metrics['betweenness_centrality'][node],
#                 'Closeness_Centrality': metrics['closeness_centrality'][node],
#                 'Eigenvector_Centrality': metrics['eigenvector_centrality'][node]
#             })
        
#         metrics_df = pd.DataFrame(results)
#     else:
#         metrics_df = pd.DataFrame()
    
#     return G, metrics_df, len(filtered_df)

# def visualize_topic_network(G, topic_num, special_nodes):
#     plt.figure(figsize=(8, 8))
    
#     # Calculate layout
#     pos = nx.spring_layout(G, k=10, iterations=150)
    
#     # Get edge weights for scaling
#     weights = [G[u][v]['weight'] for u, v in G.edges()]
#     max_weight = max(weights) if weights else 1
#     scaled_weights = [0.2 + (w / max_weight) * 1.5 for w in weights]
    
#     # Get node sizes based on degree
#     degrees = dict(G.degree())
#     max_degree = max(degrees.values()) if degrees else 1
#     node_sizes = [(degrees[node] / max_degree) * 50 for node in G.nodes()]
    
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, 
#                           width=scaled_weights,
#                           alpha=0.3,
#                           edge_color='gray')
    
#     # Draw regular nodes
#     regular_nodes = [node for node in G.nodes() if node not in special_nodes]
#     if regular_nodes:
#         nx.draw_networkx_nodes(G, pos,
#                              nodelist=regular_nodes,
#                              node_size=[node_sizes[i] for i, node in enumerate(G.nodes()) if node in regular_nodes],
#                              node_color='lightblue',
#                              alpha=0.6)
    
#     # Draw special nodes
#     special_nodes_present = [node for node in special_nodes if node in G.nodes()]
#     if special_nodes_present:
#         nx.draw_networkx_nodes(G, pos,
#                              nodelist=special_nodes_present,
#                              node_size=[node_sizes[i] for i, node in enumerate(G.nodes()) if node in special_nodes_present],
#                              node_color='red',
#                              alpha=0.8)
    
#     # Add labels
#     label_pos = {node: (coord[0], coord[1] + 0.02) for node, coord in pos.items()}
    
#     # Labels for special nodes
#     special_labels = {node: node for node in G.nodes() if node in special_nodes}
#     nx.draw_networkx_labels(G, label_pos, special_labels, font_size=6, font_weight='bold')
    
#     # Labels for regular nodes
#     regular_labels = {node: node for node in G.nodes() if node not in special_nodes}
#     nx.draw_networkx_labels(G, label_pos, regular_labels, font_size=6)
    
#     plt.title(f"Topic {topic_num} Co-authorship Network\nEdge weights: Sum of topic scores for co-authored papers", 
#               pad=20, fontsize=12)
#     plt.axis('off')
#     plt.tight_layout()
    
#     plt.savefig(f'topic_{topic_num}_network.png', dpi=300, bbox_inches='tight')
#     plt.close()

# def main(filepath):
#     df = pd.read_csv(filepath)
#     special_nodes = ["United Kingdom", "United States", "Norway", "Australia", "Chile"]
#     conservation_topics = [1, 5, 7, 14]
    
#     for topic_num in conservation_topics:
#         # Create and analyze network for this topic
#         G, metrics_df, num_papers = create_topic_network(df, topic_num)
        
#         # Save metrics
#         metrics_df.to_csv(f'topic_{topic_num}_metrics.csv', index=False)
        
#         # Create visualization
#         visualize_topic_network(G, topic_num, special_nodes)
        
#         # Print summary statistics
#         print(f"\nTopic {topic_num} Network Analysis Summary:")
#         print("-" * 40)
#         print(f"Number of papers above knee point: {num_papers}")
#         print(f"Number of authors (nodes): {len(G.nodes())}")
#         print(f"Number of collaborations (edges): {len(G.edges())}")
#         print(f"Network density: {nx.density(G):.4f}")
        
#         if len(metrics_df) > 0:
#             print("\nTop 5 authors by different measures:")
#             measures = ['Topic_Score', 'Degree', 'Degree_Centrality', 
#                        'Betweenness_Centrality', 'Closeness_Centrality', 
#                        'Eigenvector_Centrality']
            
#             for measure in measures:
#                 print(f"\nTop 5 by {measure}:")
#                 print(metrics_df.nlargest(5, measure)[['Author', measure]].to_string(index=False))
        
#         print("\n" + "="*50 + "\n")

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')

# --------------------------------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def create_topic_network(df, topic_num):
#     """Create network for a specific topic using all papers with weighted metrics"""
#     topic_col = f'Topic {topic_num} Score'
    
#     # Create a dictionary to store co-authorship topic score sums
#     coauthor_topic_sums = defaultdict(lambda: defaultdict(float))
#     author_topic_scores = defaultdict(float)
    
#     # Process each paper's authors and their topic scores
#     for _, row in df.iterrows():
#         authors = [author.strip() for author in row['Submitted By'].split(',')]
#         topic_score = row[topic_col]
        
#         # Sum topic scores for each author
#         for author in authors:
#             author_topic_scores[author] += topic_score
        
#         # Calculate edge weights as sum of topic scores for co-authored papers
#         for i in range(len(authors)):
#             for j in range(i + 1, len(authors)):
#                 author1 = authors[i]
#                 author2 = authors[j]
#                 coauthor_topic_sums[author1][author2] += topic_score
#                 coauthor_topic_sums[author2][author1] += topic_score
    
#     # Create NetworkX graph
#     G = nx.Graph()
    
#     # Add edges with weights
#     for author1 in coauthor_topic_sums:
#         for author2, weight in coauthor_topic_sums[author1].items():
#             if weight > 0:
#                 G.add_edge(author1, author2, weight=weight)
    
#     # Calculate weighted centrality metrics
#     metrics = {}
    
#     # Degree centrality using weights
#     degree_dict = dict(G.degree(weight='weight'))
#     max_degree = max(degree_dict.values()) if degree_dict else 1
#     metrics['degree_centrality'] = {node: deg/max_degree for node, deg in degree_dict.items()}
    
#     # Other centrality metrics with weights
#     try:
#         metrics['betweenness_centrality'] = nx.betweenness_centrality(G, weight='weight')
#         metrics['closeness_centrality'] = nx.closeness_centrality(G, distance='weight')
#         metrics['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(G, weight='weight')
#     except:
#         # Fallback if weighted calculation fails
#         print(f"Warning: Some weighted centrality calculations failed for Topic {topic_num}")
#         metrics['betweenness_centrality'] = nx.betweenness_centrality(G)
#         metrics['closeness_centrality'] = nx.closeness_centrality(G)
#         metrics['eigenvector_centrality'] = nx.eigenvector_centrality_numpy(G)
    
#     # Create results DataFrame
#     results = []
#     for node in G.nodes():
#         results.append({
#             'Author': node,
#             'Topic_Score': author_topic_scores[node],
#             'Weighted_Degree': degree_dict[node],
#             'Degree_Centrality': metrics['degree_centrality'][node],
#             'Betweenness_Centrality': metrics['betweenness_centrality'][node],
#             'Closeness_Centrality': metrics['closeness_centrality'][node],
#             'Eigenvector_Centrality': metrics['eigenvector_centrality'][node]
#         })
    
#     metrics_df = pd.DataFrame(results)
#     return G, metrics_df

# def visualize_topic_network(G, topic_num, special_nodes):
#     plt.figure(figsize=(8, 8))
    
#     # Calculate layout
#     pos = nx.spring_layout(G, k=3, iterations=150)
    
#     # Get edge weights for scaling
#     weights = [G[u][v]['weight'] for u, v in G.edges()]
#     max_weight = max(weights) if weights else 1
#     scaled_weights = [0.2 + (w / max_weight) * 1.5 for w in weights]
    
#     # Get node sizes based on degree
#     degrees = dict(G.degree())
#     max_degree = max(degrees.values()) if degrees else 1
#     node_sizes = [(degrees[node] / max_degree) * 50 for node in G.nodes()]
    
#     # Draw edges
#     nx.draw_networkx_edges(G, pos, 
#                           width=scaled_weights,
#                           alpha=0.3,
#                           edge_color='gray')
    
#     # Draw regular nodes
#     regular_nodes = [node for node in G.nodes() if node not in special_nodes]
#     nx.draw_networkx_nodes(G, pos,
#                           nodelist=regular_nodes,
#                           node_size=[node_sizes[i] for i, node in enumerate(G.nodes()) if node in regular_nodes],
#                           node_color='lightblue',
#                           alpha=0.6)
    
#     # Draw special nodes
#     special_nodes_present = [node for node in special_nodes if node in G.nodes()]
#     if special_nodes_present:
#         nx.draw_networkx_nodes(G, pos,
#                              nodelist=special_nodes_present,
#                              node_size=[node_sizes[i] for i, node in enumerate(G.nodes()) if node in special_nodes_present],
#                              node_color='red',
#                              alpha=0.8)
    
#     # Add labels
#     label_pos = {node: (coord[0], coord[1] + 0.02) for node, coord in pos.items()}
    
#     # Labels for special nodes
#     special_labels = {node: node for node in G.nodes() if node in special_nodes}
#     nx.draw_networkx_labels(G, label_pos, special_labels, font_size=6, font_weight='bold')
    
#     # Labels for regular nodes
#     regular_labels = {node: node for node in G.nodes() if node not in special_nodes}
#     nx.draw_networkx_labels(G, label_pos, regular_labels, font_size=6)
    
#     plt.title(f"Topic {topic_num} Co-authorship Network\nEdge weights: Sum of topic scores for co-authored papers", 
#               pad=20, fontsize=12)
#     plt.axis('off')
#     plt.tight_layout()
    
#     plt.savefig(f'topic_{topic_num}_network.png', dpi=300, bbox_inches='tight')
#     plt.close()

# def main(filepath):
#     df = pd.read_csv(filepath)
#     special_nodes = ["United Kingdom", "United States", "Norway", "Australia", "Chile"]
#     conservation_topics = [1, 5, 7, 14]
    
#     for topic_num in conservation_topics:
#         # Create and analyze network for this topic
#         G, metrics_df = create_topic_network(df, topic_num)
        
#         # Save metrics
#         metrics_df.to_csv(f'topic_{topic_num}_metrics.csv', index=False)
        
#         # Create visualization
#         visualize_topic_network(G, topic_num, special_nodes)
        
#         # Print summary statistics
#         print(f"\nTopic {topic_num} Network Analysis Summary:")
#         print("-" * 40)
#         print(f"Number of authors (nodes): {len(G.nodes())}")
#         print(f"Number of collaborations (edges): {len(G.edges())}")
#         print(f"Network density: {nx.density(G):.4f}")
#         print(f"Total topic score weight: {sum([d['weight'] for (u,v,d) in G.edges(data=True)]):.2f}")
        
#         if len(metrics_df) > 0:
#             print("\nTop 5 authors by different measures:")
#             measures = ['Topic_Score', 'Weighted_Degree', 'Degree_Centrality', 
#                        'Betweenness_Centrality', 'Closeness_Centrality', 
#                        'Eigenvector_Centrality']
            
#             for measure in measures:
#                 print(f"\nTop 5 by {measure}:")
#                 print(metrics_df.nlargest(5, measure)[['Author', measure]].to_string(index=False))
        
#         # Print edge weights for special nodes
#         print("\nCollaboration scores between special nodes (if any):")
#         special_nodes_present = [node for node in special_nodes if node in G.nodes()]
#         for i in range(len(special_nodes_present)):
#             for j in range(i + 1, len(special_nodes_present)):
#                 node1 = special_nodes_present[i]
#                 node2 = special_nodes_present[j]
#                 if G.has_edge(node1, node2):
#                     weight = G[node1][node2]['weight']
#                     print(f"{node1} - {node2}: {weight:.2f}")
        
#         print("\n" + "="*50 + "\n")

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')

#------------------------------------------------------------------------------------------------------------

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns

# def load_and_prepare_data(main_metrics_file, topic_metrics_file):
#     # Load both metric files
#     main_df = pd.read_csv(main_metrics_file)
#     topic_df = pd.read_csv(topic_metrics_file)
    
#     # Merge dataframes
#     merged_df = pd.merge(main_df, topic_df, 
#                         on='Author', 
#                         suffixes=('_main', '_topic'))
    
#     return merged_df, main_df, topic_df

# def calculate_metric_differences(merged_df, metrics):
#     differences = {}
#     for metric in metrics:
#         differences[metric] = merged_df[f'{metric}_topic'] - merged_df[f'{metric}_main']
    
#     return pd.DataFrame(differences)

# def plot_metric_differences(merged_df, topic_num):
#     # Metrics to compare
#     metrics = ['Degree_Centrality', 'Betweenness_Centrality', 
#               'Closeness_Centrality', 'Eigenvector_Centrality']
    
#     # Calculate differences
#     diff_df = calculate_metric_differences(merged_df, metrics)
    
#     # Create subplot for each metric
#     fig, axes = plt.subplots(len(metrics), 1, figsize=(15, 4*len(metrics)))
#     fig.suptitle(f'Metric Differences between Main Network and Topic {topic_num} Network', 
#                  fontsize=16, y=1.02)
    
#     for idx, metric in enumerate(metrics):
#         # Sort differences by absolute value
#         diff_series = diff_df[metric].sort_values(key=abs, ascending=False)
        
#         # Get top 20 differences by magnitude and their authors
#         top_diff = diff_series.head(20)
#         authors = merged_df.loc[top_diff.index, 'Author']
        
#         # Create bar plot
#         bars = axes[idx].bar(authors, top_diff)
        
#         # Add labels
#         axes[idx].set_title(f'{metric} Difference', pad=10)
#         axes[idx].set_xticks(range(len(authors)))
#         axes[idx].set_xticklabels(authors, rotation=45, ha='right')
#         axes[idx].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
#         # Color bars based on positive/negative
#         for bar in bars:
#             if bar.get_height() < 0:
#                 bar.set_color('red')
#             else:
#                 bar.set_color('green')
        
#         # Add value labels
#         for i, v in enumerate(top_diff):
#             axes[idx].text(i, v + (0.01 if v >= 0 else -0.01),
#                          f'{v:.3f}',
#                          ha='center', va='bottom' if v >= 0 else 'top',
#                          fontsize=8)
        
#         # Adjust layout for better readability
#         axes[idx].tick_params(axis='x', labelsize=8)
#         axes[idx].set_xlabel('Authors', fontsize=10)
#         axes[idx].set_ylabel('Difference', fontsize=10)
        
#         # Add grid for better readability
#         axes[idx].grid(True, axis='y', linestyle='--', alpha=0.3)
    
#     plt.tight_layout()
#     plt.savefig(f'topic_{topic_num}_metric_differences.png', dpi=300, bbox_inches='tight')
#     plt.close()

# def plot_topic_proportion(main_df, topic_num):
#     # Calculate total topic score and specific topic proportion
#     topic_cols = [col for col in main_df.columns if 'Topic' in col and 'Score' in col]
#     main_df['Total_Topic_Score'] = main_df[topic_cols].sum(axis=1)
#     main_df[f'Topic_{topic_num}_Proportion'] = main_df[f'Topic {topic_num} Score'] / main_df['Total_Topic_Score']
    
#     # Sort by proportion
#     top_authors = main_df.nlargest(20, f'Topic_{topic_num}_Proportion')
    
#     # Create visualization
#     plt.figure(figsize=(15, 8))
    
#     # Create bar plot
#     bars = plt.bar(range(len(top_authors)), 
#                   top_authors[f'Topic_{topic_num}_Proportion'] * 100)  # Convert to percentage
    
#     # Customize plot
#     plt.title(f'Top 20 Authors by Topic {topic_num} Score Proportion', pad=20)
#     plt.xlabel('Authors')
#     plt.ylabel('Percentage of Total Topic Score (%)')
    
#     # Add author labels
#     plt.xticks(range(len(top_authors)), top_authors['Author'], rotation=45, ha='right')
    
#     # Add percentage labels on bars
#     for i, bar in enumerate(bars):
#         height = bar.get_height()
#         plt.text(bar.get_x() + bar.get_width()/2., height,
#                 f'{height:.1f}%',
#                 ha='center', va='bottom')
    
#     plt.tight_layout()
#     plt.savefig(f'topic_{topic_num}_proportion.png', dpi=300, bbox_inches='tight')
#     plt.close()

# def main():
#     topic_nums = [1, 5, 7, 14]
    
#     for topic_num in topic_nums:
#         print(f"\nProcessing Topic {topic_num}...")
        
#         # Load data
#         merged_df, main_df, topic_df = load_and_prepare_data(
#             'coauthor_metrics.csv',
#             f'topic_{topic_num}_metrics.csv'
#         )
        
#         # Create difference plots
#         plot_metric_differences(merged_df, topic_num)
        
#         # Create topic proportion plot
#         plot_topic_proportion(main_df, topic_num)
        
#         # Print summary statistics
#         print(f"\nTop 10 Metric Changes for Topic {topic_num}:")
#         metrics = ['Degree_Centrality', 'Betweenness_Centrality', 
#                   'Closeness_Centrality', 'Eigenvector_Centrality']
        
#         for metric in metrics:
#             diff = merged_df[f'{metric}_topic'] - merged_df[f'{metric}_main']
#             print(f"\nLargest changes in {metric}:")
#             print(diff.nlargest(10).to_string())

# if __name__ == "__main__":
#     main()
# ---------------------------------------------- report
# import pandas as pd
# import numpy as np
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator
# from docx import Document
# from docx.shared import Inches

# def process_data(filepath):
#     """Load and process the CSV data"""
#     df = pd.read_csv(filepath)
#     return df

# def get_knee_point(scores, sigma=1):
#     """Calculate knee point for a series of scores with configurable smoothing"""
#     sorted_scores = np.sort(scores)[::-1]
#     smoothed_scores = gaussian_filter1d(sorted_scores, sigma=sigma)
#     x = np.arange(len(smoothed_scores))
#     knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
    
#     if knee_locator.knee is not None:
#         knee_index = knee_locator.knee
#         knee_value = smoothed_scores[knee_index]
#         return knee_value
#     else:
#         # Fallback: use mean + 1 std if no knee is found
#         return sorted_scores.mean() + sorted_scores.std()

# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8  # Fixed threshold for Topic 7
#     elif topic_num == 11:
#         return 0.5
#     else:
#         return get_knee_point(scores, sigma=1)

# def create_topic_report(df, topic_num):
#     """Generate a docx report for a specific topic"""
#     topic_col = f'Topic {topic_num} Score'
    
#     # Get threshold value
#     threshold = get_threshold(df[topic_col], topic_num)
    
#     # Filter papers above threshold and sort by score
#     filtered_df = df[df[topic_col] > threshold].copy()
#     filtered_df = filtered_df.sort_values(by=topic_col, ascending=False)
    
#     # Remove duplicates based on Title
#     filtered_df = filtered_df.drop_duplicates(subset=['Title'])
    
#     # Create document
#     doc = Document()
#     doc.add_heading(f'Topic {topic_num} - Top Papers Report', 0)
    
#     # Add summary statistics
#     doc.add_paragraph(f'Total papers analyzed: {len(df)}')
#     doc.add_paragraph(f'Papers above threshold: {len(filtered_df)}')
#     doc.add_paragraph(f'Threshold value: {threshold:.4f}')
#     if topic_num == 7 or topic_num == 11:
#         doc.add_paragraph(f'Note: Using fixed threshold of 0.8 for Topic {topic_num}')
    
#     # Add table
#     table = doc.add_table(rows=1, cols=5)  # Increased to 5 columns
#     table.style = 'Table Grid'
    
#     # Set header
#     header_cells = table.rows[0].cells
#     headers = ['Title', 'Year', 'Type', 'Submitted By', f'Topic {topic_num} Score']  # Added 'Type'
#     for i, header in enumerate(headers):
#         header_cells[i].text = header
    
#     # Add data rows
#     for _, row in filtered_df.iterrows():
#         row_cells = table.add_row().cells
#         row_cells[0].text = str(row['Title'])
#         row_cells[1].text = str(row['Year'])
#         row_cells[2].text = str(row['Type'])  # Added Type column
#         row_cells[3].text = str(row['Submitted By'])
#         row_cells[4].text = f"{row[topic_col]:.4f}"
    
#     # Adjust column widths
#     widths = [5, 1, 1.5, 2, 1.5]  # Updated proportional widths
#     for i, width in enumerate(widths):
#         for cell in table.columns[i].cells:
#             cell.width = Inches(width)
    
#     # Save document
#     filename = f'topic_{topic_num}_report.docx'
#     doc.save(filename)
#     print(f"Generated report: {filename}")
#     return len(filtered_df)

# def main(filepath):
#     df = process_data(filepath)
#     green_topics = {1, 5, 7, 14}
    
#     print("\nGenerating reports for selected topics:")
#     print("-" * 60)
#     print(f"{'Topic':^10} {'Total Papers':^15} {'Selected Papers':^15}")
#     print("-" * 60)
    
#     for topic_num in range(20):
#         selected_papers = create_topic_report(df, topic_num)
#         total_papers = len(df)
#         print(f"{topic_num:^10} {total_papers:^15} {selected_papers:^15}")
    
#     print("-" * 60)

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')
# --------------------------------------------------------------------Topic-specific kneed networks
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator
# from collections import defaultdict

# def process_data(filepath):
#     """Load and process the CSV data"""
#     df = pd.read_csv(filepath)
#     return df

# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()

# def create_collaboration_network(df, topic_num):
#     """Create and visualize collaboration network for a specific topic"""
#     topic_col = f'Topic {topic_num} Score'
    
#     # Get threshold and filter papers
#     threshold = get_threshold(df[topic_col], topic_num)
#     filtered_df = df[df[topic_col] > threshold].copy()
    
#     # Remove duplicates based on Title
#     filtered_df = filtered_df.drop_duplicates(subset=['Title'])
    
#     # Initialize network
#     G = nx.Graph()
    
#     # Track author topic scores and collaboration weights
#     author_topic_scores = defaultdict(float)
#     collaboration_weights = defaultdict(float)
#     collaboration_counts = defaultdict(int)
    
#     # Process each paper to calculate author topic scores
#     for _, row in filtered_df.iterrows():
#         submitters = row['Submitted By'].split(', ')
#         topic_score = row[topic_col]
        
#         # Add topic score for each submitter
#         for submitter in submitters:
#             author_topic_scores[submitter] += topic_score
        
#         # Add collaboration edges if multiple submitters
#         if len(submitters) > 1:
#             for i in range(len(submitters)):
#                 for j in range(i + 1, len(submitters)):
#                     author1, author2 = submitters[i], submitters[j]
#                     # Edge weight is sum of both authors' topic scores
#                     collaboration_weights[(author1, author2)] = (
#                         author_topic_scores[author1] + author_topic_scores[author2]
#                     )
#                     collaboration_counts[(author1, author2)] += 1
    
#     # Add all authors as nodes with their topic scores
#     for author, score in author_topic_scores.items():
#         G.add_node(author, topic_score=score)
    
#     # Add weighted edges for collaborations
#     for (author1, author2), weight in collaboration_weights.items():
#         G.add_edge(author1, author2, weight=weight, count=collaboration_counts[(author1, author2)])
    
#     # Calculate node sizes based on topic scores
#     node_sizes = [G.nodes[node]['topic_score'] * 300 for node in G.nodes()]
    
#     # Calculate edge widths based on weights
#     edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
#     max_weight = max(edge_weights) if edge_weights else 1
#     edge_widths = [2 * weight/max_weight for weight in edge_weights]
    
#     # Create figure
#     plt.figure(figsize=(15, 10))
    
#     # Set random seed for layout reproducibility
#     pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
#     # Draw network
#     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
#                           node_color='lightblue', alpha=0.6)
#     nx.draw_networkx_edges(G, pos, width=edge_widths, 
#                           alpha=0.5, edge_color='gray')
#     nx.draw_networkx_labels(G, pos, font_size=8)
    
#     # Add title and info
#     plt.title(f'Topic {topic_num} Author Network\n'
#               f'Papers above threshold: {len(filtered_df)}, '
#               f'Authors: {len(G.nodes)}, '
#               f'Collaborations: {len(G.edges)}')
    
#     # Save plot
#     plt.savefig(f'topic_{topic_num}_author_network.png', 
#                 dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Print network statistics
#     print(f"\nTopic {topic_num} Network Statistics:")
#     print(f"Number of papers above threshold: {len(filtered_df)}")
#     print(f"Number of authors: {len(G.nodes)}")
#     print(f"Number of collaboration edges: {len(G.edges)}")
#     print(f"Authors with solo papers only: {sum(1 for node in G.nodes() if G.degree(node) == 0)}")
#     if len(G.edges) > 0:
#         print(f"Average collaborations per collaborative author: {2*len(G.edges)/(len(G.nodes)-sum(1 for node in G.nodes() if G.degree(node) == 0)):.2f}")
    
#     # Calculate centrality metrics
#     degree_centrality = nx.degree_centrality(G)
#     betweenness_centrality = nx.betweenness_centrality(G)
#     closeness_centrality = nx.closeness_centrality(G)
#     eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
#     # Calculate weighted degree (sum of edge weights) and degree (count of edges)
#     weighted_degree = {node: sum(d['weight'] for _, _, d in G.edges(node, data=True)) for node in G.nodes()}
#     degree = {node: sum(1 for _ in G.edges(node)) for node in G.nodes()}
    
#     # Create dataframe with metrics
#     metrics_df = pd.DataFrame({
#         'Author': list(G.nodes()),
#         'Topic_Score': [G.nodes[node]['topic_score'] for node in G.nodes()],
#         'Weighted_Degree': [weighted_degree.get(node, 0) for node in G.nodes()],
#         'Degree': [degree.get(node, 0) for node in G.nodes()],
#         'Degree_Centrality': [degree_centrality[node] for node in G.nodes()],
#         'Betweenness_Centrality': [betweenness_centrality[node] for node in G.nodes()],
#         'Closeness_Centrality': [closeness_centrality[node] for node in G.nodes()],
#         'Eigenvector_Centrality': [eigenvector_centrality[node] for node in G.nodes()]
#     })
    
#     # Save metrics to CSV
#     metrics_df.to_csv(f'topic_{topic_num}_network_metrics_kneed.csv', index=False)
    
#     return G

# def main(filepath):
#     df = process_data(filepath)
#     green_topics = {1, 5, 7, 14}
    
#     print("\nGenerating author networks for selected topics:")
#     print("-" * 60)
    
#     networks = {}
#     for topic_num in green_topics:
#         print(f"\nProcessing Topic {topic_num}...")
#         networks[topic_num] = create_collaboration_network(df, topic_num)
    
#     print("\nAll networks generated successfully!")

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')
# ----------------------------------------------------------------------------- Line plot
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator
# from collections import defaultdict

# def process_data(filepath):
#     """Load and process the CSV data"""
#     df = pd.read_csv(filepath)
#     return df

# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8
#     elif topic_num == 11:
#         return 0.5
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()


# def create_topic_trends(df, selected_topics):
#     """Create a line plot showing yearly trends for selected topics"""
#     plt.figure(figsize=(12, 6))
    
#     # Store filtered data for each topic
#     filtered_data = {}
    
#     # Process each topic
#     for topic_num in selected_topics:
#         topic_col = f'Topic {topic_num} Score'
        
#         # Get threshold and filter papers
#         threshold = get_threshold(df[topic_col], topic_num)
#         filtered_df = df[df[topic_col] > threshold].copy()
#         filtered_df = filtered_df.drop_duplicates(subset=['Title'])
        
#         # Calculate yearly sums for this topic
#         yearly_sums = filtered_df.groupby('Year')[topic_col].sum().reset_index()
#         filtered_data[topic_num] = yearly_sums
        
#         # Plot line for this topic
#         plt.plot(yearly_sums['Year'], yearly_sums[topic_col], 
#                 marker='o', label=f'Topic {topic_num}')
    
#     # Customize plot
#     plt.title('Yearly Topic Score Trends\n(Filtered by Knee Detection Threshold)')
#     plt.xlabel('Year')
#     plt.ylabel('Total Topic Score')
#     plt.legend(title='Topic')
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)
    
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
    
#     # Save plot
#     plt.savefig('topic_trends_all.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return filtered_data

# def main(filepath):
#     df = process_data(filepath)
#     green_topics = {1, 5, 7, 14}
    
    
#     # Create topic trends plot
#     print("\nGenerating topic trends plot...")
#     filtered_data = create_topic_trends(df, range(0,20))
    
#     print("\nAll visualizations generated successfully!")
    
#     return filtered_data

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')
# -------------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator

# def process_data(filepath):
#     """Load and process the CSV data"""
#     df = pd.read_csv(filepath)
#     return df

# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8
#     elif topic_num == 11:
#         return 0.5
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()
# def create_individual_topic_trends(df, selected_topics):
#     """Create separate line plots for each topic, with Type as legend"""
#     filtered_data = {}
    
#     # Create a subplot for each topic
#     num_topics = len(selected_topics)
#     fig, axes = plt.subplots(num_topics, 1, figsize=(12, 6*num_topics))
    
#     # Process each topic
#     for idx, topic_num in enumerate(selected_topics):
#         topic_col = f'Topic {topic_num} Score'
#         ax = axes[idx] if num_topics > 1 else axes
        
#         # Get threshold and filter papers
#         threshold = get_threshold(df[topic_col], topic_num)
#         filtered_df = df[df[topic_col] > threshold].copy()
#         filtered_df = filtered_df.drop_duplicates(subset=['Title'])
        
#         # Calculate yearly sums for each Type
#         types = filtered_df['Type'].unique()
#         for type_name in types:
#             type_df = filtered_df[filtered_df['Type'] == type_name]
#             yearly_sums = type_df.groupby('Year')[topic_col].sum().reset_index()
            
#             # Plot line for this type
#             ax.plot(yearly_sums['Year'], yearly_sums[topic_col], 
#                    marker='o', label=f'{type_name}')
            
#             # Store filtered data
#             if topic_num not in filtered_data:
#                 filtered_data[topic_num] = {}
#             filtered_data[topic_num][type_name] = yearly_sums
        
#         # Customize subplot
#         ax.set_title(f'Topic {topic_num} Yearly Score Trends\n(Filtered by Knee Detection Threshold)')
#         ax.set_xlabel('Year')
#         ax.set_ylabel('Total Topic Score')
#         ax.legend(title='Type')
#         ax.grid(True, linestyle='--', alpha=0.7)
        
#         # Rotate x-axis labels for better readability
#         ax.tick_params(axis='x', rotation=45)
    
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
    
#     # Save plot
#     plt.savefig('individual_topic_trends_all.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return filtered_data

# def main(filepath):
#     df = process_data(filepath)
#     green_topics = {1, 5, 7, 14}
    
#     # Create individual topic trends plots
#     print("\nGenerating individual topic trends plots...")
#     filtered_data = create_individual_topic_trends(df, range(0,20))
    
#     print("\nAll visualizations generated successfully!")
    
#     return filtered_data

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')

# -------------------------------------------------------------------------------
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator

# def process_data(filepath):
#     """Load and process the CSV data"""
#     df = pd.read_csv(filepath)
#     return df

# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8
#     elif topic_num == 11:
#         return 0.5
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()

# def explode_submitters(df):
#     """
#     Create separate rows for papers with multiple submitters,
#     duplicating the paper's data for each submitter
#     """
#     # Create a copy of the dataframe
#     df_exploded = df.copy()
    
#     # Split the 'Submitted By' column and explode it
#     df_exploded['Submitted By'] = df_exploded['Submitted By'].str.split(', ')
#     df_exploded = df_exploded.explode('Submitted By')
    
#     return df_exploded

# def create_individual_topic_trends(df, selected_topics):
#     """Create separate line plots for each topic, with Submitted By as legend"""
#     filtered_data = {}
    
#     # Explode the dataframe to handle multiple submitters
#     df_exploded = explode_submitters(df)
    
#     # Create a subplot for each topic
#     num_topics = len(selected_topics)
#     fig, axes = plt.subplots(num_topics, 1, figsize=(15, 8*num_topics))
    
#     # Process each topic
#     for idx, topic_num in enumerate(selected_topics):
#         topic_col = f'Topic {topic_num} Score'
#         ax = axes[idx] if num_topics > 1 else axes
        
#         # Get threshold and filter papers
#         threshold = get_threshold(df[topic_col], topic_num)
#         filtered_df = df_exploded[df_exploded[topic_col] > threshold].copy()
#         filtered_df = filtered_df.drop_duplicates(subset=['Title', 'Submitted By'])
        
#         # Get top submitters by total contribution to avoid overcrowding the plot
#         submitter_totals = filtered_df.groupby('Submitted By')[topic_col].sum()
#         top_submitters = submitter_totals.nlargest(10).index
        
#         # Calculate yearly sums for each submitter
#         for submitter in top_submitters:
#             submitter_df = filtered_df[filtered_df['Submitted By'] == submitter]
#             yearly_sums = submitter_df.groupby('Year')[topic_col].sum().reset_index()
            
#             # Only plot if there are enough data points
#             if len(yearly_sums) > 0:
#                 # Plot line for this submitter
#                 ax.plot(yearly_sums['Year'], yearly_sums[topic_col], 
#                        marker='o', label=f'{submitter}')
                
#                 # Store filtered data
#                 if topic_num not in filtered_data:
#                     filtered_data[topic_num] = {}
#                 filtered_data[topic_num][submitter] = yearly_sums
        
#         # Customize subplot
#         ax.set_title(f'Topic {topic_num} Yearly Score Trends by Top Submitters\n(Filtered by Knee Detection Threshold)')
#         ax.set_xlabel('Year')
#         ax.set_ylabel('Total Topic Score')
#         ax.legend(title='Submitted By', bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax.grid(True, linestyle='--', alpha=0.7)
        
#         # Rotate x-axis labels for better readability
#         ax.tick_params(axis='x', rotation=45)
    
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
    
#     # Save plot
#     plt.savefig('individual_topic_trends_by_submitter_all.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return filtered_data

# def main(filepath):
#     df = process_data(filepath)
#     green_topics = {1, 5, 7, 14}
    
#     # Create individual topic trends plots
#     print("\nGenerating individual topic trends plots by submitter...")
#     filtered_data = create_individual_topic_trends(df, range(0,20))
    
#     print("\nAll visualizations generated successfully!")
    
#     return filtered_data

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')

# ---------------------------------------------------------------------------- Full network metrics(without the knee)
# import pandas as pd
# import numpy as np
# import networkx as nx
# import matplotlib.pyplot as plt
# from collections import defaultdict

# def process_data(filepath):
#     """Load and process the CSV data"""
#     df = pd.read_csv(filepath)
#     return df

# def create_collaboration_network(df, topic_num):
#     """Create and visualize collaboration network for a specific topic"""
#     topic_col = f'Topic {topic_num} Score'
    
#     # Remove duplicates based on Title
#     filtered_df = df.drop_duplicates(subset=['Title'])
    
#     # Initialize network
#     G = nx.Graph()
    
#     # Track author topic scores and edge weights
#     author_topic_scores = defaultdict(float)
#     edge_weights = defaultdict(float)
    
#     # Process each paper to calculate author topic scores and edge weights
#     for _, row in filtered_df.iterrows():
#         submitters = row['Submitted By'].split(', ')
#         topic_score = row[topic_col]
        
#         # Add topic score for each submitter
#         for submitter in submitters:
#             author_topic_scores[submitter] += topic_score
        
#         # Add edges with weights based on topic scores if multiple submitters
#         if len(submitters) > 1:
#             for i in range(len(submitters)):
#                 for j in range(i + 1, len(submitters)):
#                     author1, author2 = submitters[i], submitters[j]
#                     edge_weights[(author1, author2)] += topic_score  # Weight based on topic score
    
#     # Add all authors as nodes with their topic scores
#     for author, score in author_topic_scores.items():
#         G.add_node(author, topic_score=score)
    
#     # Add weighted edges for collaborations
#     for (author1, author2), weight in edge_weights.items():
#         G.add_edge(author1, author2, weight=weight)
    
#     # Calculate node sizes based on topic scores
#     node_sizes = [G.nodes[node]['topic_score'] * 300 for node in G.nodes()]
    
#     # Calculate edge widths based on weights
#     edge_weights_list = [G[u][v]['weight'] for u, v in G.edges()]
#     max_weight = max(edge_weights_list) if edge_weights_list else 1
#     edge_widths = [2 * weight/max_weight for weight in edge_weights_list]
    
#     # Create figure
#     plt.figure(figsize=(15, 10))
    
#     # Set random seed for layout reproducibility
#     pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
    
#     # Draw network
#     nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
#                           node_color='lightblue', alpha=0.6)
#     nx.draw_networkx_edges(G, pos, width=edge_widths, 
#                           alpha=0.5, edge_color='gray')
#     nx.draw_networkx_labels(G, pos, font_size=8)
    
#     # Add title and info
#     plt.title(f'Topic {topic_num} Author Network\n'
#               f'Total Papers: {len(filtered_df)}, '
#               f'Authors: {len(G.nodes)}, '
#               f'Collaborations: {len(G.edges)}')
    
#     # Save plot
#     plt.savefig(f'topic_{topic_num}_author_network.png', 
#                 dpi=300, bbox_inches='tight')
#     plt.close()
    
#     # Print network statistics
#     print(f"\nTopic {topic_num} Network Statistics:")
#     print(f"Total number of papers: {len(filtered_df)}")
#     print(f"Number of authors: {len(G.nodes)}")
#     print(f"Number of collaboration edges: {len(G.edges)}")
#     print(f"Authors with solo papers only: {sum(1 for node in G.nodes() if G.degree(node) == 0)}")
#     if len(G.edges) > 0:
#         print(f"Average collaborations per collaborative author: {2*len(G.edges)/(len(G.nodes)-sum(1 for node in G.nodes() if G.degree(node) == 0)):.2f}")
    
#     # Calculate centrality metrics
#     degree_centrality = nx.degree_centrality(G)
#     betweenness_centrality = nx.betweenness_centrality(G)
#     closeness_centrality = nx.closeness_centrality(G)
#     eigenvector_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
#     # Calculate weighted degree (sum of edge weights) and degree (count of edges)
#     weighted_degree = {node: sum(d['weight'] for _, _, d in G.edges(node, data=True)) for node in G.nodes()}
#     degree = {node: sum(1 for _ in G.edges(node)) for node in G.nodes()}
    
#     # Create dataframe with metrics
#     metrics_df = pd.DataFrame({
#         'Author': list(G.nodes()),
#         'Topic_Score': [G.nodes[node]['topic_score'] for node in G.nodes()],
#         'Weighted_Degree': [weighted_degree.get(node, 0) for node in G.nodes()],
#         'Degree': [degree.get(node, 0) for node in G.nodes()],
#         'Degree_Centrality': [degree_centrality[node] for node in G.nodes()],
#         'Betweenness_Centrality': [betweenness_centrality[node] for node in G.nodes()],
#         'Closeness_Centrality': [closeness_centrality[node] for node in G.nodes()],
#         'Eigenvector_Centrality': [eigenvector_centrality[node] for node in G.nodes()]
#     })
    
#     # Save metrics to CSV
#     metrics_df.to_csv(f'topic_{topic_num}_network_metrics_full.csv', index=False)
    
#     return G

# def main(filepath):
#     df = process_data(filepath)
#     topics = {1, 5, 7, 14}  # Changed to include topic 11 instead of 14
    
#     print("\nGenerating author networks for selected topics:")
#     print("-" * 60)
    
#     networks = {}
#     for topic_num in topics:
#         print(f"\nProcessing Topic {topic_num}...")
#         networks[topic_num] = create_collaboration_network(df, topic_num)
    
#     print("\nAll networks generated successfully!")

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')

# ----------------------------------------------------------------------------------- Line plot counts
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator

# def process_data(filepath):
#     """Load and process the CSV data"""
#     df = pd.read_csv(filepath)
#     return df

# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8
#     elif topic_num == 11:
#         return 0.5
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()

# def create_topic_trends(df, selected_topics):
#     """Create a line plot showing yearly paper counts for selected topics"""
#     plt.figure(figsize=(12, 6))
    
#     # Store filtered data for each topic
#     filtered_data = {}
    
#     # Process each topic
#     for topic_num in selected_topics:
#         topic_col = f'Topic {topic_num} Score'
        
#         # Get threshold and filter papers
#         threshold = get_threshold(df[topic_col], topic_num)
#         filtered_df = df[df[topic_col] > threshold].copy()
#         filtered_df = filtered_df.drop_duplicates(subset=['Title'])
        
#         # Calculate yearly paper counts for this topic
#         yearly_counts = filtered_df.groupby('Year').size().reset_index(name='count')
#         filtered_data[topic_num] = yearly_counts
        
#         # Plot line for this topic
#         plt.plot(yearly_counts['Year'], yearly_counts['count'], 
#                 marker='o', label=f'Topic {topic_num}')
    
#     # Customize plot
#     plt.title('Yearly Number of Papers by Topic\n(Filtered by Knee Detection Threshold)')
#     plt.xlabel('Year')
#     plt.ylabel('Number of Papers')
#     plt.legend(title='Topic')
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     # Rotate x-axis labels for better readability
#     plt.xticks(rotation=45)
    
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
    
#     # Save plot
#     plt.savefig('topic_trends_paper_counts_all.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return filtered_data

# def main(filepath):
#     df = process_data(filepath)
#     green_topics = {1, 5, 7, 14}
    
#     # Create topic trends plot
#     print("\nGenerating topic trends plot...")
#     filtered_data = create_topic_trends(df, range(0,20))
    
#     print("\nAll visualizations generated successfully!")
    
#     return filtered_data

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator

# def process_data(filepath):
#     """Load and process the CSV data"""
#     df = pd.read_csv(filepath)
#     return df

# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8
#     elif topic_num == 11:
#         return 0.5
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()
# def create_individual_topic_trends(df, selected_topics):
#     """Create separate line plots for each topic, with Type as legend, showing paper counts"""
#     filtered_data = {}
    
#     # Create a subplot for each topic
#     num_topics = len(selected_topics)
#     fig, axes = plt.subplots(num_topics, 1, figsize=(12, 6*num_topics))
    
#     # Process each topic
#     for idx, topic_num in enumerate(selected_topics):
#         topic_col = f'Topic {topic_num} Score'
#         ax = axes[idx] if num_topics > 1 else axes
        
#         # Get threshold and filter papers
#         threshold = get_threshold(df[topic_col], topic_num)
#         filtered_df = df[df[topic_col] > threshold].copy()
#         filtered_df = filtered_df.drop_duplicates(subset=['Title'])
        
#         # Calculate yearly counts for each Type
#         types = filtered_df['Type'].unique()
#         for type_name in types:
#             type_df = filtered_df[filtered_df['Type'] == type_name]
#             yearly_counts = type_df.groupby('Year').size().reset_index(name='count')
            
#             # Plot line for this type
#             ax.plot(yearly_counts['Year'], yearly_counts['count'], 
#                    marker='o', label=f'{type_name}')
            
#             # Store filtered data
#             if topic_num not in filtered_data:
#                 filtered_data[topic_num] = {}
#             filtered_data[topic_num][type_name] = yearly_counts
        
#         # Customize subplot
#         ax.set_title(f'Topic {topic_num} Yearly Paper Counts by Type\n(Filtered by Knee Detection Threshold)')
#         ax.set_xlabel('Year')
#         ax.set_ylabel('Number of Papers')
#         ax.legend(title='Type')
#         ax.grid(True, linestyle='--', alpha=0.7)
        
#         # Rotate x-axis labels for better readability
#         ax.tick_params(axis='x', rotation=45)
    
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
    
#     # Save plot
#     plt.savefig('individual_topic_trends_paper_counts_all.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return filtered_data

# def main(filepath):
#     df = process_data(filepath)
#     green_topics = {1, 5, 7, 14}
    
#     # Create individual topic trends plots
#     print("\nGenerating individual topic trends plots...")
#     filtered_data = create_individual_topic_trends(df, range(0,20))
    
#     print("\nAll visualizations generated successfully!")
    
#     return filtered_data

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.ndimage import gaussian_filter1d
# from kneed import KneeLocator

# def process_data(filepath):
#     """Load and process the CSV data"""
#     df = pd.read_csv(filepath)
#     return df

# def get_threshold(scores, topic_num):
#     """Get threshold value based on topic number"""
#     if topic_num == 7:
#         return 0.8
#     elif topic_num == 11:
#         return 0.5
#     else:
#         sorted_scores = np.sort(scores)[::-1]
#         smoothed_scores = gaussian_filter1d(sorted_scores, sigma=1)
#         x = np.arange(len(smoothed_scores))
#         knee_locator = KneeLocator(x, smoothed_scores, curve='convex', direction='decreasing')
        
#         if knee_locator.knee is not None:
#             knee_index = knee_locator.knee
#             return smoothed_scores[knee_index]
#         else:
#             return sorted_scores.mean() + sorted_scores.std()

# def explode_submitters(df):
#     """
#     Create separate rows for papers with multiple submitters,
#     duplicating the paper's data for each submitter
#     """
#     # Create a copy of the dataframe
#     df_exploded = df.copy()
    
#     # Split the 'Submitted By' column and explode it
#     df_exploded['Submitted By'] = df_exploded['Submitted By'].str.split(', ')
#     df_exploded = df_exploded.explode('Submitted By')
    
#     return df_exploded

# def create_individual_topic_trends(df, selected_topics):
#     """Create separate line plots for each topic, with Submitted By as legend, showing paper counts"""
#     filtered_data = {}
    
#     # Explode the dataframe to handle multiple submitters
#     df_exploded = explode_submitters(df)
    
#     # Create a subplot for each topic
#     num_topics = len(selected_topics)
#     fig, axes = plt.subplots(num_topics, 1, figsize=(15, 8*num_topics))
    
#     # Process each topic
#     for idx, topic_num in enumerate(selected_topics):
#         topic_col = f'Topic {topic_num} Score'
#         ax = axes[idx] if num_topics > 1 else axes
        
#         # Get threshold and filter papers
#         threshold = get_threshold(df[topic_col], topic_num)
#         filtered_df = df_exploded[df_exploded[topic_col] > threshold].copy()
#         filtered_df = filtered_df.drop_duplicates(subset=['Title', 'Submitted By'])
        
#         # Get top submitters by total number of papers to avoid overcrowding the plot
#         submitter_counts = filtered_df['Submitted By'].value_counts()
#         top_submitters = submitter_counts.nlargest(10).index
        
#         # Calculate yearly counts for each submitter
#         for submitter in top_submitters:
#             submitter_df = filtered_df[filtered_df['Submitted By'] == submitter]
#             yearly_counts = submitter_df.groupby('Year').size().reset_index(name='count')
            
#             # Only plot if there are enough data points
#             if len(yearly_counts) > 0:
#                 # Plot line for this submitter
#                 ax.plot(yearly_counts['Year'], yearly_counts['count'], 
#                        marker='o', label=f'{submitter}')
                
#                 # Store filtered data
#                 if topic_num not in filtered_data:
#                     filtered_data[topic_num] = {}
#                 filtered_data[topic_num][submitter] = yearly_counts
        
#         # Customize subplot
#         ax.set_title(f'Topic {topic_num} Yearly Paper Counts by Top Submitters\n(Filtered by Knee Detection Threshold)')
#         ax.set_xlabel('Year')
#         ax.set_ylabel('Number of Papers')
#         ax.legend(title='Submitted By', bbox_to_anchor=(1.05, 1), loc='upper left')
#         ax.grid(True, linestyle='--', alpha=0.7)
        
#         # Rotate x-axis labels for better readability
#         ax.tick_params(axis='x', rotation=45)
    
#     # Adjust layout to prevent label cutoff
#     plt.tight_layout()
    
#     # Save plot
#     plt.savefig('individual_topic_trends_by_submitter_paper_counts_all.png', dpi=300, bbox_inches='tight')
#     plt.close()
    
#     return filtered_data

# def main(filepath):
#     df = process_data(filepath)
#     green_topics = {1, 5, 7, 14}
    
#     # Create individual topic trends plots
#     print("\nGenerating individual topic trends plots by submitter...")
#     filtered_data = create_individual_topic_trends(df, range(0,20))
    
#     print("\nAll visualizations generated successfully!")
    
#     return filtered_data

# if __name__ == "__main__":
#     main('Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv')

import pandas as pd
import networkx as nx
from itertools import combinations

def create_coauthor_network(csv_path):
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create an empty graph
    G = nx.Graph()
    
    # Create a dictionary to store co-authorship counts
    coauthor_counts = {}
    
    # Process each paper's submitters
    for submitters in df['Submitted By']:
        # Split the submitters and strip whitespace
        author_list = [author.strip() for author in submitters.split(',')]
        
        # Add nodes for each author if they don't exist
        G.add_nodes_from(author_list)
        
        # Create edges for each pair of co-authors
        for author1, author2 in combinations(author_list, 2):
            if (author1, author2) in coauthor_counts:
                coauthor_counts[(author1, author2)] += 1
            elif (author2, author1) in coauthor_counts:
                coauthor_counts[(author2, author1)] += 1
            else:
                coauthor_counts[(author1, author2)] = 1
    
    # Add weighted edges to the graph
    for (author1, author2), weight in coauthor_counts.items():
        G.add_edge(author1, author2, weight=weight)
    
    # Calculate network metrics
    metrics = {
        'Author': [],
        'Weighted_Degree': [],
        'Degree': [],
        'Degree_Centrality': [],
        'Betweenness_Centrality': [],
        'Closeness_Centrality': [],
        'Eigenvector_Centrality': []
    }
    
    # Calculate degree centrality
    deg_centrality = nx.degree_centrality(G)
    
    # Calculate betweenness centrality
    betw_centrality = nx.betweenness_centrality(G)
    
    # Calculate closeness centrality
    close_centrality = nx.closeness_centrality(G)
    
    # Calculate eigenvector centrality
    eigenv_centrality = nx.eigenvector_centrality(G, max_iter=1000)
    
    # For each node in the graph
    for author in G.nodes():
        metrics['Author'].append(author)
        
        # Weighted degree (sum of edge weights)
        weighted_deg = sum(G[author][neighbor]['weight'] for neighbor in G[author])
        metrics['Weighted_Degree'].append(weighted_deg)
        
        # Regular degree (number of connections)
        metrics['Degree'].append(G.degree(author))
        
        # Add other centrality measures
        metrics['Degree_Centrality'].append(deg_centrality[author])
        metrics['Betweenness_Centrality'].append(betw_centrality[author])
        metrics['Closeness_Centrality'].append(close_centrality[author])
        metrics['Eigenvector_Centrality'].append(eigenv_centrality[author])
    
    # Create DataFrame with results
    results_df = pd.DataFrame(metrics)
    
    # Round numerical columns to 4 decimal places
    numeric_columns = results_df.columns.drop('Author')
    results_df[numeric_columns] = results_df[numeric_columns].round(4)
    
    return results_df

# Usage
if __name__ == "__main__":
    # Replace 'your_file.csv' with your actual CSV file path
    csv_path = 'Stopwords_added_AllSubmitters_Since91_20_Topic_Loading_Score.csv'
    
    try:
        # Create the network and calculate metrics
        results = create_coauthor_network(csv_path)
        
        # Save results to CSV
        output_path = 'network_metrics_Full_NoTopics.csv'
        results.to_csv(output_path, index=False)
        print(f"Results have been saved to {output_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")