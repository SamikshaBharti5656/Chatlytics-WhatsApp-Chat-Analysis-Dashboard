import pandas as pd
import networkx as nx
from datetime import timedelta
from collections import defaultdict
import plotly.graph_objects as go
import community.community_louvain as community_louvain

def detect_interactions(df, time_window_seconds=300):
    """
    Detect interactions between users based on:
    - Reply patterns (messages sent within time_window after another user)
    - @mentions in messages
    
    Returns a dictionary with interaction counts between users
    """
    interactions = defaultdict(int)
    
    # Filter out system messages
    df_users = df[df['user'] != 'group_notification'].copy()
    df_users = df_users.sort_values('date').reset_index(drop=True)
    
    # Detect time-based interactions (reply patterns)
    for i in range(1, len(df_users)):
        current_user = df_users.iloc[i]['user']
        current_time = df_users.iloc[i]['date']
        
        # Look at previous messages within time window
        for j in range(i-1, -1, -1):
            prev_user = df_users.iloc[j]['user']
            prev_time = df_users.iloc[j]['date']
            
            time_diff = (current_time - prev_time).total_seconds()
            
            # If within time window and different users
            if time_diff <= time_window_seconds and current_user != prev_user:
                # Create directed edge (prev_user -> current_user)
                interactions[(prev_user, current_user)] += 1
            elif time_diff > time_window_seconds:
                break  # Stop looking further back
    
    # Detect @mentions
    for idx, row in df_users.iterrows():
        message = str(row['message']).lower()
        sender = row['user']
        
        # Check for mentions of other users
        for other_user in df_users['user'].unique():
            if other_user != sender:
                # Simple mention detection (customize as needed)
                if f"@{other_user.lower()}" in message or other_user.lower() in message:
                    interactions[(sender, other_user)] += 1
    
    return interactions


def build_interaction_graph(df, min_interactions=1, time_window_seconds=300):
    """
    Build a weighted directed graph of user interactions
    
    Args:
        df: DataFrame with chat data
        min_interactions: Minimum number of interactions to include an edge
        time_window_seconds: Time window for considering replies
    
    Returns:
        NetworkX DiGraph
    """
    interactions = detect_interactions(df, time_window_seconds)
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Get all users (excluding system messages)
    users = df[df['user'] != 'group_notification']['user'].unique()
    
    # Add all users as nodes
    for user in users:
        user_data = df[df['user'] == user]
        G.add_node(user, 
                   message_count=len(user_data),
                   word_count=sum(len(str(msg).split()) for msg in user_data['message']))
    
    # Add edges with weights
    for (user1, user2), weight in interactions.items():
        if weight >= min_interactions:
            if G.has_edge(user1, user2):
                G[user1][user2]['weight'] += weight
            else:
                G.add_edge(user1, user2, weight=weight)
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    return G, G_undirected


def compute_graph_metrics(G, G_undirected):
    """
    Compute various graph metrics for nodes
    
    Returns:
        Dictionary with metrics for each node
    """
    metrics = {}
    
    # Degree centrality
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    # Weighted degree
    weighted_in_degree = dict(G.in_degree(weight='weight'))
    weighted_out_degree = dict(G.out_degree(weight='weight'))
    
    # Betweenness centrality (using undirected graph)
    try:
        betweenness = nx.betweenness_centrality(G_undirected, weight='weight')
    except:
        betweenness = {node: 0 for node in G.nodes()}
    
    # PageRank
    try:
        pagerank = nx.pagerank(G, weight='weight')
    except:
        pagerank = {node: 0 for node in G.nodes()}
    
    # Community detection (Louvain method)
    try:
        communities = community_louvain.best_partition(G_undirected, weight='weight')
    except:
        communities = {node: 0 for node in G.nodes()}
    
    for node in G.nodes():
        metrics[node] = {
            'in_degree': in_degree.get(node, 0),
            'out_degree': out_degree.get(node, 0),
            'total_degree': in_degree.get(node, 0) + out_degree.get(node, 0),
            'weighted_in_degree': weighted_in_degree.get(node, 0),
            'weighted_out_degree': weighted_out_degree.get(node, 0),
            'betweenness': betweenness.get(node, 0),
            'pagerank': pagerank.get(node, 0),
            'community': communities.get(node, 0),
            'message_count': G.nodes[node].get('message_count', 0),
            'word_count': G.nodes[node].get('word_count', 0)
        }
    
    return metrics, communities


def create_network_visualization(G, metrics, communities, anonymize=False, max_nodes=50):
    """
    Create an interactive Plotly network visualization
    
    Args:
        G: NetworkX graph
        metrics: Node metrics dictionary
        communities: Community assignments
        anonymize: Whether to anonymize usernames
        max_nodes: Maximum number of nodes to display
    
    Returns:
        Plotly figure object
    """
    # Limit nodes if needed (keep most active users)
    if len(G.nodes()) > max_nodes:
        top_nodes = sorted(metrics.keys(), 
                          key=lambda x: metrics[x]['message_count'], 
                          reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
    
    # Use spring layout for positioning
    try:
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    except:
        pos = nx.circular_layout(G)
    
    # Create edges
    edge_trace = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G[edge[0]][edge[1]].get('weight', 1)
        
        edge_trace.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=min(weight * 0.5, 10), color='#888'),
                hoverinfo='none',
                showlegend=False
            )
        )
    
    # Create nodes
    node_x = []
    node_y = []
    node_text = []
    node_size = []
    node_color = []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        # Anonymize if needed
        display_name = f"User {hash(node) % 1000}" if anonymize else node
        
        # Hover text
        m = metrics[node]
        text = (f"<b>{display_name}</b><br>"
                f"Messages: {m['message_count']}<br>"
                f"Words: {m['word_count']}<br>"
                f"Connections: {m['total_degree']}<br>"
                f"Interactions: {m['weighted_in_degree'] + m['weighted_out_degree']}<br>"
                f"Community: {m['community']}<br>"
                f"Influence (PageRank): {m['pagerank']:.3f}")
        node_text.append(text)
        
        # Size based on message count
        node_size.append(max(10, min(50, m['message_count'] * 0.5)))
        
        # Color based on community
        node_color.append(m['community'])
    
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"User {hash(n) % 1000}" if anonymize else n.split()[0] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=8),
        hovertext=node_text,
        marker=dict(
            showscale=True,
            colorscale='Viridis',
            size=node_size,
            color=node_color,
            colorbar=dict(
                thickness=15,
                title=dict(text='Community', side='right'),
                xanchor='left'
            ),
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace],
                    layout=go.Layout(
                        title=dict(text='Chat Network Graph', font=dict(size=16)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=40),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',
                        height=600
                    ))
    
    return fig


def get_network_insights(G, metrics, communities):
    """
    Generate summary insights about the network
    
    Returns:
        Dictionary with various insights
    """
    insights = {}
    
    # Basic stats
    insights['total_nodes'] = len(G.nodes())
    insights['total_edges'] = len(G.edges())
    insights['total_interactions'] = sum(d['weight'] for u, v, d in G.edges(data=True))
    
    # Top connectors (by total degree)
    top_connectors = sorted(metrics.items(), 
                           key=lambda x: x[1]['total_degree'], 
                           reverse=True)[:5]
    insights['top_connectors'] = [(user, m['total_degree']) for user, m in top_connectors]
    
    # Most influential (by PageRank)
    top_influential = sorted(metrics.items(), 
                            key=lambda x: x[1]['pagerank'], 
                            reverse=True)[:5]
    insights['top_influential'] = [(user, m['pagerank']) for user, m in top_influential]
    
    # Isolated or low-interaction users
    low_interaction = [(user, m['total_degree']) 
                       for user, m in metrics.items() 
                       if m['total_degree'] <= 1]
    insights['low_interaction_users'] = len(low_interaction)
    
    # Community analysis
    community_sizes = defaultdict(int)
    for node, comm in communities.items():
        community_sizes[comm] += 1
    
    insights['num_communities'] = len(community_sizes)
    insights['community_sizes'] = dict(sorted(community_sizes.items(), 
                                              key=lambda x: x[1], 
                                              reverse=True))
    insights['largest_community_size'] = max(community_sizes.values()) if community_sizes else 0
    
    # Network density
    try:
        insights['density'] = nx.density(G)
    except:
        insights['density'] = 0
    
    return insights


def export_network_data(G, metrics):
    """
    Export network data as DataFrames for CSV download
    
    Returns:
        nodes_df, edges_df
    """
    # Nodes DataFrame
    nodes_data = []
    for node in G.nodes():
        m = metrics[node]
        nodes_data.append({
            'user': node,
            'message_count': m['message_count'],
            'word_count': m['word_count'],
            'in_degree': m['in_degree'],
            'out_degree': m['out_degree'],
            'total_connections': m['total_degree'],
            'weighted_in': m['weighted_in_degree'],
            'weighted_out': m['weighted_out_degree'],
            'betweenness': m['betweenness'],
            'pagerank': m['pagerank'],
            'community': m['community']
        })
    nodes_df = pd.DataFrame(nodes_data)
    
    # Edges DataFrame
    edges_data = []
    for u, v, data in G.edges(data=True):
        edges_data.append({
            'source': u,
            'target': v,
            'weight': data.get('weight', 1)
        })
    edges_df = pd.DataFrame(edges_data)
    
    return nodes_df, edges_df

