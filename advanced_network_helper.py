"""
Advanced Network Analysis with Sophisticated Interaction Detection
Handles:
- Question-answer patterns
- Topic clustering
- Conversational threads
- Multiple reply types
- Quote detection
- Contextual analysis
"""

import pandas as pd
import networkx as nx
from datetime import timedelta
from collections import defaultdict, Counter
import re
import plotly.graph_objects as go
import community.community_louvain as community_louvain
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class AdvancedInteractionDetector:
    """
    Sophisticated interaction detection system for WhatsApp chats
    """
    
    def __init__(self, df, time_window_seconds=300):
        self.df = df[df['user'] != 'group_notification'].copy()
        self.df = self.df.sort_values('date').reset_index(drop=True)
        self.time_window = time_window_seconds
        self.interactions = defaultdict(int)
        self.interaction_types = defaultdict(lambda: defaultdict(int))
        
    def detect_all_interactions(self):
        """
        Run all detection methods and aggregate results
        """
        print("Running advanced interaction detection...")
        
        # Method 1: Time-based proximity (baseline)
        self._detect_temporal_replies()
        
        # Method 2: Question-Answer patterns
        self._detect_question_answer_pairs()
        
        # Method 3: @Mentions and direct addressing
        self._detect_explicit_mentions()
        
        # Method 4: Topic-based clustering
        self._detect_topic_threads()
        
        # Method 5: Conversational bursts
        self._detect_conversation_bursts()
        
        # Method 6: Quote/Reply detection
        self._detect_quoted_replies()
        
        return self.interactions, self.interaction_types
    
    def _detect_temporal_replies(self):
        """
        Basic time-based reply detection (refined)
        """
        for i in range(1, len(self.df)):
            current_user = self.df.iloc[i]['user']
            current_time = self.df.iloc[i]['date']
            
            # Look at previous messages within time window
            for j in range(i-1, max(-1, i-10), -1):  # Check last 10 messages max
                prev_user = self.df.iloc[j]['user']
                prev_time = self.df.iloc[j]['date']
                
                time_diff = (current_time - prev_time).total_seconds()
                
                if time_diff <= self.time_window and current_user != prev_user:
                    # Weight decreases with time
                    weight = 1.0 - (time_diff / self.time_window) * 0.5
                    self.interactions[(prev_user, current_user)] += weight
                    self.interaction_types[(prev_user, current_user)]['temporal'] += weight
                elif time_diff > self.time_window:
                    break
    
    def _detect_question_answer_pairs(self):
        """
        Detect question-answer patterns
        """
        question_patterns = [
            r'\?$',  # Ends with ?
            r'^(what|where|when|who|why|how|kya|kaise|kab|kahan|kaun)',  # Question words
            r'(should|could|would|can|chahiye|kar sakte)',  # Modal verbs
            r'(anyone|kisi ne|kisiko)',  # Open questions
        ]
        
        for i in range(len(self.df)):
            message = str(self.df.iloc[i]['message']).lower()
            user = self.df.iloc[i]['user']
            time = self.df.iloc[i]['date']
            
            # Check if it's a question
            is_question = any(re.search(pattern, message, re.IGNORECASE) 
                            for pattern in question_patterns)
            
            if is_question:
                # Look for answers in next 5 messages or within 10 minutes
                for j in range(i+1, min(len(self.df), i+6)):
                    responder = self.df.iloc[j]['user']
                    resp_time = self.df.iloc[j]['date']
                    
                    if responder != user and (resp_time - time).total_seconds() <= 600:
                        # Strong weight for Q&A
                        self.interactions[(user, responder)] += 2.0
                        self.interaction_types[(user, responder)]['question_answer'] += 2.0
    
    def _detect_explicit_mentions(self):
        """
        Detect @mentions and name mentions with better accuracy
        """
        # Create name variations for better matching
        user_names = {}
        for user in self.df['user'].unique():
            user_names[user] = self._get_name_variations(user)
        
        for idx, row in self.df.iterrows():
            message = str(row['message'])
            sender = row['user']
            
            for target_user, name_variants in user_names.items():
                if target_user == sender:
                    continue
                
                # Check for @mentions (exact)
                if any(f"@{variant}" in message for variant in name_variants):
                    self.interactions[(sender, target_user)] += 3.0  # High weight
                    self.interaction_types[(sender, target_user)]['explicit_mention'] += 3.0
                
                # Check for name at start of message (addressing)
                elif any(message.lower().startswith(variant.lower()) 
                        for variant in name_variants):
                    self.interactions[(sender, target_user)] += 2.5
                    self.interaction_types[(sender, target_user)]['direct_address'] += 2.5
                
                # Check for name in message with word boundaries
                elif any(re.search(r'\b' + re.escape(variant) + r'\b', 
                                  message, re.IGNORECASE) 
                        for variant in name_variants):
                    self.interactions[(sender, target_user)] += 1.5
                    self.interaction_types[(sender, target_user)]['name_mention'] += 1.5
    
    def _get_name_variations(self, username):
        """
        Generate name variations for better matching
        """
        variations = [username]
        
        # Split on spaces and get first name
        parts = username.split()
        if len(parts) > 1:
            variations.append(parts[0])  # First name
            variations.append(' '.join(parts[:2]))  # First two parts
        
        # Remove common suffixes
        clean_name = re.sub(r'\s+(IIT|Sharma|Aggarwal|Kumar)$', '', username)
        if clean_name != username:
            variations.append(clean_name)
        
        return list(set(variations))
    
    def _detect_topic_threads(self):
        """
        Use TF-IDF to detect topically related messages
        """
        if len(self.df) < 5:
            return
        
        # Prepare messages for TF-IDF
        messages = [str(msg) for msg in self.df['message']]
        
        try:
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english', 
                                        min_df=1, max_df=0.8)
            tfidf_matrix = vectorizer.fit_transform(messages)
            
            # Calculate similarity within time windows
            for i in range(len(self.df)):
                current_user = self.df.iloc[i]['user']
                current_time = self.df.iloc[i]['date']
                
                # Check next 10 messages
                for j in range(i+1, min(len(self.df), i+11)):
                    other_user = self.df.iloc[j]['user']
                    other_time = self.df.iloc[j]['date']
                    
                    if other_user == current_user:
                        continue
                    
                    # Within 30 minutes
                    if (other_time - current_time).total_seconds() <= 1800:
                        # Calculate topic similarity
                        similarity = cosine_similarity(
                            tfidf_matrix[i:i+1], 
                            tfidf_matrix[j:j+1]
                        )[0][0]
                        
                        if similarity > 0.15:  # Threshold for topic relatedness
                            weight = similarity * 1.5
                            self.interactions[(current_user, other_user)] += weight
                            self.interaction_types[(current_user, other_user)]['topic_thread'] += weight
        except:
            pass  # Skip if TF-IDF fails (too few messages, etc.)
    
    def _detect_conversation_bursts(self):
        """
        Detect rapid exchanges between users (conversation bursts)
        """
        for i in range(len(self.df) - 1):
            current_user = self.df.iloc[i]['user']
            current_time = self.df.iloc[i]['date']
            
            # Check next few messages for burst pattern
            consecutive_users = []
            for j in range(i+1, min(len(self.df), i+5)):
                next_user = self.df.iloc[j]['user']
                next_time = self.df.iloc[j]['date']
                
                time_gap = (next_time - current_time).total_seconds()
                
                # Rapid exchange (within 2 minutes)
                if time_gap <= 120:
                    consecutive_users.append(next_user)
                else:
                    break
            
            # If there's a back-and-forth pattern
            for other_user in set(consecutive_users):
                if other_user != current_user:
                    count = consecutive_users.count(other_user)
                    if count >= 2:  # At least 2 messages
                        weight = count * 0.8
                        self.interactions[(current_user, other_user)] += weight
                        self.interaction_types[(current_user, other_user)]['burst'] += weight
    
    def _detect_quoted_replies(self):
        """
        Detect quoted replies (when users quote previous messages)
        """
        for idx, row in self.df.iterrows():
            message = str(row['message'])
            sender = row['user']
            
            # Look for quote patterns
            quote_patterns = [
                r'^[">»]',  # Starts with quote marker
                r'^\s*"[^"]+"\s*$',  # Text in quotes
            ]
            
            is_quoted = any(re.search(pattern, message) for pattern in quote_patterns)
            
            if is_quoted and idx > 0:
                # Find who was quoted (look back)
                for j in range(idx-1, max(-1, idx-10), -1):
                    prev_user = self.df.iloc[j]['user']
                    prev_msg = str(self.df.iloc[j]['message'])
                    
                    # Check if quote contains part of previous message
                    if prev_user != sender:
                        # Simple substring check
                        quote_text = message.strip('"">» ').lower()
                        if len(quote_text) > 10 and quote_text[:20] in prev_msg.lower():
                            self.interactions[(prev_user, sender)] += 2.5
                            self.interaction_types[(prev_user, sender)]['quoted_reply'] += 2.5
                            break


def build_advanced_interaction_graph(df, min_interactions=1, time_window_seconds=300):
    """
    Build graph using advanced detection
    """
    detector = AdvancedInteractionDetector(df, time_window_seconds)
    interactions, interaction_types = detector.detect_all_interactions()
    
    # Build graph
    G = nx.DiGraph()
    
    # Add nodes
    users = df[df['user'] != 'group_notification']['user'].unique()
    for user in users:
        user_data = df[df['user'] == user]
        G.add_node(user, 
                   message_count=len(user_data),
                   word_count=sum(len(str(msg).split()) for msg in user_data['message']))
    
    # Add edges with detailed interaction types
    for (user1, user2), weight in interactions.items():
        if weight >= min_interactions:
            if G.has_edge(user1, user2):
                G[user1][user2]['weight'] += weight
            else:
                G.add_edge(user1, user2, weight=weight)
            
            # Store interaction type breakdown
            G[user1][user2]['types'] = dict(interaction_types[(user1, user2)])
    
    # Convert to undirected for community detection
    G_undirected = G.to_undirected()
    
    return G, G_undirected


def create_advanced_network_visualization(G, metrics, communities, anonymize=False, max_nodes=50):
    """
    Enhanced visualization with interaction type information
    """
    # Limit nodes if needed
    if len(G.nodes()) > max_nodes:
        top_nodes = sorted(metrics.keys(), 
                          key=lambda x: metrics[x]['message_count'], 
                          reverse=True)[:max_nodes]
        G = G.subgraph(top_nodes).copy()
    
    # Layout
    try:
        pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42, weight='weight')
    except:
        pos = nx.circular_layout(G)
    
    # Create edges with hover info
    edge_traces = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        weight = G[edge[0]][edge[1]].get('weight', 1)
        
        # Get interaction types
        types = G[edge[0]][edge[1]].get('types', {})
        type_info = '<br>'.join([f"{k}: {v:.1f}" for k, v in types.items()])
        
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(width=min(weight * 0.3, 8), color='rgba(150,150,150,0.5)'),
            hoverinfo='text',
            hovertext=f"Weight: {weight:.1f}<br>{type_info}",
            showlegend=False
        )
        edge_traces.append(edge_trace)
    
    # Create nodes
    node_x, node_y, node_text, node_size, node_color = [], [], [], [], []
    
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        display_name = f"User {hash(node) % 1000}" if anonymize else node
        m = metrics[node]
        
        text = (f"<b>{display_name}</b><br>"
                f"Messages: {m['message_count']}<br>"
                f"Words: {m['word_count']}<br>"
                f"Connections: {m['total_degree']}<br>"
                f"Weighted Interactions: {m['weighted_in_degree'] + m['weighted_out_degree']:.1f}<br>"
                f"Community: {m['community']}<br>"
                f"PageRank: {m['pagerank']:.3f}<br>"
                f"Betweenness: {m['betweenness']:.3f}")
        node_text.append(text)
        
        node_size.append(max(15, min(60, m['message_count'] * 0.8)))
        node_color.append(m['community'])
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[f"User {hash(n) % 1000}" if anonymize else n.split()[0] for n in G.nodes()],
        textposition="top center",
        textfont=dict(size=9, color='black'),
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
    fig = go.Figure(data=edge_traces + [node_trace],
                    layout=go.Layout(
                        title=dict(text='Advanced Chat Network Graph', font=dict(size=18)),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20, l=5, r=5, t=50),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                        plot_bgcolor='white',
                        height=700
                    ))
    
    return fig


# Import existing functions from network_helper for compatibility
from network_helper import (
    compute_graph_metrics,
    get_network_insights,
    export_network_data
)

