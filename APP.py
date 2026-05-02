import streamlit as st
import preprocessor
import helper
import network_helper
import advanced_network_helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from datetime import datetime
st.sidebar.title("WhatsApp Chat Analyzer")

# Initialize session state
if 'analysis_run' not in st.session_state:
    st.session_state.analysis_run = False

uploaded_file = st.sidebar.file_uploader("Choose the txt chat file")
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode('utf-8')
    df = preprocessor.preprocess(data)

    # fetch unique users
    user_list = df['user'].unique().tolist()
    if 'group_notification' in user_list:
        user_list.remove("group_notification")
    user_list.sort()
    user_list.insert(0, "Overall")
    selected_user = st.sidebar.selectbox("Show Analysis WRT", user_list)

    col_btn1, col_btn2 = st.sidebar.columns(2)
    with col_btn1:
        if st.button("Show Analysis"):
            st.session_state.analysis_run = True
    with col_btn2:
        if st.button("Reset"):
            st.session_state.analysis_run = False
            st.rerun()

    if st.session_state.analysis_run:
        num_messages, words, avg_words_pertext, num_mediamess, num_links = helper.fetch_stats(selected_user,df)
        col1, col2,col_ex, col3, col4 = st.columns(5)
        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col_ex:
            st.header("Avg words per text")
            st.title(avg_words_pertext)
        with col3:
            st.header("Total No of media files shared")
            st.title(num_mediamess)
        with col4:
            st.header("Total No of links shared")
            st.title(num_links)
        # finding the most busy user (only for group chats with more than 2 users)
        # Count actual users (excluding group_notification)
        actual_users = df[df['user'] != 'group_notification']['user'].nunique()
        
        if selected_user == "Overall" and actual_users > 2:
            st.markdown("<h1 style='text-align: center; font-size: 30px;'> Top 5 Users with percent chat </h1>",
                        unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            x, df_percent = helper.most_busy_users(df)
            fig, ax = plt.subplots()
            with col1:
                ax.bar(x.index,x.values)
                plt.xticks(rotation='vertical', color="red")
                st.pyplot(fig)
            with col2:
                st.dataframe(df_percent)
        elif selected_user == "Overall" and actual_users <= 2:
            st.info("📱 This appears to be a personal chat (1-on-1 conversation)")

        # Word Cloud
        st.title("Word Cloud")
        df_wc = helper.create_wordcloud(selected_user,df)
        fig, ax = plt.subplots()
        ax.imshow(df_wc)
        ax.axis('off')  # Hide axis
        st.pyplot(fig)

        # Most common words
        st.title('Most Common Words')
        most_common_df = helper.most_common_words(selected_user,df)
        
        if not most_common_df.empty:
            fig, ax = plt.subplots()
            ax.barh(most_common_df['word'], most_common_df['count'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        else:
            st.info("No common words found after filtering stop words")

        # Monthly timeline
        st.title("Monthly Timeline")
        timeline = helper.monthly_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(timeline['time'], timeline['message'], color='green')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Daily timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'], color='black')
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        # Activity map
        st.title('Activity Map')
        col1, col2 = st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day = helper.week_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_day.index, busy_day.values, color='purple')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        # Weekly Activity Heatmap
        st.title("Weekly Activity Heatmap")
        user_heatmap = helper.activity_heatmap(selected_user, df)
        fig, ax = plt.subplots()
        sns.heatmap(user_heatmap, ax=ax)
        st.pyplot(fig)

        # Emoji analysis
        st.title("Emoji Analysis")
        # Emoji analysis
        st.title("Emoji Analysis")
        emoji_df = helper.emoji_helper(selected_user, df)

        if not emoji_df.empty:
            st.dataframe(emoji_df)
        else:
            st.info("No emojis found in the selected messages")


        # ================== KEY INSIGHTS ==================
        st.markdown("---")
        st.title(" Key Insights")

        insights = helper.generate_insights(df)

        for i in insights:
            st.write("•", i)


        # ================== SENTIMENT ==================
        st.markdown("---")
        st.title(" Sentiment Trend")

        sentiment_df = helper.sentiment_timeline(selected_user, df)

        if not sentiment_df.empty:
            fig, ax = plt.subplots()
            ax.plot(sentiment_df['date'], sentiment_df['sentiment'])
            plt.xticks(rotation='vertical')
            st.pyplot(fig)
        else:
            st.info("Not enough data for sentiment analysis")


        # ================== TOPIC MODELING ==================
        st.markdown("---")
        st.title("Topic Modeling")

        try:
            topics = helper.topic_modeling(selected_user, df)
            for t in topics:
                st.write("•", t)
        except:
            st.info("Not enough data for topic modeling")
        
        if not emoji_df.empty:
            st.dataframe(emoji_df)
        else:
            st.info("No emojis found in the selected messages")

        # ============ NETWORK ANALYSIS SECTION ============
        st.markdown("---")
        st.title("🕸️ Chat Network Graph")
        
        # Only show network analysis for group chats or when "Overall" is selected
        if selected_user == "Overall" and actual_users > 2:
            st.markdown("""
            Visualize how users interact with each other. The network graph shows:
            - **Nodes**: Each user in the chat
            - **Edges**: Connections based on reply patterns and interactions
            - **Node size**: Based on message activity
            - **Node color**: Community groups detected by the algorithm
            - **Edge thickness**: Strength of interaction
            """)
            
            # Create filter controls in sidebar with form for batching updates
            st.sidebar.markdown("---")
            st.sidebar.subheader("🕸️ Network Filters")
            
            # Date range filter
            min_date = df['only_date'].min()
            max_date = df['only_date'].max()
            
            with st.sidebar.form(key='network_filters'):
                date_range = st.date_input(
                    "Date Range",
                    value=(min_date, max_date),
                    min_value=min_date,
                    max_value=max_date
                )
                
                # Minimum interaction weight slider
                min_weight = st.slider(
                    "Minimum Interactions",
                    min_value=1,
                    max_value=10,
                    value=1,
                    help="Filter edges with fewer interactions than this value"
                )
                
                # Time window for reply detection
                time_window = st.slider(
                    "Reply Time Window (minutes)",
                    min_value=1,
                    max_value=30,
                    value=5,
                    help="Consider messages as replies if sent within this time window"
                )
                
                # Max nodes limit
                max_nodes = st.slider(
                    "Max Nodes to Display",
                    min_value=10,
                    max_value=100,
                    value=50,
                    step=5,
                    help="Limit number of nodes for better performance"
                )
                
                # Anonymize toggle
                anonymize = st.checkbox(
                    "Anonymize Usernames",
                    value=False,
                    help="Replace usernames with anonymous identifiers"
                )
                
                # Advanced detection toggle
                st.markdown("---")
                use_advanced = st.checkbox(
                    "Use Advanced Detection",
                    value=True,
                    help="Uses sophisticated algorithms including Q&A detection, topic clustering, and contextual analysis"
                )
                
                # Submit button for the form
                update_network = st.form_submit_button("🔄 Update Network Graph")
            
            # Apply date filter
            if len(date_range) == 2:
                df_filtered = df[(df['only_date'] >= date_range[0]) & 
                                (df['only_date'] <= date_range[1])]
            else:
                df_filtered = df
            
            if use_advanced:
                st.sidebar.info("Advanced mode includes:\n"
                              "• Question-Answer detection\n"
                              "• Topic similarity\n"
                              "• Conversation bursts\n"
                              "• Quote detection\n"
                              "• Better name matching")
            
            # Build and analyze network
            spinner_text = "Building network graph (Advanced mode)..." if use_advanced else "Building network graph..."
            with st.spinner(spinner_text):
                try:
                    # Build graph using selected method
                    if use_advanced:
                        G, G_undirected = advanced_network_helper.build_advanced_interaction_graph(
                            df_filtered, 
                            min_interactions=min_weight,
                            time_window_seconds=time_window * 60
                        )
                    else:
                        G, G_undirected = network_helper.build_interaction_graph(
                            df_filtered, 
                            min_interactions=min_weight,
                            time_window_seconds=time_window * 60
                        )
                    
                    if len(G.nodes()) < 2:
                        st.warning("Not enough interactions to build a network. Try adjusting the filters.")
                    else:
                        # Compute metrics
                        metrics, communities = network_helper.compute_graph_metrics(G, G_undirected)
                        top_influencer = helper.get_top_influencer(metrics)

                        if top_influencer:
                            st.write(f"Top Influencer: {top_influencer}")
                        # Get insights
                        insights = network_helper.get_network_insights(G, metrics, communities)
                        
                        # Create two columns: graph and insights
                        col1, col2 = st.columns([2, 1])
                        
                        with col1:
                            # Create and display visualization
                            if use_advanced:
                                fig = advanced_network_helper.create_advanced_network_visualization(
                                    G, metrics, communities, 
                                    anonymize=anonymize, 
                                    max_nodes=max_nodes
                                )
                            else:
                                fig = network_helper.create_network_visualization(
                                    G, metrics, communities, 
                                    anonymize=anonymize, 
                                    max_nodes=max_nodes
                                )
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with col2:
                            # Display insights
                            st.subheader("📊 Network Insights")
                            
                            st.metric("Total Users", insights['total_nodes'])
                            st.metric("Total Connections", insights['total_edges'])
                            st.metric("Total Interactions", insights['total_interactions'])
                            st.metric("Network Density", f"{insights['density']:.3f}")
                            
                            st.markdown("---")
                            st.markdown("**🌟 Top Connectors**")
                            for user, degree in insights['top_connectors'][:5]:
                                display_name = f"User {hash(user) % 1000}" if anonymize else user
                                st.write(f"• {display_name}: {degree} connections")
                            
                            st.markdown("---")
                            st.markdown("**🎯 Most Influential**")
                            st.caption("(Based on PageRank)")
                            for user, pagerank in insights['top_influential'][:5]:
                                display_name = f"User {hash(user) % 1000}" if anonymize else user
                                st.write(f"• {display_name}: {pagerank:.3f}")
                            
                            st.markdown("---")
                            st.markdown("**👥 Communities Detected**")
                            st.write(f"Number of communities: {insights['num_communities']}")
                            st.write(f"Largest community: {insights['largest_community_size']} members")
                            
                            with st.expander("Community Sizes"):
                                for comm_id, size in list(insights['community_sizes'].items())[:10]:
                                    st.write(f"Community {comm_id + 1}: {size} members")
                            
                            st.markdown("---")
                            st.markdown("**⚠️ Low Interaction Users**")
                            st.write(f"{insights['low_interaction_users']} users with ≤1 connection")
                        
                        # Export section
                        st.markdown("---")
                        st.subheader("📥 Export Network Data")
                        
                        col1, col2 = st.columns(2)
                        
                        # Export nodes
                        nodes_df, edges_df = network_helper.export_network_data(G, metrics)
                        
                        with col1:
                            st.download_button(
                                label="Download Nodes CSV",
                                data=nodes_df.to_csv(index=False),
                                file_name="network_nodes.csv",
                                mime="text/csv",
                                help="Download node data with metrics"
                            )
                        
                        with col2:
                            st.download_button(
                                label="Download Edges CSV",
                                data=edges_df.to_csv(index=False),
                                file_name="network_edges.csv",
                                mime="text/csv",
                                help="Download edge data with weights"
                            )
                        
                        # Detailed metrics table
                        with st.expander("📋 View Detailed Node Metrics"):
                            display_df = nodes_df.copy()
                            if anonymize:
                                display_df['user'] = display_df['user'].apply(
                                    lambda x: f"User {hash(x) % 1000}"
                                )
                            st.dataframe(display_df, use_container_width=True)
                
                except Exception as e:
                    st.error(f"Error building network: {str(e)}")
                    st.info("Try adjusting the filters or check if there's enough interaction data.")
        
        elif selected_user == "Overall" and actual_users <= 2:
            st.info("📱 Network analysis is only available for group chats with 3 or more users.")
        
        else:
            st.info("🔍 Network analysis is only available when 'Overall' view is selected for group chats.")