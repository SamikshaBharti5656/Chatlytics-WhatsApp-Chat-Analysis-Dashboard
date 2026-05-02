from urlextract import URLExtract
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

extract = URLExtract()

# ================== SENTIMENT ==================
def sentiment_analysis(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user].copy()

    analyzer = SentimentIntensityAnalyzer()

    sentiments = []
    for message in df['message']:
        if isinstance(message, str):
            sentiments.append(analyzer.polarity_scores(message)['compound'])

    sentiment_df = pd.DataFrame(sentiments, columns=['score'])

    def label(score):
        if score > 0.05:
            return 'Positive'
        elif score < -0.05:
            return 'Negative'
        else:
            return 'Neutral'

    sentiment_df['type'] = sentiment_df['score'].apply(label)
    return sentiment_df


# ================== FETCH STATS ==================
def fetch_stats(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user].copy()

    num_messages = df.shape[0]

    words = []
    for message in df['message']:
        if isinstance(message, str):
            words.extend(re.findall(r'\b\w+\b', message.lower()))

    avg_words = round(len(words) / num_messages, 2) if num_messages > 0 else 0

    num_media = df['message'].str.contains('<Media omitted>', na=False).sum()

    links = []
    for message in df['message']:
        if isinstance(message, str):
            links.extend(extract.find_urls(message))

    return num_messages, len(words), avg_words, num_media, len(links)


# ================== WORDCLOUD ==================
def create_wordcloud(selected_user, df):

    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().split()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user].copy()

    temp = df[(df['user'] != 'group_notification') &
              (~df['message'].str.contains('<Media omitted>', na=False))]

    def clean(msg):
        if not isinstance(msg, str):
            return ""
        words = re.findall(r'\b\w+\b', msg.lower())
        return " ".join([w for w in words if w not in stop_words])

    temp['message'] = temp['message'].apply(clean)

    text = " ".join(temp['message'].astype(str))
    if not text.strip():
        text = "No meaningful words"

    wc = WordCloud(width=500, height=500, background_color='white')
    return wc.generate(text)


# ================== COMMON WORDS ==================
def most_common_words(selected_user, df):

    with open('stop_hinglish.txt', 'r') as f:
        stop_words = f.read().split()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp = df[(df['user'] != 'group_notification') &
              (~df['message'].str.contains('<Media omitted>', na=False))]

    words = []
    for message in temp['message']:
        if isinstance(message, str):
            tokens = re.findall(r'\b\w+\b', message.lower())
            words.extend([w for w in tokens if w not in stop_words])

    return pd.DataFrame(Counter(words).most_common(20), columns=['word', 'count'])


# ================== TIMELINES ==================
def monthly_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year', 'month_num', 'month'])['message'].count().reset_index()
    timeline['time'] = timeline['month'] + "-" + timeline['year'].astype(str)
    return timeline


def daily_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    return df.groupby('only_date')['message'].count().reset_index()


# ================== HEATMAP ==================
def activity_heatmap(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    heatmap = df.pivot_table(index='day_name', columns='period',
                             values='message', aggfunc='count').fillna(0)

    order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
             'Friday', 'Saturday', 'Sunday']

    return heatmap.reindex(order)


# ================== RESPONSE TIME ==================
def response_time_analysis(selected_user, df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df = df[df['user'] != 'group_notification'].sort_values(by='date')

    if df.empty:
        return pd.DataFrame(columns=['user', 'response_time'])

    data = []
    for i in range(1, len(df)):
        if df.iloc[i]['user'] != df.iloc[i-1]['user']:
            diff = (df.iloc[i]['date'] - df.iloc[i-1]['date']).total_seconds()/60
            data.append({'user': df.iloc[i]['user'], 'response_time': diff})

    return pd.DataFrame(data)


# ================== TOPIC MODELING ==================
def topic_modeling(selected_user, df, n_topics=3):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    messages = df['message'].dropna().astype(str)

    vectorizer = CountVectorizer(stop_words='english', max_df=0.9, min_df=2)
    X = vectorizer.fit_transform(messages)

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    words = vectorizer.get_feature_names_out()

    topics = []
    for idx, topic in enumerate(lda.components_):
        top_words = [words[i] for i in topic.argsort()[-10:]]
        topics.append(f"Topic {idx+1}: " + ", ".join(top_words))

    return topics


# ================== LEADER ==================
def leader_detection(df):
    df = df[df['user'] != 'group_notification']
    counts = df['user'].value_counts()
    return (counts / counts.sum() * 100).sort_values(ascending=False)

def most_busy_users(df):
    # remove system messages
    df = df[df['user'] != 'group_notification']

    # top users
    x = df['user'].value_counts().head()

    # percentage contribution
    df_percent = (df['user'].value_counts() / df.shape[0] * 100).round(2).reset_index()
    df_percent.columns = ['name', 'percent']

    return x, df_percent

def week_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['day_name'].value_counts()

def month_activity_map(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    return df['month'].value_counts()

def emoji_helper(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []

    for message in df['message']:
        if isinstance(message, str):
            emojis.extend([c for c in message if c in emoji.EMOJI_DATA])

    if len(emojis) == 0:
        return pd.DataFrame(columns=['emoji', 'count'])

    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(emojis)),
                            columns=['emoji', 'count'])

    return emoji_df

def generate_insights(df):
    insights = []

    df = df[df['user'] != 'group_notification']

    if df.empty:
        return ["Not enough data"]

    # Most active user
    most_active = df['user'].value_counts().idxmax()
    insights.append(f"Most active user: {most_active}")

    # Peak day
    peak_day = df['day_name'].value_counts().idxmax()
    insights.append(f"Peak activity day: {peak_day}")

    # Peak month
    peak_month = df['month'].value_counts().idxmax()
    insights.append(f"Peak activity month: {peak_month}")

    # Avg messages per day
    avg_msgs = df.groupby('only_date').size().mean()
    insights.append(f"Average messages per day: {round(avg_msgs)}")

    return insights


def sentiment_timeline(selected_user, df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    analyzer = SentimentIntensityAnalyzer()

    scores = []
    dates = []

    for _, row in df.iterrows():
        msg = row['message']
        if isinstance(msg, str):
            score = analyzer.polarity_scores(msg)['compound']
            scores.append(score)
            dates.append(row['only_date'])

    if len(scores) == 0:
        return pd.DataFrame(columns=['date', 'sentiment'])

    temp_df = pd.DataFrame({'date': dates, 'sentiment': scores})
    return temp_df.groupby('date')['sentiment'].mean().reset_index()

def get_top_influencer(metrics):
    pagerank = metrics.get('pagerank', {})
    if not pagerank:
        return None
    return max(pagerank, key=pagerank.get)