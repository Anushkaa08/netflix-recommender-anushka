import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Load and process data
df = pd.read_csv("movies.csv")
df['genres'] = df['genres'].str.replace('|', ' ', regex=False)

vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df['genres'])
similarity = cosine_similarity(genre_matrix)

# Recommendation function
def get_recommendations(title):
    if title not in df['title'].values:
        return ["❌ Movie not found."]
    idx = df[df['title'] == title].index[0]
    sim_scores = list(enumerate(similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]
    return [df.iloc[i[0]]['title'] for i in sim_scores]

# --- Streamlit App Interface ---
st.title("🎬 Netflix-style Movie Recommender")
st.write("Pick a movie and we'll show you what to watch next!")

movie_choice = st.selectbox("Choose a movie:", df['title'].values)

if st.button("Recommend"):
    recommendations = get_recommendations(movie_choice)
    st.subheader("🎥 Recommended Movies:")
    for rec in recommendations:
        st.write(f"- {rec}")
