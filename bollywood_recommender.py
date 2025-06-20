

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import requests

# Load datasets
main_df = pd.read_csv("movies.csv")  
bollywood_df = pd.read_csv("bollywood_2024_25.csv")  
bollywood_df['overview'] = "Bollywood movie — overview not available."


# Combine them
combined_df = pd.concat([main_df[['title', 'overview']], bollywood_df[['title', 'overview']]], ignore_index=True)

# Fill any missing descriptions
combined_df['overview'] = combined_df['overview'].fillna('')

# Vectorize overview text
vectorizer = CountVectorizer(stop_words='english')
vectors = vectorizer.fit_transform(combined_df['overview'])
similarity = cosine_similarity(vectors)

# Recommendation function
def recommend(movie_title):
    movie_title = movie_title.lower()
    indices = combined_df[combined_df['title'].str.lower() == movie_title].index
    if len(indices) == 0:
        return []
    index = indices[0]
    distances = list(enumerate(similarity[index]))
    distances = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [combined_df.iloc[i[0]].title for i in distances]

# Streamlit UI
st.title("Movie Recommender (Hollywood + Bollywood)")
movie_list = combined_df['title'].values
selected_movie = st.selectbox("Choose a movie:", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)
    for title in recommendations:
        st.write(title)
