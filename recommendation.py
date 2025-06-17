import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the movie data
df = pd.read_csv("movies.csv")

# Preprocess genres
df['genres'] = df['genres'].str.replace('|', ' ', regex=False)

# Vectorize the genres
vectorizer = CountVectorizer()
genre_matrix = vectorizer.fit_transform(df['genres'])

# Compute similarity
similarity = cosine_similarity(genre_matrix)

def recommend(title):
    if title not in df['title'].values:
        print("❌ Movie not found.")
        return

    idx = df[df['title'] == title].index[0]
    scores = list(enumerate(similarity[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]

    print(f"\n✅ Recommendations for '{title}':")
    for i in scores:
        print(f"🎥 {df.iloc[i[0]]['title']}")

# Example call
recommend("The Matrix")
