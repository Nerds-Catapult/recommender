import openai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os 

openai.api_key = os.getenv("OPENAI_API_KEY")

data = pd.read_json("data.json")
data = data.dropna()
corpus = data["description"].tolist()


vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)

similarity_matrix = cosine_similarity(X)

user_input = "Zola dresses"
response = openai.completions.create(
    engine="text-davinci-002",
    prompt=f"reccomend products similar to {user_input}",
    max_tokens=200,
)

recommendations = response.choices[0]["text"].splitlines()

print("top 5 Recommendations: ")
for i, recommendations in enumerate(recommendations[:5]):
    print(f"{i+1}. {recommendations}")
