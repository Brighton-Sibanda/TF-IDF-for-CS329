from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def search_documents(documents, query, top_n):
    
    # Initialize a TfidfVectorizer
    vectorizer = TfidfVectorizer(stop_words='english')
    
    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    # Transform the query to the same TF-IDF space as the documents
    query_tfidf = vectorizer.transform([query])
    
    # Compute cosine similarity between the query and all documents
    similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    # Get the indices of the top_n documents in descending order of similarity
    top_indices = np.argsort(similarities)[::-1][:top_n]
    
    # Return a list of tuples (index, similarity score, document text)
    return [(index, similarities[index], documents[index]) for index in top_indices]

# We can test here
documents = [
    "Data science is an inter-disciplinary field that uses scientific methods, processes, algorithms and systems to extract knowledge and insights from structured and unstructured data.",
    "Machine learning is a type of data analysis that automates analytical model building. It is a branch of artificial intelligence based on the idea that systems can learn from data, identify patterns and make decisions with minimal human intervention.",
    "The quick brown fox jumps over the lazy dog."
]

query = "data science insights"
top_n = 2
result = search_documents(documents, query, top_n)

for idx, score, doc in result:
    print(f"Document {idx} (Score: {score:.4f}): {doc[:100]}...")  # Display the beginning of each document
