from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def extract_keywords(text, num_keywords):
    # Convert the text into a list (as TfidfVectorizer expects a list of documents)
    documents = [text]

    # Initialize the TF-IDF Vectorizer with stop words removal
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')

    # Fit and transform the documents
    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)

    # Get the feature names (words)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    # Get the TF-IDF scores for the words in the text
    tfidf_scores = tfidf_matrix.toarray().flatten()

    # Get the indices of the top 'num_keywords' scores
    top_indices = np.argsort(tfidf_scores)[::-1][:num_keywords]

    # Get the top keywords using the indices
    top_keywords = [feature_names[index] for index in top_indices]

    return top_keywords

# We can test here
text = """
In the matter before the court today, the plaintiff, John Doe, alleges that the defendant, Jane Smith, breached the terms of their contract. 
The contract in question pertains to the sale of a property located at 123 Maple Street. The plaintiff contends that the defendant failed to 
disclose significant structural issues with the property, thereby violating the terms of their agreement. The defendant, however, argues that 
the property was sold 'as-is' and that the plaintiff had ample opportunity to inspect the property prior to the sale. During today's proceedings, 
the court will hear testimony from both parties as well as from expert witnesses. The plaintiff's attorney will present evidence, including a 
detailed inspection report, while the defendant's counsel will cross-examine the plaintiff and challenge the credibility of the inspection report. 
The court will also consider relevant case law and legal precedents before rendering a decision. The outcome of this case will hinge on the 
interpretation of the contractual obligations and the evidence presented.
"""

num_keywords = 5
keywords = extract_keywords(text, num_keywords)
print(keywords)