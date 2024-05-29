import pandas as pd
import umap
from sklearn.pipeline import make_pipeline

# pip install "embetter[text]"
from embetter.text import SentenceEncoder

# Build a sentence encoder pipeline with UMAP at the end.
text_emb_pipeline = make_pipeline(
    SentenceEncoder('all-MiniLM-L6-v2'),
    umap.UMAP()  # Instantiate UMAP object
)

# Read the CSV file
data = pd.read_csv("Corona_NLP_test.csv")

# Print column names to identify the correct column name
print("Column names:", data.columns)

# Load sentences (replace 'sentences' with the correct column name)
column_name = 'OriginalTweet'  # Assuming the correct column name is 'OriginalTweet'
sentences = list(data[column_name])

# Calculate embeddings
X_tfm = text_emb_pipeline.fit_transform(sentences)

# Write to disk. Note! Text column must be named "text"
df = pd.DataFrame({"text": sentences})
df['x'] = X_tfm[:, 0]
df['y'] = X_tfm[:, 1]
df.to_csv("ready.csv", index=False)
