import os
import vertexai
from vertexai.preview.language_models import TextEmbeddingModel

project_id = "YOUR_GCP_PROJECT_ID"   # <-- PUT YOUR REAL PROJECT ID HERE
location = "us-central1"

# Initialize Vertex AI client
vertexai.init(project=project_id, location=location)

try:
    model = TextEmbeddingModel.from_pretrained("text-embedding-004")
    embeddings = model.get_embeddings(["test embedding for credit policy"])
    print("OK – Vertex embeddings working. Dim:", len(embeddings[0].values))
except Exception as e:
    print("Error:", type(e).__name__, e)
