import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient  
from qdrant_client.models import VectorParams,Distance,PointStruct
from uuid import uuid4
from PIL import Image
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import os
from pathlib import Path
PATH_IMAGES_DATA = r"D:\PTIT\AI\PROJECTS\AI_Engine_Search\images_data"
batch_size = 150
collection_name="Vector_data_collection"
model = SentenceTransformer( 
    model_name_or_path="clip-ViT-B-32",
    device= device,
    truncate_dim=256,
    trust_remote_code=False
    )

client = QdrantClient(path="imagesVector_store")
def embedding_Database():
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name= collection_name,
            vectors_config= VectorParams(size=256, distance= Distance.COSINE)
        )
        images_path = [os.path.join(PATH_IMAGES_DATA,f ) for f in os.listdir(PATH_IMAGES_DATA)]
        for i in range(0,len(images_path),batch_size):
            batch_path = images_path[i:i+batch_size]
            images_open = [Image.open(img).convert("RGB") for img in batch_path]
            embedding = model.encode(images_open, normalize_embeddings= True, batch_size= batch_size,show_progress_bar=True)
            client.upsert(
                collection_name=collection_name,
                points= [PointStruct(id=str(uuid4()),vector= embedding[j],payload={"path":batch_path[j]}) for j in range(len(batch_path))]
                )
            del images_open
def embedding_and_result(query_input):
    query_embedding = model.encode(query_input, normalize_embeddings= True, device= device)
    result = client.query_points(
        collection_name = collection_name,
        query= query_embedding,
        limit= 3
    ).points
    return result
if __name__ == "__main__":
    query_input = input("Enter your query: ")
    result = embedding_and_result(query_input)
    plt.figure(figsize=(10,5))
    for i,r in enumerate(result):
        plt.subplot(1,len(result),i+1)
        img = Image.open(r.payload["path"])
        plt.imshow(img)
        plt.title(f"score: {r.score:.2f} \n {Path(r.payload['path']).stem}")
    plt.show()
    client.close()