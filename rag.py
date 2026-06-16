import requests
import json
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(url="http://localhost:6333")

# client.delete_collection(collection_name="demo")

if not client.collection_exists(collection_name="demo"):
    client.create_collection(
        collection_name="demo",
        vectors_config=VectorParams(size=2560, distance=Distance.COSINE),
    )

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f"Instruct: {task_description}\nQuery:{query}"

def generate_response(prompt: str):
    return requests.post(
        url="http://localhost:11434/api/generate",
        json={
            "model": "qwen3.5:4b",
            "prompt": prompt,
            "stream": True,
        },
        stream=True,
    )


dummy_data = [
    "My name is Tom",
    "I have no teeth",
    "I have no friends",
    "I love playing clash royale",
]

def main():
    # for i, text in enumerate(dummy_data):
    #     response = requests.post(
    #         url="http://localhost:11434/api/embed",
    #         json={"model": "qwen3-embedding:4b", "input": text},
    #     )
    #     data = response.json()
    #     embeddings = data["embeddings"][0]
    #     client.upsert(
    #         collection_name="demo",
    #         wait=True,
    #         points=[PointStruct(id=i, vector=embeddings, payload={"text": text})]
    #     )
    
    # Each query must come with a one-sentence instruction that describes the task
    #task = "Retrieve relevant Japan travel information, transportation details, attractions, accommodations, food recommendations, " \
    #"and itinerary guidance that answer the user's question"
    task = "Retrieve the passage that contains the factual answer to the user's question."
    prompt = input("Prompt: ")
    adjusted_embed_prompt = get_detailed_instruct(task, prompt)

    response = requests.post(
        url="http://localhost:11434/api/embed",
        json={"model": "qwen3-embedding:4b", "input": adjusted_embed_prompt},
    )
    data=response.json()
    embeddings = data["embeddings"][0]

    results = client.query_points(
        collection_name="demo",
        query=embeddings,
        with_payload=True,
        limit=2
    )

    relevant_passages = "\n".join(
    f"[Score: {point.score:.4f}] {point.payload['text']}"
    for point in results.points
)
    
    augmented_prompt = f"""
        You are answering a question using retrieved context.

        Context:
        {relevant_passages}

        Question:
        {prompt}

        Answer the question using the context above.
        If the answer is not contained in the context, say so.
    """

    response = generate_response(augmented_prompt)

    print("\nAnswer: ", end="", flush=True)

    for line in response.iter_lines():
        if line:
            result = json.loads(line.decode("utf-8"))

            generated_text = result.get("response", "")
            print(generated_text, end="", flush=True)

    print()

if __name__ == "__main__":
    main()