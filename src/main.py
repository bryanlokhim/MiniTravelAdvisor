import os
import json
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

# Load destination data
with open("data/destinations.json", "r") as f:
    destinations = json.load(f)

# Initialize Chroma vector database
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(name="destinations")

# Initialize embedding function
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

# Add destinations to Chroma (only once, check if collection is empty)
if collection.count() == 0:
    for i, dest in enumerate(destinations):
        collection.add(
            documents=[dest["description"]],
            metadatas=[{"name": dest["name"], "category": dest["category"], "budget": dest["budget"]}],
            ids=[str(i)]
        )

# Initialize LangChain with Ollama
llm = Ollama(model="llama2", base_url="http://127.0.0.1:11435")
prompt_template = PromptTemplate(
    input_variables=["query", "destination"],
    template="User asked for: '{query}'. Recommend this destination: {destination}. Provide a short reason why it matches."
)

chain = LLMChain(llm=llm, prompt=prompt_template)

def get_recommendation(query: str) -> str:
    # Query Chroma for the most relevant destination
    results = collection.query(query_texts=[query], n_results=1)
    if not results["documents"][0]:
        return "Sorry, no matching destinations found."
    
    destination = results["metadatas"][0][0]["name"]
    description = results["documents"][0][0]
    
    # Generate recommendation with LangChain
    response = chain.run(query=query, destination=f"{destination}: {description}")
    return response

# Simple CLI interface
def main():
    print("Welcome to MiniTravelBuddy! Enter your travel preference (e.g., 'beach vacation') or 'quit' to exit.")
    while True:
        query = input("Preference: ")
        if query.lower() == "quit":
            break
        recommendation = get_recommendation(query)
        print("\nRecommendation:\n", recommendation, "\n")

if __name__ == "__main__":
    main()
