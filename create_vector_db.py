import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import os

class SchoolDataStore:
    def __init__(self, db_path="D:/sangam/Adarsha AI Assistant/Adarsha-Secondary-School-AI-Assistant/vector_db"):
        
        self.db_path = db_path
        
        
        os.makedirs(db_path, exist_ok=True)
        
        
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="school_data",
            metadata={"description": "Adarsha Secondary School Information"}
        )
    
    def chunk_data(self, data, prefix=""):
        """Recursively chunk nested JSON data into manageable pieces"""
        chunks = []
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_prefix = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, (dict, list)):
                    chunks.extend(self.chunk_data(value, current_prefix))
                else:
                    chunks.append({
                        "id": current_prefix,
                        "content": f"{current_prefix}: {json.dumps(value, ensure_ascii=False)}",
                        "metadata": {"section": prefix or "root", "key": key}
                    })
        
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                current_prefix = f"{prefix}[{idx}]"
                if isinstance(item, (dict, list)):
                    chunks.extend(self.chunk_data(item, current_prefix))
                else:
                    chunks.append({
                        "id": current_prefix,
                        "content": f"{current_prefix}: {json.dumps(item, ensure_ascii=False)}",
                        "metadata": {"section": prefix or "root", "index": idx}
                    })
        
        return chunks
    
    def load_and_store(self, json_file_path):
        """Load JSON file and store in ChromaDB"""
        print(f"Loading data from {json_file_path}...")
        
        with open(json_file_path, 'r', encoding='utf-8') as f:
            school_data = json.load(f)
        
        print("Chunking data...")
        chunks = self.chunk_data(school_data)
        
        print(f"Created {len(chunks)} chunks")
        print("Storing in ChromaDB...")
        
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            
            ids = [chunk["id"] for chunk in batch]
            documents = [chunk["content"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            
            
            embeddings = self.embedding_model.encode(documents).tolist()
            
            
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            print(f"Stored batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        
        print(f"Successfully stored {len(chunks)} chunks in ChromaDB")
        print(f"Database location: {self.db_path}")
    
    def query_test(self, query_text, n_results=5):
        """Test query to verify data storage"""
        print(f"\nTest query: '{query_text}'")
        
        
        query_embedding = self.embedding_model.encode([query_text]).tolist()
        
        
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        print(f"\nTop {n_results} results:")
        for i, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
            print(f"\n{i+1}. (Distance: {distance:.4f})")
            print(f"{doc[:200]}...")
        
        return results


if __name__ == "__main__":
    
    store = SchoolDataStore()
    
    
    json_file = r"D:\sangam\Adarsha AI Assistant\Adarsha-Secondary-School-AI-Assistant\json_data"
    store.load_and_store(json_file)
    
    
    print("\n" + "="*50)
    print("Testing data retrieval...")
    print("="*50)
    
    store.query_test("What is the school establishment date?")
    store.query_test("Tell me about computer engineering program")
    store.query_test("Who is the principal?")