import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from typing import List, Dict, Any
from dotenv import load_dotenv

class SchoolRAGPipeline:
    def __init__(
        self,
        db_path=r"C:\Users\USER\Documents\GitHub\Adarsha-Secondary-School-AI-Assistant\vector_db",
        model_path=r"C:\Users\USER\Desktop\models",
        groq_model="llama-3.3-70b-versatile"
    ):
        """Initialize RAG Pipeline with ChromaDB and Groq API"""
        print("="*70)
        print("INITIALIZING ADARSHA SCHOOL RAG PIPELINE")
        print("="*70 + "\n")
        
        # Load .env from parent directory using dotenv
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        env_path = os.path.join(parent_dir, '.env')
        
        print(f"Looking for .env at: {env_path}")
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
            print("✓ .env file loaded\n")
        else:
            print("⚠️  .env file not found, trying current directory...")
            load_dotenv()  # Try current directory
        
        # Get API key
        self.groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
        
        if not self.groq_api_key:
            raise ValueError(
                " GROQ_API_KEY not found!\n"
                f"Please add GROQ_API_KEY to your .env file at: {env_path}\n"
                "Format: GROQ_API_KEY=gsk_your_key_here"
            )
        
        # Verify API key format
        if not self.groq_api_key.startswith("gsk_"):
            raise ValueError(" Invalid Groq API key format! Key should start with 'gsk_'")
        
        print(f"✓ Groq API key loaded: {self.groq_api_key[:8]}...{self.groq_api_key[-4:]}")
        
        # Initialize Groq client
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            self.groq_model = groq_model
            print(f"✓ Groq client initialized")
            print(f"✓ Using model: {groq_model}\n")
            
        except Exception as e:
            raise ConnectionError(f"❌ Failed to connect to Groq API: {str(e)}")
        
        # Load embedding model
        print(f"Loading embedding model from: {model_path}")
        self.embedding_model = SentenceTransformer(model_path)
        print("✓ Embedding model loaded\n")
        
        # Connect to ChromaDB
        print(f"Connecting to ChromaDB at: {db_path}")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="adarsha_school_data")
        print(f"✓ Connected to collection with {self.collection.count()} documents\n")
        
        print("="*70)
        print("RAG PIPELINE READY - Start Chatting!")
        print("="*70 + "\n")
    
    def retrieve_context(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from vector database"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode(query).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        retrieved_docs = []
        for i in range(len(results['documents'][0])):
            retrieved_docs.append({
                'content': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_docs
    
    def format_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            context_parts.append(f"[Document {i}]")
            context_parts.append(f"Content: {doc['content']}")
            context_parts.append(f"Section: {doc['metadata'].get('section', 'N/A')}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_response(
        self,
        query: str,
        context: str,
        temperature: float = 0.3,
        max_tokens: int = 1024
    ) -> str:
        """Generate response using Groq API with retrieved context"""
        
        system_prompt = """You are an Adarsha School AI assistant made by Sangam Gautam for Adarsha Secondary School in Nepal. 
Your role is to provide accurate, helpful information about the school based on the provided context.

Guidelines:
- Answer questions accurately using ONLY the information from the provided context
- If the context doesn't contain enough information, say so politely
- Be friendly, professional, and concise
- Use Nepali names and terms correctly
- For student/staff queries, provide relevant details from the context
- If asked about something not in the context, admit you don't have that information"""

        user_prompt = f"""Context Information:
{context}

Question: {query}

Please provide a clear and accurate answer based on the context above."""

        try:
            # Call Groq API
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                model=self.groq_model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            response = chat_completion.choices[0].message.content
            return response
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                return "⚠️ Rate limit exceeded. Please wait a moment and try again."
            return f"❌ Error: {error_msg[:200]}"
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        temperature: float = 0.3
    ) -> str:
        """
        Complete RAG pipeline: retrieve, format, and generate
        Returns just the answer as a string
        """
        # Step 1: Retrieve relevant context
        retrieved_docs = self.retrieve_context(question, top_k=top_k)
        
        # Step 2: Format context
        formatted_context = self.format_context(retrieved_docs)
        
        # Step 3: Generate response
        response = self.generate_response(
            query=question,
            context=formatted_context,
            temperature=temperature
        )
        
        return response
    
    def chat(self):
        """Simple chat interface - just ask and get answers"""
        print("🎓 ADARSHA SCHOOL AI ASSISTANT")
        print("="*70)
        print("Ask me anything about Adarsha Secondary School!")
        print("Type 'exit', 'quit', or 'bye' to stop chatting.")
        print("="*70 + "\n")
        
        while True:
            try:
                # Get user input
                user_input = input("💬 You: ").strip()
                
                # Check for empty input
                if not user_input:
                    continue
                
                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'bye', 'q']:
                    print("\n👋 Thank you for chatting! Goodbye!")
                    break
                
                # Get answer from RAG pipeline
                print("\n🤖 Assistant: ", end="", flush=True)
                answer = self.query(user_input)
                print(answer)
                print()
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}\n")


def main():
    """Main function - directly starts chat mode"""
    try:
        # Initialize the RAG pipeline
        rag = SchoolRAGPipeline()
        
        # Start chatting
        rag.chat()
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure .env file exists with GROQ_API_KEY")
        print("2. Check if python-dotenv is installed: pip install python-dotenv")
        print("3. Verify your API key starts with 'gsk_'")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()