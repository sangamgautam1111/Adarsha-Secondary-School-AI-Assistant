import os
from dotenv import load_dotenv
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

class SchoolRAGPipeline:
    def __init__(self, db_path="D:/sangam/Adarsha AI Assistant/Adarsha-Secondary-School-AI-Assistant/vector_db"):
        """Initialize RAG Pipeline"""
        load_dotenv()
        
        # Initialize Groq client
        self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        
        # Initialize ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection("school_data")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # System prompt
        self.system_prompt = """You are an AI assistant for Adarsha Secondary School in Sanothimi, Bhaktapur, Nepal. 
Your role is to provide accurate, helpful information about the school based on the context provided.

Guidelines:
- Answer questions accurately using the provided context
- Be friendly and professional
- If information is not in the context, politely say you don't have that specific information
- Provide detailed answers when appropriate
- Use both English and Nepali terms when relevant
- Focus on being helpful to students, parents, and visitors

Always base your answers on the context provided."""
    
    def retrieve_context(self, query, n_results=5):
        """Retrieve relevant context from ChromaDB"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query]).tolist()
        
        # Query collection
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_results
        )
        
        # Combine documents into context
        context = "\n\n".join(results['documents'][0])
        return context, results
    
    def generate_response(self, query, context, model="llama-3.1-70b-versatile"):
        """Generate response using Groq API"""
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"""Context from school database:
{context}

User Question: {query}

Please provide a comprehensive answer based on the context above."""}
        ]
        
        try:
            response = self.groq_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9
            )
            
            return response.choices[0].message.content
        
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def query(self, user_question, n_results=5, model="llama-3.1-70b-versatile", verbose=False):
        """Main query function - retrieve and generate"""
        print(f"\nQuestion: {user_question}")
        print("-" * 50)
        
        # Retrieve context
        context, results = self.retrieve_context(user_question, n_results)
        
        if verbose:
            print("\nRetrieved Context:")
            print(context[:500] + "...")
            print("\n" + "-" * 50)
        
        # Generate response
        response = self.generate_response(user_question, context, model)
        
        print("\nAnswer:")
        print(response)
        print("\n" + "=" * 50)
        
        return {
            "question": user_question,
            "answer": response,
            "context": context,
            "sources": results['ids'][0]
        }
    
    def chat(self, model="llama-3.1-70b-versatile"):
        """Interactive chat interface"""
        print("\n" + "=" * 50)
        print("Adarsha School AI Assistant")
        print("Type 'quit' or 'exit' to end the conversation")
        print("=" * 50 + "\n")
        
        conversation_history = []
        
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using Adarsha School AI Assistant!")
                break
            
            if not user_input:
                continue
            
            # Retrieve context
            context, _ = self.retrieve_context(user_input, n_results=5)
            
            # Build messages with conversation history
            messages = [{"role": "system", "content": self.system_prompt}]
            
            # Add conversation history
            for exchange in conversation_history[-3:]:  # Keep last 3 exchanges
                messages.append({"role": "user", "content": exchange["user"]})
                messages.append({"role": "assistant", "content": exchange["assistant"]})
            
            # Add current query with context
            messages.append({
                "role": "user",
                "content": f"""Context from school database:
{context}

User Question: {user_input}

Please provide a comprehensive answer based on the context above."""
            })
            
            try:
                # Generate response
                response = self.groq_client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9
                )
                
                assistant_response = response.choices[0].message.content
                print(f"\nAssistant: {assistant_response}")
                
                # Save to conversation history
                conversation_history.append({
                    "user": user_input,
                    "assistant": assistant_response
                })
            
            except Exception as e:
                print(f"\nError: {str(e)}")


if __name__ == "__main__":
    # Initialize RAG Pipeline
    rag = SchoolRAGPipeline()
    
    # Example queries
    print("\nRunning example queries...")
    print("=" * 50)
    
    
    rag.query("When was the school established and by whom?")
    
    
    
    
    print("\n\nStarting interactive chat mode...")
    rag.chat()