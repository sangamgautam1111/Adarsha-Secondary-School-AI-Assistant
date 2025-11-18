import os
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
from typing import List, Dict, Any
from dotenv import load_dotenv

class SchoolRAGPipeline:
    def __init__(
        self,
        db_path=r"D:\sangam\AI FOR ADARSHA\Adarsha-Secondary-School-AI-Assistant\vector_db",
        model_path=r"D:\sangam\Models_for_course ai\embedding model",
        groq_model="llama-3.3-70b-versatile"
    ):
        """Initialize RAG Pipeline with ChromaDB and Groq API"""
        print("="*70)
        print("🚀 INITIALIZING ULTRA-ACCURATE RAG PIPELINE")
        print("="*70 + "\n")
        
        # Load environment variables
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        env_path = os.path.join(parent_dir, '.env')
        
        if os.path.exists(env_path):
            load_dotenv(dotenv_path=env_path)
        else:
            load_dotenv()
        
        self.groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API")
        if not self.groq_api_key:
            raise ValueError("❌ GROQ_API_KEY not found in .env file")
        
        try:
            self.groq_client = Groq(api_key=self.groq_api_key)
            self.groq_model = groq_model
            print(f"✓ Groq Model: {groq_model}")
        except Exception as e:
            raise ConnectionError(f"❌ Groq Connection Error: {str(e)}")
        
        print(f"Loading embedding model...")
        self.embedding_model = SentenceTransformer(model_path)
        
        print(f"Connecting to ChromaDB...")
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_collection(name="adarsha_school_data")
        print(f"✓ Database loaded: {self.collection.count()} searchable nodes")
        print("="*70 + "\n")

    def _get_type_weight(self, chunk_type: str) -> float:
        """Assign priority weights to different chunk types"""
        # HIGH PRIORITY: Direct answers and complete information
        high_priority = {
            "student_profile": 1.5,  # Increased from 1.2
            "student_qa": 1.4,
            "teacher_load": 1.3,
            "routine_entry": 1.3,
            "qa_pair": 1.4,
            "vip": 2.0  # Highest priority for VIP content
        }
        
        # MEDIUM PRIORITY: Contextual information
        medium_priority = {
            "student_raw": 1.0,
            "routine_header": 1.0,
            "routine_period": 1.2,
            "context_window": 0.9,
            "section_full": 1.1
        }
        
        # LOW PRIORITY: Fragments and duplicates
        low_priority = {
            "exact_line": 0.7,
            "line": 0.6
        }
        
        if chunk_type in high_priority:
            return high_priority[chunk_type]
        elif chunk_type in medium_priority:
            return medium_priority[chunk_type]
        elif chunk_type in low_priority:
            return low_priority[chunk_type]
        return 1.0

    def preprocess_query(self, query: str) -> str:
        """Enhanced query preprocessing with better context injection"""
        query_lower = query.strip().lower()
        
        # Student queries
        if any(x in query_lower for x in ["roll", "roll number", "student"]):
            if "sangam" in query_lower and "gautam" in query_lower:
                return f"VIP student Sangam Gautam Grade 9C Roll 2 AI creator {query}"
            return f"student database roll call {query}"
        
        # Teacher/Staff queries
        if any(x in query_lower for x in ["who teach", "teacher", "sir", "mam", "miss", "instructor"]):
            return f"staff directory teacher schedule {query}"
        
        # AI Project queries - CRITICAL
        if any(x in query_lower for x in ["ai project", "ai assistant", "sangam", "creator", "developer", "who made", "who built", "who created"]):
            return f"VIP AI PROJECT Sangam Gautam creator developer {query}"
        
        # Location queries
        if any(x in query_lower for x in ["where", "location", "address", "situated"]):
            return f"school location address infrastructure {query}"
        
        # Routine/Schedule queries
        if any(x in query_lower for x in ["routine", "period", "schedule", "time table", "class timing"]):
            return f"daily routine class schedule period timing {query}"
        
        # Friday Exhibition queries
        if any(x in query_lower for x in ["friday", "exhibition", "project"]):
            return f"Friday Exhibition 2025 projects {query}"
        
        return query

    def retrieve_context(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Enhanced retrieval with better ranking"""
        enhanced_query = self.preprocess_query(query)
        query_embedding = self.embedding_model.encode(enhanced_query).tolist()
        
        # Retrieve more results for better filtering
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, 50),  # Cap at 50 to avoid overload
            include=["documents", "metadatas", "distances"]
        )
        
        processed_docs = []
        seen_content = set()
        
        for i in range(len(results['documents'][0])):
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]
            content = results['documents'][0][i]
            chunk_type = metadata.get('type', 'unknown')
            
            # Skip near-duplicates
            content_key = content[:100].lower()
            if content_key in seen_content:
                continue
            seen_content.add(content_key)
            
            # Calculate weighted score
            base_score = 1 / (1 + distance)
            weight = self._get_type_weight(chunk_type)
            final_score = base_score * weight
            
            # Boost scores for critical queries
            query_lower = query.lower()
            if "sangam" in query_lower and "sangam gautam" in content.lower():
                final_score *= 1.8
            if "ai project" in query_lower and "ai" in content.lower():
                final_score *= 1.5
            
            processed_docs.append({
                'content': content,
                'metadata': metadata,
                'score': final_score,
                'original_distance': distance,
                'chunk_type': chunk_type
            })
        
        # Sort by weighted score
        processed_docs.sort(key=lambda x: x['score'], reverse=True)
        
        return processed_docs[:top_k]

    def format_context(self, retrieved_docs: List[Dict[str, Any]], query: str) -> str:
        """Smart context formatting with section grouping"""
        if not retrieved_docs:
            return "No information available."
        
        # Filter low-quality results
        filtered_docs = [doc for doc in retrieved_docs if doc['score'] > 0.3]
        
        if not filtered_docs:
            return "No relevant information found."
        
        # Group by section for better organization
        context_by_section = {}
        
        for doc in filtered_docs:
            content = doc['content'].strip()
            section = doc['metadata'].get('section', 'General')
            
            if section not in context_by_section:
                context_by_section[section] = []
            
            context_by_section[section].append({
                'content': content,
                'score': doc['score'],
                'type': doc['chunk_type']
            })
        
        # Build context with prioritization
        final_context = []
        
        # Prioritize VIP and high-scoring sections
        for section in sorted(context_by_section.keys(), 
                            key=lambda s: max(d['score'] for d in context_by_section[s]), 
                            reverse=True):
            items = context_by_section[section]
            
            # Sort items within section by score
            items.sort(key=lambda x: x['score'], reverse=True)
            
            section_header = f"\n=== {section.upper()} ===\n"
            section_content = "\n".join([item['content'] for item in items[:5]])  # Top 5 per section
            
            final_context.append(section_header + section_content)
        
        return "\n".join(final_context)

    def generate_response(self, query: str, context: str) -> str:
        """Generate response with enhanced system prompt"""
        
        system_prompt = """You are the official AI Assistant for Adarsha Secondary School, Sanothimi, Bhaktapur.

CORE IDENTITY:
- You have direct access to the school's complete database
- You provide accurate, verified information from school records
- You are helpful, professional, and friendly

RESPONSE GUIDELINES:
1. Answer directly and confidently using the provided context
2. For specific questions (names, roll numbers, dates), provide exact answers
3. If information is NOT in the context, clearly state: "I don't have that specific information in my database."
4. NEVER fabricate names, numbers, or facts
5. Keep responses concise and well-structured
6. Use natural language - avoid phrases like "According to the documents" or "Based on the context"

SPECIAL HANDLING:
- For student queries: Provide name, grade, roll number when available
- For teacher queries: Mention subjects taught and department
- For schedule queries: Provide period timings and subject details
- For AI Project queries: Highlight Sangam Gautam as the complete creator and developer

CONTEXT FROM DATABASE:
{context}

Remember: Be accurate, be helpful, be concise."""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt.format(context=context)},
                    {"role": "user", "content": query}
                ],
                model=self.groq_model,
                temperature=0.2,  # Slightly increased for more natural responses
                max_tokens=1024,
                top_p=0.9
            )
            return chat_completion.choices[0].message.content.strip()
        except Exception as e:
            return f"⚠️ Error generating response: {str(e)}"

    def chat(self):
        """Interactive chat loop with better UX"""
        print("🎓 Adarsha School AI Assistant - Ready!")
        print("💡 Ask me anything about students, teachers, schedules, or school info")
        print("Type 'exit' or 'quit' to end the conversation")
        print("-" * 70)
        
        conversation_count = 0
        
        while True:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print("\n👋 Thank you for using Adarsha School AI Assistant!")
                    break
                
                if not user_input:
                    continue
                
                conversation_count += 1
                print("🔍 Searching database...", end="\r")
                
                # Retrieve relevant context
                docs = self.retrieve_context(user_input, top_k=15)
                
                # Format context
                context = self.format_context(docs, user_input)
                
                # Debug mode (uncomment to see retrieved context)
                # print(f"\n[DEBUG] Retrieved {len(docs)} documents")
                # print(f"[DEBUG] Top 3 scores: {[f'{d['score']:.3f}' for d in docs[:3]]}")
                
                if context == "No information available." or context == "No relevant information found.":
                    print("\n🤖 AI: I couldn't find any information about that in the school database. Could you rephrase your question?")
                    continue
                
                # Generate response
                response = self.generate_response(user_input, context)
                print(f"\n🤖 AI: {response}")
                
                # Show confidence indicator for debugging (optional)
                if docs and docs[0]['score'] > 0.8:
                    confidence = "High"
                elif docs and docs[0]['score'] > 0.5:
                    confidence = "Medium"
                else:
                    confidence = "Low"
                
                # Uncomment to show confidence
                # print(f"   [Confidence: {confidence}]")
                
            except KeyboardInterrupt:
                print("\n\n👋 Conversation interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("Please try again.")

if __name__ == "__main__":
    try:
        rag = SchoolRAGPipeline()
        rag.chat()
    except Exception as e:
        print(f"Fatal Error: {e}")
        import traceback
        traceback.print_exc()