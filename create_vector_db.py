import os
import re
import uuid
import hashlib
from typing import List, Dict, Any
import chromadb
from sentence_transformers import SentenceTransformer

# ================= CONFIGURATION =================
# UPDATE THESE PATHS TO MATCH YOUR PC EXACTLY
BASE_DIR = r"D:\sangam\AI FOR ADARSHA\Adarsha-Secondary-School-AI-Assistant"
TEXT_FILE_PATH = os.path.join(BASE_DIR, "text_data", "data.txt")
DB_PATH = os.path.join(BASE_DIR, "vector_db")
MODEL_PATH = r"D:\sangam\Models_for_course ai\embedding model"
# =================================================

class NuclearVectorStore:
    """
    The 'Nuclear Option' for Vector Databases.
    Strategy: Index EVERYTHING. Redundancy is good. Context is King.
    """
    
    def __init__(self):
        print(f"{'='*60}")
        print(f"☢️  INITIALIZING NUCLEAR VECTOR STORE GENERATOR")
        print(f"{'='*60}")
        
        # 1. Setup Paths
        self.db_path = DB_PATH
        self.model_path = MODEL_PATH
        os.makedirs(self.db_path, exist_ok=True)
        
        # 2. Initialize ChromaDB
        print(f"📂 Database Path: {self.db_path}")
        self.client = chromadb.PersistentClient(path=self.db_path)
        
        # 3. Load Model
        print(f"🤖 Loading Embedding Model from: {self.model_path}")
        # Trust remote code needed for some models, standard is fine for others
        self.model = SentenceTransformer(self.model_path)
        
        # 4. Reset Collection (Delete old data to ensure 100% fresh start)
        try:
            self.client.delete_collection("adarsha_school_data")
            print("🗑️  Deleted old collection to prevent duplicates.")
        except:
            pass
            
        self.collection = self.client.get_or_create_collection(
            name="adarsha_school_data",
            metadata={"hnsw:space": "cosine"}
        )
        print("✅ Collection ready.")

    def _generate_id(self, prefix: str, content: str) -> str:
        """Generates a unique ID based on content hash to prevent duplicates but allow distinct entries."""
        content_hash = hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
        random_suffix = str(uuid.uuid4())[:6]
        return f"{prefix}_{content_hash}_{random_suffix}"

    def _clean_text(self, text: str) -> str:
        return re.sub(r'\s+', ' ', text).strip()

    # ==========================================================
    # 🛠️ PARSING LOGIC - THE "1000% COVERAGE" STRATEGY
    # ==========================================================

    def parse_student_section(self, text: str, section_name: str) -> List[Dict]:
        """
        Extracts EVERY student individually.
        Format handled: "Shurti Kumari Yadav (Roll 1), Samir Kattel (Roll 2)..."
        """
        chunks = []
        # Split by "Grade" headers first
        grade_blocks = re.split(r'\*\*Grade', text)
        
        for block in grade_blocks:
            if not block.strip(): continue
            
            # Extract Grade Name (e.g., "10A", "ECD")
            header_match = re.match(r'\s*([^\*]+?)\s*Students:\*\*', block)
            grade_name = header_match.group(1).strip() if header_match else "Unknown Grade"
            
            # Now split the rest by COMMAS to get individual students
            # This is crucial for your comma-separated list
            content_body = block[header_match.end():] if header_match else block
            
            # Regex to split by comma but keep "Name (Roll X)" together
            students = content_body.split(',')
            
            for raw_student in students:
                student_text = self._clean_text(raw_student)
                if len(student_text) < 3: continue
                
                # Extract Name and Roll if possible
                match = re.search(r'([^(]+)\(Roll\s*(\d+)\)', student_text)
                
                # STRATEGY 1: The Raw Text
                chunks.append({
                    "id": self._generate_id("stu_raw", student_text),
                    "text": f"Grade {grade_name} Student: {student_text}",
                    "meta": {"type": "student_raw", "section": section_name, "grade": grade_name}
                })
                
                # STRATEGY 2: The "Who is?" Format (if regex matches)
                if match:
                    name = match.group(1).strip()
                    roll = match.group(2).strip()
                    chunks.append({
                        "id": self._generate_id("stu_qa", name),
                        "text": f"Who is Roll Number {roll} in Grade {grade_name}? It is {name}.",
                        "meta": {"type": "student_qa", "section": section_name, "roll": roll, "grade": grade_name}
                    })
                    chunks.append({
                        "id": self._generate_id("stu_id", name),
                        "text": f"Student Profile: Name: {name}, Grade: {grade_name}, Roll: {roll}.",
                        "meta": {"type": "student_profile", "section": section_name}
                    })

        return chunks

    def parse_routine_section(self, text: str, section_name: str) -> List[Dict]:
        """Extracts routine data line by line and by period."""
        chunks = []
        lines = text.split('\n')
        current_grade = "General"
        
        for line in lines:
            line = self._clean_text(line)
            if not line: continue
            
            # Update Context (Grade Header)
            if "**Grade" in line:
                current_grade = line.replace('*', '').replace(':', '').strip()
                chunks.append({
                    "id": self._generate_id("routine_head", line),
                    "text": f"Routine Header: {line}",
                    "meta": {"type": "routine_header", "section": section_name}
                })
                continue

            # STRATEGY: Period Parsing
            if "Period" in line:
                chunks.append({
                    "id": self._generate_id("routine_period", line),
                    "text": f"{current_grade} Schedule: {line}",
                    "meta": {"type": "routine_entry", "section": section_name, "context": current_grade}
                })
                
                # Teacher Extraction (simple heuristic looking for parenthesis)
                if "(" in line and ")" in line:
                    teacher = line[line.find("(")+1:line.find(")")]
                    subject = line[:line.find("(")].replace("Period", "").split(":")[-1].strip()
                    chunks.append({
                        "id": self._generate_id("routine_teacher", teacher),
                        "text": f"Teacher {teacher} teaches {subject} in {current_grade}.",
                        "meta": {"type": "teacher_load", "section": section_name}
                    })

        return chunks

    def parse_generic_brute_force(self, text: str, section_name: str) -> List[Dict]:
        """
        The fallback strategy:
        1. Index every single line.
        2. Index sliding windows of 3 sentences (context).
        """
        chunks = []
        
        # 1. Line by Line (The "Don't miss anything" layer)
        lines = text.split('\n')
        for line in lines:
            clean = self._clean_text(line)
            if len(clean) > 5: # Ignore empty lines
                chunks.append({
                    "id": self._generate_id("line", clean),
                    "text": f"{section_name}: {clean}",
                    "meta": {"type": "exact_line", "section": section_name}
                })
                
                # Key-Value Extraction (e.g., "Official Name: Adarsha...")
                if ":" in clean and len(clean) < 100:
                    key, val = clean.split(":", 1)
                    chunks.append({
                        "id": self._generate_id("kv", clean),
                        "text": f"Question: What is the {key.strip()}? Answer: {val.strip()}",
                        "meta": {"type": "qa_pair", "section": section_name}
                    })

        # 2. Sliding Window (The "Context" layer)
        # Groups 3 sentences together so the AI understands flow
        sentences = re.split(r'(?<=[.!?])\s+', text)
        window_size = 3
        stride = 2 # Overlap
        
        for i in range(0, len(sentences), stride):
            window = " ".join(sentences[i:i+window_size])
            if len(window) > 20:
                chunks.append({
                    "id": self._generate_id("window", window[:20]),
                    "text": f"Context from {section_name}: {window}",
                    "meta": {"type": "context_window", "section": section_name}
                })
                
        return chunks

    # ==========================================================
    # 🚀 MAIN EXECUTION
    # ==========================================================

    def process_file(self):
        if not os.path.exists(TEXT_FILE_PATH):
            print(f"❌ ERROR: Data file not found at {TEXT_FILE_PATH}")
            return

        print(f"📖 Reading raw data...")
        with open(TEXT_FILE_PATH, 'r', encoding='utf-8') as f:
            full_text = f.read()

        # Split by Sections using the ## SECTION headers
        # We use a regex to capture the Title and the Body
        section_pattern = r'(## SECTION \d+: [^\n]+)(.*?)(?=## SECTION \d+:|$)'
        matches = re.findall(section_pattern, full_text, re.DOTALL)
        
        all_chunks = []

        for title, body in matches:
            section_title = self._clean_text(title.replace('#', ''))
            print(f"  👉 Processing: {section_title}")
            
            # DECIDE STRATEGY BASED ON SECTION
            section_upper = section_title.upper()
            
            if "STUDENT DATABASE" in section_upper or "ROLL CALL" in section_upper:
                # Use specialized student parser + Brute Force
                all_chunks.extend(self.parse_student_section(body, section_title))
                # We DON'T add brute force here because student section is too big and repetitive
                # unless strictly necessary. But for 1000% coverage, let's add raw lines too.
                all_chunks.extend(self.parse_generic_brute_force(body, section_title))
                
            elif "ROUTINE" in section_upper or "SCHEDULE" in section_upper:
                # Use specialized routine parser + Brute Force
                all_chunks.extend(self.parse_routine_section(body, section_title))
                all_chunks.extend(self.parse_generic_brute_force(body, section_title))
                
            else:
                # Use Brute Force for everything else (History, Admin, AI Project)
                all_chunks.extend(self.parse_generic_brute_force(body, section_title))

        # VIP HANDLE: SANGAM GAUTAM & AI PROJECT
        # Ensure this specific data is heavily weighted
        if "Sangam Gautam" in full_text:
            print("  ⭐ Adding VIP Indexing for AI Project...")
            vip_text = "Sangam Gautam is the Project Leader, Full AI Creator, and Lead Developer of the Adarsha School AI Assistant (2082). He built the system using Python, ChromaDB, and RAG."
            all_chunks.append({
                "id": "vip_sangam_001",
                "text": vip_text,
                "meta": {"type": "vip", "section": "AI PROJECT"}
            })

        # ==========================================================
        # 💾 STORAGE
        # ==========================================================
        
        print(f"\n📊 Total Chunks Generated: {len(all_chunks)}")
        print("   (If this number is over 2000, we have achieved 1000% coverage)")
        
        batch_size = 100
        total_batches = (len(all_chunks) // batch_size) + 1
        
        print(f"\n🚀 Starting Embedding & Storage Process...")
        
        for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i:i+batch_size]
            if not batch: continue
            
            ids = [x["id"] for x in batch]
            documents = [x["text"] for x in batch]
            metadatas = [x["meta"] for x in batch]
            
            # Embed
            embeddings = self.model.encode(documents).tolist()
            
            # Add to Chroma
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            print(f"   Processed Batch {i//batch_size + 1}/{total_batches}")

        print(f"\n{'='*60}")
        print(f"✅ DONE! DATABASE CONTAINS {self.collection.count()} ENTRIES.")
        print(f"   Path: {self.db_path}")
        print(f"{'='*60}")


if __name__ == "__main__":
    store = NuclearVectorStore()
    store.process_file()