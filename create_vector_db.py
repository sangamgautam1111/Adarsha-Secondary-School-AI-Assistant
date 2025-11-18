import json
import chromadb
from sentence_transformers import SentenceTransformer
import os
from typing import Dict, List, Any
import hashlib
import uuid

class SchoolDataVectorStore:
    def __init__(
        self,
        db_path=r"C:\Users\USER\Documents\GitHub\Adarsha-Secondary-School-AI-Assistant\vector_db",
        model_path=r"C:\Users\USER\Desktop\models"
    ):
        """Initialize ChromaDB with persistent storage and local model"""
        self.db_path = db_path
        self.model_path = model_path
        self.id_counter = 0  # Counter to ensure unique IDs
        self.used_ids = set()  # Track used IDs
        
        # Create database directory
        os.makedirs(db_path, exist_ok=True)
        
        print(f"Initializing ChromaDB at: {db_path}")
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Load embedding model from local path
        print(f"Loading embedding model from: {model_path}")
        self.embedding_model = SentenceTransformer(model_path)
        print("✓ Model loaded successfully")
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name="adarsha_school_data",
            metadata={
                "description": "Adarsha Secondary School comprehensive information database",
                "embedding_model": "all-MiniLM-L6-v2"
            }
        )
        print(f"✓ Collection initialized with {self.collection.count()} existing documents\n")
    
    def create_semantic_chunks(self, data: Dict, parent_path: str = "") -> List[Dict]:
        """Create semantically meaningful chunks from nested JSON data"""
        chunks = []
        
        if parent_path == "":
            # Root level - process major sections
            for section_key, section_data in data.items():
                chunks.extend(self._process_section(section_key, section_data, section_key))
        
        return chunks
    
    def _process_section(self, section_name: str, section_data: Any, path: str) -> List[Dict]:
        """Process individual sections with context-aware chunking"""
        chunks = []
        
        if isinstance(section_data, dict):
            # Create summary chunk for complex sections
            if self._is_complex_section(section_data):
                summary = self._create_section_summary(section_name, section_data)
                chunks.append({
                    "id": self._generate_id(path),
                    "content": summary,
                    "metadata": {
                        "section": section_name,
                        "type": "summary",
                        "path": path,
                        "has_subsections": True
                    }
                })
            
            # Process nested items
            for key, value in section_data.items():
                sub_path = f"{path}.{key}"
                
                if isinstance(value, dict):
                    if self._should_create_chunk(key, value):
                        chunk_content = self._format_dict_content(key, value, path)
                        chunks.append({
                            "id": self._generate_id(sub_path),
                            "content": chunk_content,
                            "metadata": {
                                "section": section_name,
                                "subsection": key,
                                "type": "detail",
                                "path": sub_path
                            }
                        })
                    chunks.extend(self._process_section(section_name, value, sub_path))
                
                elif isinstance(value, list):
                    chunks.extend(self._process_list(section_name, key, value, sub_path))
                
                else:
                    chunks.append({
                        "id": self._generate_id(sub_path),
                        "content": f"{section_name} - {key}: {value}",
                        "metadata": {
                            "section": section_name,
                            "key": key,
                            "type": "simple",
                            "path": sub_path
                        }
                    })
        
        elif isinstance(section_data, list):
            chunks.extend(self._process_list("root", section_name, section_data, path))
        
        return chunks
    
    def _process_list(self, section: str, key: str, items: List, path: str) -> List[Dict]:
        """Process list items with appropriate grouping"""
        chunks = []
        
        # Special handling for students
        if key == "students":
            for idx, item in enumerate(items):
                item_path = f"{path}[{idx}]"
                if isinstance(item, dict):
                    content = f"Student: {item.get('name', 'Unknown')} - Roll: {item.get('roll', 'N/A')} - Class: {section}"
                else:
                    content = f"{section} - Student[{idx}]: {item}"
                
                chunks.append({
                    "id": self._generate_id(item_path),
                    "content": content,
                    "metadata": {
                        "section": section,
                        "list_type": "students",
                        "index": idx,
                        "type": "student",
                        "path": item_path
                    }
                })
        
        # Special handling for staff/teachers
        elif key == "staff_list" or key == "teachers":
            for idx, item in enumerate(items):
                item_path = f"{path}[{idx}]"
                if isinstance(item, dict):
                    name = item.get('name', 'Unknown')
                    position = item.get('position', 'Staff')
                    content = f"Staff: {name} - Position: {position} - Department: {section}"
                else:
                    content = f"{section} - {key}[{idx}]: {item}"
                
                chunks.append({
                    "id": self._generate_id(item_path),
                    "content": content,
                    "metadata": {
                        "section": section,
                        "list_type": key,
                        "index": idx,
                        "type": "staff",
                        "path": item_path
                    }
                })
        
        # General list processing
        else:
            for idx, item in enumerate(items):
                item_path = f"{path}[{idx}]"
                if isinstance(item, dict):
                    content = f"{section} - {key}: {json.dumps(item, ensure_ascii=False)}"
                else:
                    content = f"{section} - {key}[{idx}]: {item}"
                
                chunks.append({
                    "id": self._generate_id(item_path),
                    "content": content,
                    "metadata": {
                        "section": section,
                        "list_type": key,
                        "index": idx,
                        "type": "list_item",
                        "path": item_path
                    }
                })
        
        return chunks
    
    def _is_complex_section(self, data: Dict) -> bool:
        """Determine if section is complex enough to need a summary"""
        if not isinstance(data, dict):
            return False
        return len(data) > 5 or any(isinstance(v, (dict, list)) for v in data.values())
    
    def _should_create_chunk(self, key: str, value: Dict) -> bool:
        """Determine if a dict should be chunked as a unit"""
        important_keys = [
            "principal", "vice_principal", "establishment", "location",
            "contact", "program", "curriculum", "facilities", "coordinator"
        ]
        return key in important_keys or len(value) > 3
    
    def _create_section_summary(self, section_name: str, data: Dict) -> str:
        """Create a summary for complex sections"""
        summary_parts = [f"Section: {section_name}"]
        
        for key, value in list(data.items())[:5]:
            if isinstance(value, (str, int, float)):
                summary_parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                summary_parts.append(f"{key}: {len(value)} items")
            elif isinstance(value, list):
                summary_parts.append(f"{key}: {len(value)} entries")
        
        return " | ".join(summary_parts)
    
    def _format_dict_content(self, key: str, data: Dict, parent: str) -> str:
        """Format dictionary content for better readability"""
        parts = [f"{parent} - {key}:"]
        for k, v in data.items():
            if not isinstance(v, (dict, list)):
                parts.append(f"  {k}: {v}")
        return "\n".join(parts)
    
    def _generate_id(self, path: str) -> str:
        """Generate unique ID from path with counter to ensure uniqueness"""
        # Create base hash from path
        base_id = hashlib.md5(path.encode()).hexdigest()[:12]
        
        # Add counter to ensure uniqueness
        unique_id = f"{base_id}_{self.id_counter:04d}"
        self.id_counter += 1
        
        # Double-check uniqueness
        while unique_id in self.used_ids:
            unique_id = f"{base_id}_{self.id_counter:04d}"
            self.id_counter += 1
        
        self.used_ids.add(unique_id)
        return unique_id
    
    def load_and_store(
        self,
        json_file_path=r"C:\Users\USER\Documents\GitHub\Adarsha-Secondary-School-AI-Assistant\json_data\school_data.json",
        clear_existing: bool = False
    ):
        """Load JSON file and store in ChromaDB"""
        print(f"\n{'='*70}")
        print(f"LOADING DATA FROM JSON FILE")
        print(f"{'='*70}")
        print(f"JSON Path: {json_file_path}\n")
        
        # Validate file path
        if not os.path.exists(json_file_path):
            raise FileNotFoundError(f"JSON file not found: {json_file_path}")
        
        if not os.path.isfile(json_file_path):
            raise ValueError(f"Path is not a file: {json_file_path}")
        
        # Clear existing data if requested
        if clear_existing and self.collection.count() > 0:
            print("⚠ Clearing existing collection...")
            self.client.delete_collection("adarsha_school_data")
            self.collection = self.client.create_collection(
                name="adarsha_school_data",
                metadata={"description": "Adarsha Secondary School data"}
            )
            print("✓ Collection cleared\n")
        
        # Reset ID tracking for fresh load
        self.id_counter = 0
        self.used_ids.clear()
        
        # Load JSON data
        print("Loading JSON data...")
        with open(json_file_path, 'r', encoding='utf-8') as f:
            school_data = json.load(f)
        print("✓ JSON data loaded\n")
        
        print("Creating semantic chunks...")
        chunks = self.create_semantic_chunks(school_data)
        print(f"✓ Created {len(chunks)} chunks\n")
        
        # Verify no duplicate IDs
        chunk_ids = [chunk["id"] for chunk in chunks]
        unique_ids = set(chunk_ids)
        if len(chunk_ids) != len(unique_ids):
            duplicates = [id for id in chunk_ids if chunk_ids.count(id) > 1]
            print(f"⚠ WARNING: Found {len(chunk_ids) - len(unique_ids)} duplicate IDs")
            print(f"Duplicate IDs: {set(duplicates)}")
            raise ValueError("Duplicate IDs found in chunks")
        
        print("Storing chunks in ChromaDB...")
        print("-" * 70)
        
        # Store in batches
        batch_size = 100
        total_batches = (len(chunks) - 1) // batch_size + 1
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            ids = [chunk["id"] for chunk in batch]
            documents = [chunk["content"] for chunk in batch]
            metadatas = [chunk["metadata"] for chunk in batch]
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(
                documents,
                show_progress_bar=False
            ).tolist()
            
            # Store in ChromaDB
            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            
            print(f"  ✓ Batch {batch_num}/{total_batches} stored ({len(batch)} documents)")
        
        print("-" * 70)
        print(f"\n{'='*70}")
        print(f"STORAGE COMPLETE")
        print(f"{'='*70}")
        print(f"✓ Total chunks stored: {len(chunks)}")
        print(f"✓ Total documents in collection: {self.collection.count()}")
        print(f"✓ Database location: {self.db_path}")
        print(f"{'='*70}\n")
    
    def get_statistics(self):
        """Get database statistics"""
        total = self.collection.count()
        print(f"\n{'='*70}")
        print(f"DATABASE STATISTICS")
        print(f"{'='*70}")
        print(f"Total documents: {total}")
        print(f"Collection name: {self.collection.name}")
        print(f"Database path: {self.db_path}")
        print(f"Model path: {self.model_path}")
        print(f"{'='*70}\n")


def main():
    """Main execution function - Only stores data"""
    print("\n" + "="*70)
    print(" ADARSHA SECONDARY SCHOOL - VECTOR DATABASE CREATOR")
    print("="*70 + "\n")
    
    # Define paths
    DB_PATH = r"C:\Users\USER\Documents\GitHub\Adarsha-Secondary-School-AI-Assistant\vector_db"
    JSON_PATH = r"C:\Users\USER\Documents\GitHub\Adarsha-Secondary-School-AI-Assistant\json_data\school_data.json"
    MODEL_PATH = r"C:\Users\USER\Desktop\models"
    
    print("Configuration:")
    print(f"  Database Path: {DB_PATH}")
    print(f"  JSON File Path: {JSON_PATH}")
    print(f"  Model Path: {MODEL_PATH}")
    print()
    
    # Verify paths exist
    if not os.path.exists(JSON_PATH):
        print(f"\n❌ ERROR: JSON file not found at: {JSON_PATH}")
        print("Please verify the file exists and the path is correct.")
        return
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n❌ ERROR: Model path not found at: {MODEL_PATH}")
        print("Please verify the model is downloaded and the path is correct.")
        return
    
    try:
        # Initialize vector store
        print("Initializing Vector Store...")
        store = SchoolDataVectorStore(
            db_path=DB_PATH,
            model_path=MODEL_PATH
        )
        
        # Load and store data (set clear_existing=True to replace all data)
        store.load_and_store(
            json_file_path=JSON_PATH,
            clear_existing=True  # Change to False to append instead of replace
        )
        
        # Get final statistics
        store.get_statistics()
        
        print("✓ Vector database creation completed successfully!")
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        print("Please check that all paths are correct.")
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()