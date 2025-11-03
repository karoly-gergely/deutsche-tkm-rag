#!/usr/bin/env python3
"""Development check script - prints environment, data count, and Chroma status."""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from config import settings
except ImportError as e:
    print(f"⚠️  Could not import settings: {e}")
    print("   Please install dependencies: make install")
    sys.exit(1)


def print_environment():
    """Print environment information."""
    print("=" * 60)
    print("Environment Information")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Platform: {sys.platform}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Project root: {Path(__file__).parent.parent}")
    print()


def print_data_count():
    """Print data file count."""
    print("=" * 60)
    print("Data Status")
    print("=" * 60)
    
    data_folder = Path(settings.DATA_FOLDER)
    
    if not data_folder.exists():
        print(f"❌ Data folder does not exist: {data_folder}")
        print()
        return
    
    if not data_folder.is_dir():
        print(f"⚠️  Data folder is not a directory: {data_folder}")
        print()
        return
    
    # Count .txt files
    txt_files = list(data_folder.glob("*.txt"))
    total_size = sum(f.stat().st_size for f in txt_files if f.is_file())
    
    print(f"✓ Data folder: {data_folder}")
    print(f"  .txt files: {len(txt_files)}")
    print(f"  Total size: {total_size:,} bytes ({total_size / 1024 / 1024:.2f} MB)")
    
    # Count other file types if present
    all_files = list(data_folder.glob("*"))
    other_files = [f for f in all_files if f.is_file() and f.suffix != ".txt"]
    if other_files:
        print(f"  Other files: {len(other_files)}")
    
    print()


def print_chroma_status():
    """Print ChromaDB directory status."""
    print("=" * 60)
    print("ChromaDB Status")
    print("=" * 60)
    
    chroma_dir = Path(settings.CHROMA_DIR)
    
    if chroma_dir.exists() and chroma_dir.is_dir():
        # Count files in ChromaDB directory
        chroma_files = list(chroma_dir.rglob("*"))
        chroma_files = [f for f in chroma_files if f.is_file()]
        chroma_size = sum(f.stat().st_size for f in chroma_files if f.exists())
        
        print(f"✓ ChromaDB directory exists: {chroma_dir}")
        print(f"  Files: {len(chroma_files)}")
        print(f"  Size: {chroma_size:,} bytes ({chroma_size / 1024 / 1024:.2f} MB)")
        
        # Try to get collection count if ChromaDB is initialized
        try:
            try:
                from langchain_community.vectorstores import Chroma
            except ImportError:
                from langchain.vectorstores import Chroma
            from core.embeddings import get_embeddings
            
            embeddings = get_embeddings()
            vectordb = Chroma(
                persist_directory=str(chroma_dir),
                embedding_function=embeddings
            )
            
            # Try to get collection count
            if hasattr(vectordb, "_collection"):
                collection = vectordb._collection
                count = collection.count() if hasattr(collection, "count") else "N/A"
                print(f"  Vector count: {count}")
        except ImportError:
            print(f"  Note: ChromaDB dependencies not installed")
        except Exception as e:
            print(f"  Note: Could not load ChromaDB ({type(e).__name__}: {e})")
    else:
        print(f"❌ ChromaDB directory does not exist: {chroma_dir}")
        print("   Run 'make ingest' or 'python scripts/ingest.py' to create index")
    
    print()


def print_settings_summary():
    """Print key settings summary."""
    print("=" * 60)
    print("Key Settings")
    print("=" * 60)
    print(f"MODEL_ID: {settings.MODEL_ID}")
    print(f"EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
    print(f"DEVICE: {settings.DEVICE}")
    print(f"DATA_FOLDER: {settings.DATA_FOLDER}")
    print(f"CHROMA_DIR: {settings.CHROMA_DIR}")
    print(f"CHUNK_SIZE: {settings.CHUNK_SIZE}")
    print(f"CHUNK_OVERLAP: {settings.CHUNK_OVERLAP}")
    print(f"TOP_K: {settings.TOP_K}")
    print()


def main():
    """Main function."""
    print("\n🔍 Development Environment Check\n")
    
    print_environment()
    print_settings_summary()
    print_data_count()
    print_chroma_status()
    
    print("=" * 60)
    print("Check complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()

