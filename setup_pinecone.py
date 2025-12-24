"""
Quick script to create Pinecone index for MediBlaze
Run this once to set up the vector database
"""
import os
from dotenv import load_dotenv
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import time

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not PINECONE_API_KEY:
    print("❌ PINECONE_API_KEY not found in .env file")
    print("Please add your Pinecone API key to the .env file")
    exit(1)

print("=" * 70)
print("🔧 MediBlaze Pinecone Index Setup")
print("=" * 70)
print()

try:
    # Initialize Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "mediblaze-bot"
    
    print(f"📋 Checking existing indexes...")
    existing_indexes = [index.name for index in pc.list_indexes()]
    print(f"   Found {len(existing_indexes)} existing index(es): {existing_indexes}")
    print()
    
    if index_name in existing_indexes:
        print(f"✅ Index '{index_name}' already exists!")
        print()
        
        # Check index stats
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        print(f"📊 Index Statistics:")
        print(f"   Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"   Dimension: {stats.get('dimension', 'N/A')}")
        print()
        
        if stats.get('total_vector_count', 0) == 0:
            print("⚠️  Index is empty. You need to upload documents.")
            print("💡 Run: python src/rag_upload.py")
        else:
            print("✅ Index has data and is ready to use!")
            
    else:
        print(f"🔨 Creating new index: {index_name}")
        print(f"   Dimension: 1024 (multilingual-e5-large)")
        print(f"   Metric: cosine")
        print(f"   Cloud: AWS us-east-1 (serverless)")
        print()
        
        pc.create_index(
            name=index_name,
            dimension=1024,  # Dimension for multilingual-e5-large embeddings
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        print("⏳ Waiting for index to be ready (this may take 30-60 seconds)...")
        time.sleep(45)
        
        # Verify index is ready
        if index_name in [idx.name for idx in pc.list_indexes()]:
            print(f"✅ Index '{index_name}' created successfully!")
            print()
            print("📚 Next step: Upload your medical documents")
            print("💡 Run: python src/rag_upload.py")
        else:
            print("⚠️  Index creation may still be in progress. Check Pinecone dashboard.")
    
    print()
    print("=" * 70)
    print("✅ Setup Complete!")
    print("=" * 70)
    print()
    print("🚀 Your MediBlaze knowledge base is ready!")
    print("   To use RAG, make sure to upload documents if not already done.")
    print()
    
except Exception as e:
    print()
    print("=" * 70)
    print("❌ Error occurred!")
    print("=" * 70)
    print(f"Error: {e}")
    print()
    print("💡 Troubleshooting:")
    print("   1. Check your PINECONE_API_KEY in .env file")
    print("   2. Verify your Pinecone account has access")
    print("   3. Check Pinecone dashboard: https://app.pinecone.io/")
    print()
