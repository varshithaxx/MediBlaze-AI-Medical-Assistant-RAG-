import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# print(PINECONE_API_KEY)

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_pdf_file(data_dir="."):
    # Resolve path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, data_dir)
    data_path = os.path.normpath(data_path)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data directory not found: {data_path}")
    
    print(f"📂 Loading PDFs from: {data_path}")
    loader = DirectoryLoader(data_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} documents")
    return documents

extracted_data = load_pdf_file("../Data")

#CHunking
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20,
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks

text_chunks = text_split(extracted_data)
print(f"📝 Created {len(text_chunks)} text chunks")

from langchain_pinecone import PineconeEmbeddings
# Initialize Pinecone embeddings model
embeddings = PineconeEmbeddings(model="multilingual-e5-large")
print("🔧 Initialized Pinecone embeddings model")

from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore

pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "mediblaze-bot"

# Check if index exists, if not create it
import time
existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"Creating index {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=1024,  # Dimension of the embeddings
        metric="cosine",  # Similarity metric
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Waiting for index to be ready...")
    time.sleep(60)
else:
    print(f"Index {index_name} already exists, proceeding with upload...")

# Upload text chunks to the index in batches
print("Starting batch upload of text chunks...")
batch_size = 100  # Upload 100 chunks at a time
total_chunks = len(text_chunks)
uploaded_count = 0

for i in range(0, total_chunks, batch_size):
    batch = text_chunks[i:i + batch_size]
    current_batch_size = len(batch)
    
    print(f"Uploading batch {i//batch_size + 1}: chunks {i+1}-{i+current_batch_size} of {total_chunks}")
    
    try:
        if i == 0:
            # Create the vector store with the first batch
            docsearch = PineconeVectorStore.from_documents(
                documents=batch,
                index_name=index_name,
                embedding=embeddings,
            )
        else:
            # Add subsequent batches to the existing vector store
            docsearch.add_documents(documents=batch)
        
        uploaded_count += current_batch_size
        print(f"Successfully uploaded batch. Total uploaded: {uploaded_count}/{total_chunks}")
        
        # Small delay between batches to avoid rate limiting
        time.sleep(2)
        
    except Exception as e:
        print(f"Error uploading batch {i//batch_size + 1}: {e}")
        print(f"Retrying batch after 10 seconds...")
        time.sleep(10)
        
        try:
            if i == 0:
                docsearch = PineconeVectorStore.from_documents(
                    documents=batch,
                    index_name=index_name,
                    embedding=embeddings,
                )
            else:
                docsearch.add_documents(documents=batch)
            
            uploaded_count += current_batch_size
            print(f"Retry successful. Total uploaded: {uploaded_count}/{total_chunks}")
        except Exception as retry_error:
            print(f"Retry failed for batch {i//batch_size + 1}: {retry_error}")
            continue

print(f"Upload completed! Successfully uploaded {uploaded_count}/{total_chunks} chunks to Pinecone index '{index_name}'")
