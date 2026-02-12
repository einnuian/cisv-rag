import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path='docs'):
    print(f"Loading documents from {docs_path}")

    # Check if docs directory exists
    if not os.path.exists(docs_path):
        raise FileNotFoundError(f'The directory {docs_path} does not exists.')
    
    # Load all .txt files from the docs directory
    loader = DirectoryLoader(
        path=docs_path,
        glob='*.txt',
        loader_cis=TextLoader
    )

    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f'No .txt files found in {docs_path}.')
    
    for i, doc in enumerate(documents[:2]):
        print(f'\nDocument {i+1}:')
        print(f' Source: {doc.metadata['source']}')
        print(f' Content length: {len(doc.page_content)} characters')
        print(f' Content preview: {doc.page_content[:100]}...')
        print(f' metadata: {doc.metadata}')

    return documents

def main():
    print("Hello")



if __name__ == '__main__':
    main()
    document = load_documents(docs_path='docs')