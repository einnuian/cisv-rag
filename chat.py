"""CLI for the CISV advisor: retrieve context and stream a cited answer."""

import os

import chromadb
from openai import OpenAI

from rag.config import CHROMA_PATH, COLLECTION_NAME, LLM_PROVIDER, PROVIDER_KEYS
from rag.providers import make_provider
from rag.retrieval import retrieve


def main():
    required = ['OPENAI_API_KEY', PROVIDER_KEYS.get(LLM_PROVIDER, '')]
    missing = [key for key in required if key and not os.getenv(key)]
    if missing:
        raise SystemExit(f'Missing {", ".join(missing)} — copy .env.example to .env and fill in your keys.')

    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        collection = chroma.get_collection(COLLECTION_NAME)
    except Exception:
        raise SystemExit('No document index found — run `python -m rag.ingestion` first.')

    provider = make_provider(LLM_PROVIDER)
    openai_client = OpenAI()

    print(f'CISV advisor Q&A ({provider.name}) — ask a question, or type "quit" to exit.')
    while True:
        try:
            question = input('\nQ: ').strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question.lower() in ('quit', 'exit'):
            break

        print()
        chunks = retrieve(question, openai_client, collection)
        try:
            sources = provider.ask(question, chunks)
        except Exception as e:
            print(f'\n{provider.name} API error: {e}')
            continue
        if sources:
            print('\nSources:')
            for source in sources:
                print(f'  - {source}')


if __name__ == '__main__':
    main()
