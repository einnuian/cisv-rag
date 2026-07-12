"""Generation backends for the CISV advisor.

Each provider owns its own conversation history because the two APIs represent
history differently: Anthropic stores rich content blocks (thinking + native
citations), while Mistral stores plain role/content strings. Both expose the
same `ask(question, chunks)` method: it streams the answer to stdout and returns
the list of source titles to display.
"""

import anthropic
from mistralai.client import Mistral

ANTHROPIC_MODEL = 'claude-opus-4-8'
MISTRAL_MODEL = 'mistral-small-latest'  # newest Small; pin a snapshot e.g. 'mistral-small-2506'

SYSTEM_PROMPT = """You are an experienced CISV advisor. You answer questions from \
volunteers and staff using ONLY the reference documents provided in each message.

Each document is labelled with a [Source: ...] tag.

Rules:
- Base every answer on the provided documents and cite the source tag inline, e.g. \
"[Source: handbook.pdf (page 3)]", whenever you use a document.
- If the documents don't cover the question, say so plainly ("That isn't covered \
in the documents I have") rather than guessing or using outside knowledge.
- Be practical and concise, like an experienced colleague explaining a procedure."""


def chunk_title(chunk):
    """Human-readable label for a retrieved chunk, including page if present."""
    title = chunk['source']
    if chunk['page']:
        title += f' (page {chunk["page"]})'
    return title


class AnthropicProvider:
    """Claude backend using native document citations."""

    name = 'Claude'

    def __init__(self, model=ANTHROPIC_MODEL):
        self.client = anthropic.Anthropic()
        self.model = model
        self.messages = []

    def ask(self, question, chunks):
        content = []
        for chunk in chunks:
            content.append({
                'type': 'document',
                'source': {'type': 'text', 'media_type': 'text/plain', 'data': chunk['text']},
                'title': chunk_title(chunk),
                'citations': {'enabled': True},
            })
        content.append({'type': 'text', 'text': question})
        self.messages.append({'role': 'user', 'content': content})

        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=16000,
                thinking={'type': 'adaptive'},
                system=SYSTEM_PROMPT,
                messages=self.messages,
            ) as stream:
                for text in stream.text_stream:
                    print(text, end='', flush=True)
                final = stream.get_final_message()
            print()
        except Exception:
            self.messages.pop()  # drop the unanswered turn so history stays valid
            raise

        sources = []
        for block in final.content:
            if block.type == 'text' and block.citations:
                for citation in block.citations:
                    if citation.document_title and citation.document_title not in sources:
                        sources.append(citation.document_title)

        # Keep the full content blocks (including thinking) so follow-up turns replay
        # cleanly, but drop empty text blocks — citations can leave a trailing one, and
        # the API rejects them ("text content blocks must be non-empty") when replayed.
        self.messages.append({
            'role': 'assistant',
            'content': [b for b in final.content if b.type != 'text' or b.text],
        })
        return sources


class MistralProvider:
    """Mistral backend. No native citations, so documents are labelled with
    [Source: ...] tags in the prompt and sources are the retrieved titles."""

    name = 'Mistral'

    def __init__(self, model=MISTRAL_MODEL):
        import os

        self.client = Mistral(api_key=os.environ['MISTRAL_API_KEY'])
        self.model = model
        self.messages = []

    def ask(self, question, chunks):
        blocks = [f'[Source: {chunk_title(c)}]\n{c["text"]}' for c in chunks]
        context = '\n\n'.join(blocks)
        self.messages.append({
            'role': 'user',
            'content': f'Reference documents:\n\n{context}\n\nQuestion: {question}',
        })

        parts = []
        try:
            stream = self.client.chat.stream(
                model=self.model,
                messages=[{'role': 'system', 'content': SYSTEM_PROMPT}] + self.messages,
            )
            for event in stream:
                delta = event.data.choices[0].delta.content
                if delta:
                    print(delta, end='', flush=True)
                    parts.append(delta)
            print()
        except Exception:
            self.messages.pop()  # drop the unanswered turn so history stays valid
            raise

        self.messages.append({'role': 'assistant', 'content': ''.join(parts)})

        # No token-level citations from Mistral; report the documents we retrieved.
        sources = []
        for chunk in chunks:
            title = chunk_title(chunk)
            if title not in sources:
                sources.append(title)
        return sources


def make_provider(name):
    """Return a provider instance for 'anthropic' or 'mistral'."""
    providers = {'anthropic': AnthropicProvider, 'mistral': MistralProvider}
    if name not in providers:
        raise SystemExit(
            f"Unknown LLM_PROVIDER {name!r} — set it to one of: {', '.join(providers)}."
        )
    return providers[name]()
