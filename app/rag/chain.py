from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.core.config import settings
from app.rag.vector_store import get_vector_store, get_retriever

# ---------------------------------------------------------------------------
# PROMPT TEMPLATE
# ---------------------------------------------------------------------------
# A ChatPromptTemplate defines the exact message structure sent to the LLM.
# {context} will be replaced with the retrieved document chunks.
# {question} will be replaced with the user's actual question.
#
# The system message tells the LLM *who it is* and *how to behave*.
# The human message carries the actual payload (context + question).
# Keeping them separate matters: the LLM weights system messages as
# instructions and human messages as the task — mixing them confuses the model.
#
# "Use only the context below" is the core RAG instruction. Without it the
# model will confidently answer from its training data and hallucinate hotel
# details that don't exist in our documents.
_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are a helpful concierge assistant for Grand Lekki Hotel. "
                "Answer the guest's question using ONLY the context provided below. "
                "If the answer is not in the context, say you don't have that information "
                "and offer to connect the guest with the front desk. "
                "Be warm, professional, and concise.\n\n"
                "Context:\n{context}"
            ),
        ),
        ("human", "{question}"),
    ]
)


# ---------------------------------------------------------------------------
# CONTEXT FORMATTER
# ---------------------------------------------------------------------------
# The retriever returns a list of Document objects. The LLM expects a single
# string. This function joins the page_content of each chunk with a blank
# line separator so the model can clearly see where one chunk ends and the
# next begins — important for accuracy when chunks discuss different topics.
def _format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------------------------------------------------------------------
# CHAIN FACTORY
# ---------------------------------------------------------------------------
def get_rag_chain():
    """Return a runnable RAG chain ready to call with {"question": "..."}."""

    # Build the retriever fresh each call so the chain always uses the current
    # state of ChromaDB (important after a new document is ingested at runtime).
    retriever = get_retriever(get_vector_store())

    # --- LCEL pipe syntax explained ---
    # The | operator chains runnables left to right, like a Unix pipeline.
    # Each stage receives the output of the stage before it.
    #
    # Stage 1 — input router:
    #   RunnablePassthrough() keeps the original {"question": ...} dict intact
    #   so the prompt stage can still read it.
    #   The retriever runs in parallel, receives the raw question string, and
    #   its output is assigned to the "context" key.
    #
    #   Result after stage 1: {"context": [Document, ...], "question": "..."}
    #
    # Stage 2 — prompt:
    #   ChatPromptTemplate fills {context} and {question} into the template.
    #   _format_docs converts the Document list to a plain string first.
    #
    #   Result after stage 2: a filled ChatPromptValue (list of messages)
    #
    # Stage 3 — LLM:
    #   ChatOllama sends the messages to Ollama and returns an AIMessage.
    #
    # Stage 4 — output parser:
    #   StrOutputParser extracts the plain text from the AIMessage so the
    #   caller receives a normal Python string, not a LangChain object.

    chain = (
        {
            # Run the retriever on the raw question string, then format the
            # returned documents into a single context string.
            "context": retriever | _format_docs,
            # Pass the question through unchanged so the prompt can use it.
            "question": RunnablePassthrough(),
        }
        | _PROMPT
        | ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            # temperature=0 means no randomness — the model always picks the
            # highest-probability token. Correct for a factual Q&A assistant;
            # wrong for creative tasks. Raise it (0–1) if you ever want variety.
            temperature=0,
        )
        | StrOutputParser()
    )

    return chain
