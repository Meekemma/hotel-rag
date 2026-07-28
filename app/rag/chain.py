from operator import itemgetter

from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.core.config import settings
from app.rag.retriever import get_advanced_retriever

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
#
# MessagesPlaceholder("history") reserves a spot in the message sequence for
# prior conversation turns. At invoke time it gets replaced with an actual
# list of HumanMessage/AIMessage objects (see to_messages below) — the LLM
# then sees a real back-and-forth (system, then each past human/ai pair, then
# the current human question) instead of a single isolated question. This is
# what lets a guest ask "and what about breakfast?" and have "and" resolve
# correctly against whatever was asked before.
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
        MessagesPlaceholder("history"),
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
# HISTORY CONVERTER
# ---------------------------------------------------------------------------
# app.memory.session_store stores turns as plain dicts ({"question": ...,
# "answer": ...}) — it's LangChain-agnostic on purpose, since its only job is
# talking to Redis. This function is the boundary where that raw storage
# format gets turned into the LangChain message objects MessagesPlaceholder
# expects. Each stored turn becomes a HumanMessage (what the guest asked)
# followed by an AIMessage (what the assistant answered), in that order, so
# the LLM sees the same alternating pattern a real conversation transcript
# would have.
def to_messages(turns: list[dict]) -> list[BaseMessage]:
    messages: list[BaseMessage] = []

    for turn in turns:
        messages.append(HumanMessage(content=turn["question"]))
        messages.append(AIMessage(content=turn["answer"]))

    return messages


# ---------------------------------------------------------------------------
# CHAIN FACTORY
# ---------------------------------------------------------------------------
def get_rag_chain():
    """Return a runnable RAG chain ready to call with {"question": "..."}."""

    # Phase 2 advanced retriever: hybrid search → multi-query → rerank.
    # Built fresh each call so it always reflects the current ChromaDB state.
    retriever = get_advanced_retriever()

    # --- LCEL pipe syntax explained ---
    # The | operator chains runnables left to right, like a Unix pipeline.
    # Each stage receives the output of the stage before it.
    #
    # Stage 1 — input router:
    #   The chain now expects {"question": "...", "history": [BaseMessage, ...]}.
    #   itemgetter("question") pulls just the string out for the branches that
    #   need a plain string (the retriever, and the prompt's {question} slot).
    #   itemgetter("history") passes the message list straight through
    #   unchanged — it was already converted from Redis's raw dict format to
    #   BaseMessage objects by to_messages() before invoke() was called
    #   (see routes.py), so no transformation is needed here.
    #
    #   Result after stage 1:
    #   {"context": [Document, ...], "question": "...", "history": [BaseMessage, ...]}
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
            # Pull the question string out of the input dict, run the
            # retriever on it, then format the returned documents into a
            # single context string.
            "context": itemgetter("question") | retriever | _format_docs,
            # Pass the question string through unchanged so the prompt can use it.
            "question": itemgetter("question"),
            # Pass the pre-converted message list through unchanged — this is
            # what fills MessagesPlaceholder("history") in _PROMPT.
            "history": itemgetter("history"),
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
