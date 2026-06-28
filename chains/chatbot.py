from config.settings import (
    BM25_K,
    DATA_PATH,
    ENABLE_FALSE_PREMISE_RETRY,
    ENABLE_MULTI_QUERY_RETRIEVAL,
    ENABLE_NEIGHBOR_EXPANSION,
    ENABLE_QUESTION_REWRITE,
    HYBRID_FINAL_K,
    MAX_CANDIDATES_BEFORE_RERANK,
    MAX_CONTEXT_CHARS,
    MAX_RETRIEVAL_QUERIES,
    MIN_QUALITY_SCORE,
    NEIGHBOR_WINDOW,
    NO_ANSWER_TEXT,
    RERANK_TOP_N,
    SEMANTIC_K,
    SOURCE_TOP_N,
)

# Optional balancer settings para hindi mag-error kung hindi pa updated ang settings.py.
try:
    from config.settings import (
        BALANCED_RERANK_TOP_N,
        CANDIDATE_MAX_PER_SOURCE,
        ENABLE_CANDIDATE_BALANCING,
    )
except ImportError:
    ENABLE_CANDIDATE_BALANCING = True
    CANDIDATE_MAX_PER_SOURCE = 2
    BALANCED_RERANK_TOP_N = 8
from embeddings.embedding_model import get_embedding_model
from vectorstore.chroma_store import load_chroma_vectorstore

from utils.bm25_cache import load_or_create_bm25
from utils.chunk_cache import load_or_create_chunks

from retrieval.context_filter import (
    expand_neighbor_chunks,
    filter_low_quality_docs,
    has_document_evidence,
    has_searchable_question_terms,
    limit_context_docs,
    needs_strict_assumption_check,
)
from retrieval.hybrid_retriever import hybrid_search
from retrieval.candidate_balancer import (
    balance_candidates,
    balance_reranked_documents,
)
from retrieval.retrieval_query_builder import (
    build_retrieval_queries,
    combine_retrieval_queries,
)
from retrieval.reranker import load_reranker, rerank_documents

from llm.ollama_llm import load_llm
from chains.rag_chain import (
    clean_generated_answer,
    generate_answer,
    get_sources,
    stream_answer,
)

from memory.question_rewriter import rewrite_question
from suggestions.suggestion_generator import generate_suggestions


def build_response(answer, sources=None, documents=None, suggestions=None):
    # Standard response format para sa UI.
    return {
        "answer": answer,
        "sources": sources or [],
        "documents": documents or [],
        "suggestions": suggestions or [],
    }


def is_no_answer(answer):
    # Check kung fallback/no-answer ang sagot.
    if not answer:
        return False

    return NO_ANSWER_TEXT.lower() in str(answer).lower()


def safe_generate_suggestions(question, answer, llm):
    # Gumawa ng suggestions pero huwag sirain main flow kapag may error.
    try:
        return generate_suggestions(
            question=question,
            answer=answer,
            llm=llm,
        )
    except Exception:
        return []


def safe_rewrite_question(question, chat_history, llm):
    # Rewrite follow-up question kapag may chat history.
    if not ENABLE_QUESTION_REWRITE or not chat_history:
        return question

    try:
        rewritten_question = rewrite_question(
            question=question,
            chat_history=chat_history,
            llm=llm,
        )

        if rewritten_question and rewritten_question.strip():
            return rewritten_question.strip()

    except Exception:
        pass

    return question


def load_chatbot_components():
    # Load lahat ng reusable components ng chatbot.
    embedding_model = get_embedding_model()

    vectorstore = load_chroma_vectorstore(
        embedding_model=embedding_model,
    )

    chunks = load_or_create_chunks(DATA_PATH)

    bm25_retriever = load_or_create_bm25(
        chunks=chunks,
        k=BM25_K,
    )

    return {
        "vectorstore": vectorstore,
        "bm25_retriever": bm25_retriever,
        "reranker": load_reranker(),
        "llm": load_llm(),
        "chunks": chunks,
    }


def retrieve_documents(
    retrieval_query,
    vectorstore,
    bm25_retriever,
    reranker,
    all_chunks=None,
    answer_query=None,
    debug=False,
):
    # Hybrid retrieval per query -> merge/deduplicate -> rerank once -> filter -> optional expand -> limit context.
    # retrieval_query can be a string or a list of strings.
    if isinstance(retrieval_query, str):
        retrieval_queries = [retrieval_query]
    else:
        retrieval_queries = list(retrieval_query or [])

    retrieval_queries = [str(query).strip() for query in retrieval_queries if str(query).strip()]

    if not retrieval_queries:
        return []

    hybrid_doc_groups = []

    for query in retrieval_queries:
        # Run semantic + BM25 hybrid search for each retrieval query.
        hybrid_docs = hybrid_search(
            query=query,
            vectorstore=vectorstore,
            bm25_retriever=bm25_retriever,
            semantic_k=SEMANTIC_K,
            bm25_k=BM25_K,
            final_k=HYBRID_FINAL_K,
            use_rrf=True,
        )

        hybrid_doc_groups.append(hybrid_docs)

        if debug:
            print(f"Hybrid candidates for query [{query}]: {len(hybrid_docs)}")

    # Balance candidates bago rerank para hindi ma-dominate ng isang source/file.
    candidate_docs = balance_candidates(
        document_groups=hybrid_doc_groups,
        max_docs=MAX_CANDIDATES_BEFORE_RERANK,
        max_per_source=CANDIDATE_MAX_PER_SOURCE,
        enabled=ENABLE_CANDIDATE_BALANCING,
    )

    if debug:
        print(f"Balanced candidates before rerank: {len(candidate_docs)}")

    # Rerank once only para hindi sobrang bumagal kahit multiple retrieval queries.
    # Kapag balancer is enabled, rerank a slightly larger pool then balance down to final top N.
    rerank_query = str(answer_query or retrieval_queries[0]).strip()
    rerank_top_n = RERANK_TOP_N

    if ENABLE_CANDIDATE_BALANCING:
        rerank_top_n = max(RERANK_TOP_N, BALANCED_RERANK_TOP_N)

    reranked_docs = rerank_documents(
        query=rerank_query,
        documents=candidate_docs,
        reranker=reranker,
        top_n=rerank_top_n,
        show_scores=debug,
    )

    balanced_reranked_docs = balance_reranked_documents(
        docs=reranked_docs,
        max_docs=RERANK_TOP_N,
        max_per_source=CANDIDATE_MAX_PER_SOURCE,
        enabled=ENABLE_CANDIDATE_BALANCING,
    )

    clean_docs = filter_low_quality_docs(
        balanced_reranked_docs,
        min_score=MIN_QUALITY_SCORE,
    )

    if debug:
        print(f"Clean reranked docs: {len(clean_docs)}")

    if ENABLE_NEIGHBOR_EXPANSION and all_chunks:
        clean_docs = expand_neighbor_chunks(
            selected_docs=clean_docs,
            all_chunks=all_chunks,
            window=NEIGHBOR_WINDOW,
        )

    return limit_context_docs(
        clean_docs,
        max_chars=MAX_CONTEXT_CHARS,
    )


def maybe_retry_false_premise(question, answer, final_docs, llm, chat_history="", debug=False):
    # Second chance kapag strict question pero nag-fallback kahit may docs.
    if not ENABLE_FALSE_PREMISE_RETRY:
        return answer

    if not final_docs or not is_no_answer(answer):
        return answer

    if not needs_strict_assumption_check(question):
        return answer

    retry_answer = generate_answer(
        question=question,
        docs=final_docs,
        llm=llm,
        chat_history=chat_history,
        debug=debug,
        strict_assumption_check=True,
        correction_retry=True,
    )

    if retry_answer and not is_no_answer(retry_answer):
        return retry_answer

    return answer


def ask_rag(
    question,
    vectorstore,
    bm25_retriever,
    reranker,
    llm,
    chat_history="",
    debug=False,
    all_chunks=None,
):
    # Non-streaming RAG answer.
    question = str(question or "").strip()

    if not question:
        return build_response(answer="No question entered.")

    # Fast guard bago retrieval para hindi gumastos kapag walang searchable term.
    if not chat_history and not has_searchable_question_terms(question):
        return build_response(answer=NO_ANSWER_TEXT)

    rewritten_question = safe_rewrite_question(
        question=question,
        chat_history=chat_history,
        llm=llm,
    )

    retrieval_queries = build_retrieval_queries(
        question=question,
        rewritten_question=rewritten_question,
        enabled=ENABLE_MULTI_QUERY_RETRIEVAL,
        max_queries=MAX_RETRIEVAL_QUERIES,
    )

    retrieval_query_text = combine_retrieval_queries(retrieval_queries)

    final_docs = retrieve_documents(
        retrieval_query=retrieval_queries,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        all_chunks=all_chunks,
        answer_query=question,
        debug=debug,
    )

    if not final_docs:
        return build_response(answer=NO_ANSWER_TEXT)

    # Evidence guard pagkatapos retrieval pero bago LLM.
    # Dito binablock ang partial match gaya ng sahod + araw ng kalayaan.
    if not has_document_evidence(
        question=question,
        retrieval_query=retrieval_query_text,
        docs=final_docs,
        debug=debug,
    ):
        return build_response(answer=NO_ANSWER_TEXT)

    strict_check = needs_strict_assumption_check(question)

    answer = generate_answer(
        question=question,
        docs=final_docs,
        llm=llm,
        chat_history=chat_history,
        debug=debug,
        strict_assumption_check=strict_check,
    )

    answer = maybe_retry_false_premise(
        question=question,
        answer=answer,
        final_docs=final_docs,
        llm=llm,
        chat_history=chat_history,
        debug=debug,
    )

    sources = get_sources(final_docs[:SOURCE_TOP_N])

    if is_no_answer(answer):
        sources = []
        suggestions = []
    else:
        suggestions = safe_generate_suggestions(
            question=question,
            answer=answer,
            llm=llm,
        )

    return build_response(
        answer=answer,
        sources=sources,
        documents=final_docs,
        suggestions=suggestions,
    )


def ask_rag_stream(
    question,
    vectorstore,
    bm25_retriever,
    reranker,
    llm,
    chat_history="",
    debug=False,
    all_chunks=None,
):
    # Streaming RAG answer.
    question = str(question or "").strip()

    if not question:
        yield {"type": "done", **build_response(answer="No question entered.")}
        return

    # Fast guard bago retrieval para hindi gumastos kapag walang searchable term.
    if not chat_history and not has_searchable_question_terms(question):
        yield {"type": "done", **build_response(answer=NO_ANSWER_TEXT)}
        return

    rewritten_question = safe_rewrite_question(
        question=question,
        chat_history=chat_history,
        llm=llm,
    )

    retrieval_queries = build_retrieval_queries(
        question=question,
        rewritten_question=rewritten_question,
        enabled=ENABLE_MULTI_QUERY_RETRIEVAL,
        max_queries=MAX_RETRIEVAL_QUERIES,
    )

    retrieval_query_text = combine_retrieval_queries(retrieval_queries)

    final_docs = retrieve_documents(
        retrieval_query=retrieval_queries,
        vectorstore=vectorstore,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
        all_chunks=all_chunks,
        answer_query=question,
        debug=debug,
    )

    if not final_docs:
        yield {"type": "done", **build_response(answer=NO_ANSWER_TEXT)}
        return

    # Evidence guard pagkatapos retrieval pero bago LLM.
    # Kapag kulang ang support sa final docs, fallback agad at walang source.
    if not has_document_evidence(
        question=question,
        retrieval_query=retrieval_query_text,
        docs=final_docs,
        debug=debug,
    ):
        yield {"type": "done", **build_response(answer=NO_ANSWER_TEXT)}
        return

    strict_check = needs_strict_assumption_check(question)
    full_answer = ""

    for chunk in stream_answer(
        question=question,
        docs=final_docs,
        llm=llm,
        chat_history=chat_history,
        debug=debug,
        strict_assumption_check=strict_check,
    ):
        full_answer += chunk
        yield {"type": "chunk", "content": chunk}

    final_answer = clean_generated_answer(
        answer=full_answer,
        question=question,
    )

    sources = get_sources(final_docs[:SOURCE_TOP_N])

    if is_no_answer(final_answer):
        sources = []
        suggestions = []
    else:
        suggestions = safe_generate_suggestions(
            question=question,
            answer=final_answer,
            llm=llm,
        )

    yield {
        "type": "done",
        **build_response(
            answer=final_answer,
            sources=sources,
            documents=final_docs,
            suggestions=suggestions,
        ),
    }
