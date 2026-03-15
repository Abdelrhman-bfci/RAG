import os
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None # Will handle failure in get_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from app.config import Config
from app.vectorstore.factory import VectorStoreFactory
from sentence_transformers import CrossEncoder
import time
import traceback
import json

# Global shared reranker singleton
_shared_reranker = None


def get_rag_chain(deep_thinking: bool = False, is_continuation: bool = False, last_answer: str = "", conversation_history: list = None):
    """
    Creates and returns the RAG chain for Question Answering.
    """
    # 0. Defensive Initialization
    llm = None
    prompt = None
    
    print(f"DEBUG: Initializing RAG chain. Provider: {Config.LLM_PROVIDER}")

    # 1. Initialize Vector Store
    store = VectorStoreFactory.get_instance()
    vectorstore = store.get_vectorstore()

    # 1.5 Initialize Session Memory if History Exists
    history_retriever = None
    if conversation_history and len(conversation_history) > 0:
        try:
            from app.services.session_memory import LanceDBSessionMemory
            # store.embeddings should exist if FAISSStore or ChromaStore initializes them
            session_memory = LanceDBSessionMemory(
                store.embeddings, 
                k=Config.LLM_HISTORY_K, 
                score_threshold=Config.LLM_HISTORY_SCORE_THRESHOLD
            )
            
            # Format each pair of user/assistant messages as distinct documents
            history_strings = []
            for msg in conversation_history:
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role", "unknown").capitalize()
                content = msg.get("content", "")
                history_strings.append(f"{role}: {content}")
                
            session_memory.add_history(history_strings)
            history_retriever = session_memory.get_retriever()
            print("DEBUG: Loaded LanceDB Session Memory Retriever")
        except Exception as e:
            print(f"DEBUG Error initializing session memory: {e}")

    # 2. Select Prompt (Enhanced for accuracy and strict context adherence)
    # Master Chat Template
    chat_prompt = ChatPromptTemplate.from_template("""
You are a professional Document Assistant acting as a closed-domain reasoning engine.
        
CORE DIRECTIVE:
You must answer the user's question using ONLY the information provided in the "Context" below. You are a highly intelligent and helpful AI assistant for ASU Engineering Faculty. 
Your goal is to provide accurate, concise, and professional answers based EXCLUSIVELY on the provided context.

**CRITICAL ACCURACY RULES:**
1. **STRICT CONTEXT ADHERENCE**: Only use information explicitly stated in the Context. Do not infer, assume, or add information not present.
2. **VERIFICATION**: Before making any claim, verify it exists verbatim or can be directly inferred from the Context.
3. **UNCERTAINTY HANDLING**: If information is partial or unclear in the Context, explicitly state the limitations.
4. **NO HALLUCINATION**: If the answer is not in the context, you MUST state: "I cannot answer this based on the provided documents."

**Response Structure:**
1. **CONTENT:** The actual answer to the user's question, with inline citations.
2. **METADATA:** Complete list of all sources used with page numbers and URLs.

**Citation Requirements:**
- For EVERY factual claim, include an inline citation immediately after: `[Source Name (Page X)](URL)` or `[Source Name](URL)`
- Use Markdown format for citations
- If multiple sources support a claim, cite all of them
- Page numbers are mandatory when available

**Response Format:**
CONTENT: 
[Your detailed answer here with inline citations like this: According to the document [Document.pdf (Page 5)](URL), the requirements are...]

METADATA:
- Source: [Filename/URL], Page: [Number], URL: [URL]
- Source: [Filename/URL], URL: [URL]

**Quality Checks Before Responding:**
1. Can I answer this question using ONLY the Context provided? 
   - If Context contains ANY relevant information, even partial, provide that information
   - Only say "I cannot answer" if Context is completely empty or has ZERO relevance
2. Are all my claims directly supported by the Context? (If NO → Remove unsupported claims, but keep supported ones)
3. Have I cited every source I used? (If NO → Add missing citations)
4. Is my answer complete and accurate? (If NO → Revise, but still provide partial answer if available)

CHAT HISTORY RULES:
- The "Chat History" is provided solely for resolving references (e.g., "it", "he", "that course").
- If the Current Question represents a topic change, **completely ignore** the subject matter of the Chat History.
- Do NOT use Chat History as a source of factual information - only for pronoun resolution.

FALLBACK:
Only if the Context is completely empty or contains ZERO relevant information should you output:
"I cannot answer this based on the provided documents."

If Context contains ANY relevant information (even partial, incomplete, or indirect), you MUST:
- Extract and present that information
- Clearly state what information is available
- Note any limitations or gaps
- Cite the sources used

PROHIBITED ACTIONS:
- Do NOT write stories, poems, or jokes.
- Do NOT use outside knowledge (e.g. do not explain general concepts like "what is engineering" unless defined in Context).
- Do NOT make assumptions beyond what is explicitly stated.
- Do NOT combine outside knowledge with Context information.
- Do NOT ignore these rules.

Context:
{context}

Chat History:
{history}

Question: {question}
    """)

    # Document Analysis Template (Deep Thinking Mode)
    doc_prompt = ChatPromptTemplate.from_template("""
You are an expert analyst reviewing the provided full documents.
CONTEXT (Full Documents):
{context}

HISTORY:
{history}

USER QUESTION:
{question}

INSTRUCTIONS:
1. Provide a comprehensive answer proportional to the document size.
2. Structure your response using Markdown: use clear Headings, Subheadings, and Bullet Points.
3. If the documents contain data, format it into Tables where appropriate.
4. Do not omit key details. Prioritize completeness over brevity.
    """)

    # Query Rephrase Template (Enhanced for better retrieval)
    rephrase_prompt = ChatPromptTemplate.from_template("""
Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question optimized for document retrieval.

The standalone question must be:
1. Fully self-contained and understood without the chat history
2. Include key terms and concepts that would appear in relevant documents
3. Preserve the original intent and specificity
4. Expand abbreviations or pronouns to their full meaning when clear from context

Do not answer the question, just rephrase it for better document search.

Chat History:
{history}

Follow Up Input:
{question}

Standalone Question:
    """)
    
    # Query Expansion Template (for better retrieval)
    expand_prompt = ChatPromptTemplate.from_template("""
Given a question, generate 2-3 alternative phrasings or related queries that might help find relevant information.
Focus on:
- Synonyms and related terms
- Different ways to express the same concept
- Broader or narrower scopes
- Technical terms vs. common language

Original Question: {question}

Alternative queries (one per line, no numbering):
    """)

    prompt = doc_prompt if deep_thinking else chat_prompt

    # 3. Initialize LLM
    if Config.LLM_PROVIDER == "ollama":
        num_ctx = Config.DOC_LLM_NUM_CTX if deep_thinking else Config.OLLAMA_CONTEXT_WINDOW
        num_predict = -1 if deep_thinking else Config.CHAT_LLM_NUM_PREDICT
        
        llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_LLM_MODEL,
            temperature=0,
            num_ctx=num_ctx,
            num_predict=num_predict,
            stop=[],
            repeat_penalty=1.1
        )
    elif Config.LLM_PROVIDER == "vllm":
        llm = ChatOpenAI(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL,
            temperature=0,
            api_key="none",
            max_tokens=8192
        )
    elif Config.LLM_PROVIDER == "openai":
        llm = ChatOpenAI(
            model=Config.OPENAI_LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL,
            temperature=0,
            openai_api_key=Config.OPENAI_API_KEY,
            max_tokens=8192
        )
    elif Config.LLM_PROVIDER == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed. Please run 'pip install langchain-google-genai'")
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0,
            max_output_tokens=8192
        )
    else:
        # Default fallback to Ollama
        print(f"DEBUG: Falling back to Ollama default")
        llm = ChatOllama(
            model=Config.OLLAMA_LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            num_predict=-1,
            stop=[],
            repeat_penalty=1.1
        )
    
    if llm is None:
        print("DEBUG: CRITICAL ERROR - LLM is still None after initialization block")
    else:
        print(f"DEBUG: LLM successfully initialized as {type(llm).__name__}")

    # 3. Initialize Shared Re-Ranker if enabled
    global _shared_reranker
    if Config.USE_RERANKER and _shared_reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            print(f"DEBUG: Loading Shared Re-Ranker '{Config.RERANKER_MODEL}'...")
            _shared_reranker = CrossEncoder(Config.RERANKER_MODEL, max_length=512)
        except Exception as e:
            print(f"DEBUG: Failed to load reranker: {e}")
            _shared_reranker = None

    # 3. Initialize Retriever
    # Advanced Hybrid Retrieval with enhanced query expansion
    class AdvancedHybridRetriever:
        def __init__(self, vectorstore, llm=None, history_retriever=None, expand_prompt=None):
            self.vectorstore = vectorstore
            self.llm = llm # Kept for signature compatibility
            self.history_retriever = history_retriever
            self.expand_prompt = expand_prompt  # Store expand prompt for query expansion
            
            global _shared_reranker
            if Config.USE_RERANKER and _shared_reranker is None:
                print(f"DEBUG: Loading Shared Re-Ranker '{Config.RERANKER_MODEL}'...")
                # Lazy load to avoid slowing down startup if disabled
                _shared_reranker = CrossEncoder(Config.RERANKER_MODEL, max_length=512)
            self.reranker = _shared_reranker
            
        def invoke(self, query):
            try:
                # Ensure query is a string
                if not isinstance(query, str):
                    query = str(query)
                
                # Enhanced retrieval with query expansion
                all_docs = []
                seen_content_hashes = set()
                
                # 1. Primary search with original query (increased retrieval for better coverage)
                k_search = 150 if Config.USE_RERANKER else 80  # Increased from 100/50
                try:
                    primary_docs = self.vectorstore.max_marginal_relevance_search(
                        query, k=k_search, fetch_k=k_search*3, lambda_mult=0.5  # Increased fetch_k for more diversity
                    )
                    print(f"DEBUG: Retrieved {len(primary_docs)} documents from MMR search (k={k_search})")
                except Exception as e:
                    print(f"DEBUG: Vector store does not support MMR, falling back to similarity search: {e}")
                    primary_docs = self.vectorstore.similarity_search(query, k=k_search)
                    print(f"DEBUG: Retrieved {len(primary_docs)} documents from similarity search (k={k_search})")
                
                # Deduplicate by content hash
                for doc in primary_docs:
                    content_hash = hash(doc.page_content[:200])  # Use first 200 chars as hash
                    if content_hash not in seen_content_hashes:
                        all_docs.append(doc)
                        seen_content_hashes.add(content_hash)
                
                # 2. Query expansion for better coverage (if LLM and expand_prompt available)
                if self.llm and self.expand_prompt and len(query.split()) > 2:  # Only expand substantial queries
                    try:
                        expand_chain = self.expand_prompt | self.llm | StrOutputParser()
                        expanded_queries = expand_chain.invoke({"question": query})
                        # Parse expanded queries (one per line)
                        alt_queries = [q.strip() for q in expanded_queries.split('\n') if q.strip() and not q.strip().startswith('#')]
                        
                        # Search with expanded queries (smaller k to avoid too many duplicates)
                        for alt_query in alt_queries[:2]:  # Limit to 2 alternative queries
                            try:
                                alt_docs = self.vectorstore.similarity_search(alt_query, k=20)
                                for doc in alt_docs:
                                    content_hash = hash(doc.page_content[:200])
                                    if content_hash not in seen_content_hashes:
                                        all_docs.append(doc)
                                        seen_content_hashes.add(content_hash)
                            except:
                                continue
                    except Exception as e:
                        print(f"DEBUG: Query expansion failed: {e}")
                
                # Merge in session history if available
                if self.history_retriever:
                    hist_docs = self.history_retriever.invoke(query)
                    if hist_docs:
                        for d in hist_docs:
                            d.metadata["source"] = "Chat History"
                            content_hash = hash(d.page_content[:200])
                            if content_hash not in seen_content_hashes:
                                all_docs.append(d)
                                seen_content_hashes.add(content_hash)
                        
            except Exception as e:
                print(f"Retrieval error: {e}")
                return []
            
            # Use all_docs instead of initial_docs
            initial_docs = all_docs
            
            print(f"DEBUG: Total documents after retrieval and deduplication: {len(initial_docs)}")
            
            # Defensive check: ensure all docs have dictionary metadata
            for doc in initial_docs:
                if not isinstance(doc.metadata, dict):
                    print(f"DEBUG: Repairing malformed metadata for document: {getattr(doc, 'metadata', 'N/A')}")
                    doc.metadata = {"source": "Unknown", "repair_flag": True}
            
            if not initial_docs:
                print(f"DEBUG: WARNING - No documents retrieved for query: '{query}'")
                return []
            
            # Log sample of retrieved documents for debugging
            if len(initial_docs) > 0:
                sample_sources = [doc.metadata.get("source", "Unknown")[:50] for doc in initial_docs[:5]]
                print(f"DEBUG: Sample sources retrieved: {sample_sources}")
                
            if Config.USE_RERANKER:
                # 2a. Enhanced Re-ranking with multi-factor scoring and better filtering
                import re
                from collections import Counter
                
                # Prepare query-document pairs for reranking
                pairs = []
                doc_contents = []
                for doc in initial_docs:
                    # Use a cleaned version of content for reranking (first 512 chars to match model max_length)
                    content = doc.page_content[:512] if len(doc.page_content) > 512 else doc.page_content
                    # Remove metadata headers that might confuse reranker
                    content_clean = re.sub(r'^---.*?---\s*', '', content, flags=re.MULTILINE)
                    pairs.append([query, content_clean])
                    doc_contents.append(content)
                
                # Get reranker scores
                try:
                    rerank_scores = self.reranker.predict(pairs)
                    rerank_scores = [float(s) for s in rerank_scores]
                except Exception as e:
                    print(f"DEBUG: Reranker prediction failed: {e}, falling back to similarity scores")
                    rerank_scores = [0.5] * len(initial_docs)  # Fallback neutral score
                
                # Calculate additional relevance signals
                query_terms = set(query.lower().split())
                query_terms = {t for t in query_terms if len(t) > 2}  # Filter short terms
                
                ranked_candidates = []
                all_scores = []
                
                for i, doc in enumerate(initial_docs):
                    rerank_score = rerank_scores[i]
                    content = doc_contents[i].lower()
                    
                    # Multi-factor scoring
                    factors = {
                        "rerank": rerank_score,
                        "term_match": 0.0,
                        "source_quality": 0.0,
                        "content_length": 0.0
                    }
                    
                    # 1. Term matching score (keyword overlap)
                    if query_terms:
                        content_words = set(re.findall(r'\b\w+\b', content))
                        matched_terms = query_terms.intersection(content_words)
                        term_match_ratio = len(matched_terms) / len(query_terms) if query_terms else 0
                        factors["term_match"] = min(term_match_ratio * 0.3, 0.3)  # Cap at 0.3
                    
                    # 2. Source quality boost (prefer certain sources)
                    source = doc.metadata.get("source", "").lower() if isinstance(doc.metadata, dict) else ""
                    if source:
                        # Boost official sources (websites, official docs)
                        if source.startswith("http") or "official" in source or "www" in source:
                            factors["source_quality"] = 0.1
                        # Slight penalty for chat history (less reliable)
                        elif "chat history" in source:
                            factors["source_quality"] = -0.05
                    
                    # 3. Content length normalization (prefer substantial content)
                    content_len = len(doc.page_content)
                    if content_len > 100:
                        # Normalize: longer content gets slight boost, but cap it
                        length_score = min((content_len - 100) / 1000, 0.1)  # Max 0.1 boost
                        factors["content_length"] = length_score
                    
                    # Combined score: weighted combination
                    # Rerank score is primary (70%), other factors add/subtract
                    combined_score = (
                        factors["rerank"] * 0.7 +
                        factors["term_match"] +
                        factors["source_quality"] +
                        factors["content_length"]
                    )
                    
                    # Normalize combined score to reasonable range
                    combined_score = max(0.0, min(1.0, combined_score))
                    
                    all_scores.append(combined_score)
                    
                    if isinstance(doc.metadata, dict):
                        doc.metadata["rerank_score"] = rerank_score
                        doc.metadata["combined_score"] = combined_score
                        doc.metadata["term_match_score"] = factors["term_match"]
                        doc.metadata["score"] = combined_score  # Use combined score as primary
                    ranked_candidates.append((doc, combined_score, rerank_score))
                
                # Enhanced adaptive threshold calculation (more lenient)
                if len(all_scores) > 10:
                    sorted_scores = sorted(all_scores, reverse=True)
                    # Use more lenient percentiles
                    p90 = sorted_scores[len(sorted_scores) // 10] if len(sorted_scores) >= 10 else sorted_scores[0]  # Top 10%
                    p75 = sorted_scores[len(sorted_scores) // 4]  # 75th percentile (top 25%)
                    p50 = sorted_scores[len(sorted_scores) // 2]  # Median
                    
                    # More lenient adaptive threshold: use 50th percentile or lower
                    adaptive_threshold = max(
                        p50 * 0.7,  # 70% of median (very lenient)
                        p75 * 0.5,  # 50% of 75th percentile
                        Config.RERANKER_THRESHOLD * 0.5  # Half of config threshold (more lenient)
                    )
                    print(f"DEBUG: Adaptive threshold calculated: {adaptive_threshold:.3f} (p90={p90:.3f}, p75={p75:.3f}, p50={p50:.3f})")
                elif len(all_scores) > 5:
                    sorted_scores = sorted(all_scores, reverse=True)
                    p50 = sorted_scores[len(sorted_scores) // 2]  # Median
                    adaptive_threshold = max(
                        p50 * 0.6,  # 60% of median
                        Config.RERANKER_THRESHOLD * 0.5
                    )
                    print(f"DEBUG: Adaptive threshold (5-10 docs): {adaptive_threshold:.3f}")
                else:
                    # Very lenient for small sets
                    adaptive_threshold = max(Config.RERANKER_THRESHOLD * 0.3, 0.05)  # Very low threshold
                    print(f"DEBUG: Adaptive threshold (small set): {adaptive_threshold:.3f}")
                
                # Sort by combined score (primary), then rerank score (tiebreaker)
                ranked_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
                
                # Filter by adaptive threshold with diversity preservation
                final_docs = []
                seen_sources = Counter()
                max_per_source = 3  # Limit docs per source for diversity
                
                for doc, combined_score, rerank_score in ranked_candidates:
                    source = doc.metadata.get("source", "Unknown") if isinstance(doc.metadata, dict) else "Unknown"
                    
                    # Check threshold
                    if combined_score >= adaptive_threshold:
                        # Diversity check: limit docs per source
                        if seen_sources[source] < max_per_source:
                            if isinstance(doc.metadata, dict):
                                doc.metadata["score"] = combined_score
                            final_docs.append(doc)
                            seen_sources[source] += 1
                        elif len(final_docs) < Config.LLM_K_FINAL:
                            # If we haven't reached target, allow more from same source
                            if isinstance(doc.metadata, dict):
                                doc.metadata["score"] = combined_score
                            final_docs.append(doc)
                            seen_sources[source] += 1
                    
                    # Early stopping if we have enough high-quality docs
                    if len(final_docs) >= Config.LLM_K_FINAL and combined_score < adaptive_threshold * 1.2:
                        break
                
                # Ensure we return reasonable number of docs (more lenient fallback)
                if not final_docs and ranked_candidates:
                    print(f"DEBUG: No docs passed threshold ({adaptive_threshold:.3f}), using fallback")
                    # Fallback: return top docs even if below threshold
                    # But still apply diversity
                    for doc, combined_score, rerank_score in ranked_candidates[:Config.LLM_K_FINAL * 3]:  # Increased from *2
                        source = doc.metadata.get("source", "Unknown") if isinstance(doc.metadata, dict) else "Unknown"
                        if seen_sources[source] < max_per_source or len(final_docs) < Config.LLM_K_FINAL:
                            if isinstance(doc.metadata, dict):
                                doc.metadata["score"] = combined_score
                            final_docs.append(doc)
                            seen_sources[source] += 1
                        if len(final_docs) >= Config.LLM_K_FINAL:
                            break
                
                # Additional safety: if still no docs, return top 5 regardless
                if not final_docs and ranked_candidates:
                    print(f"DEBUG: Still no docs, returning top 5 regardless of score")
                    for idx, (doc, combined_score, rerank_score) in enumerate(ranked_candidates[:5]):
                        if isinstance(doc.metadata, dict):
                            doc.metadata["score"] = combined_score
                        final_docs.append(doc)
                
                print(f"DEBUG: Returning {len(final_docs)} documents after reranking (threshold: {adaptive_threshold:.3f})")
                if final_docs:
                    top_scores = [d.metadata.get("score", 0) for d in final_docs[:3] if isinstance(d.metadata, dict)]
                    print(f"DEBUG: Top 3 scores: {top_scores}")
                
                return final_docs[:Config.LLM_K_FINAL]
            else:
                # 2b. Enhanced Fallback: Multi-factor scoring without reranker
                import re
                from collections import Counter
                
                query_terms = set(query.lower().split())
                query_terms = {t for t in query_terms if len(t) > 2}  # Filter short terms
                keywords = [w.lower() for w in query.split() if len(w) > 3]
                
                scored_docs = []
                seen_chunk_ids = set()
                
                for d in initial_docs:
                    chunk_id = f"{d.metadata.get('source')}_{d.metadata.get('page')}_{d.metadata.get('chunk', d.page_content[:50])}"
                    if chunk_id in seen_chunk_ids:
                        continue
                    seen_chunk_ids.add(chunk_id)
                    
                    if not isinstance(d.metadata, dict):
                        d.metadata = {"source": "Unknown"}
                    
                    content_lower = d.page_content.lower()
                    source = d.metadata.get("source", "").lower()
                    
                    # Multi-factor scoring
                    score = 0.0
                    
                    # 1. Source type boost
                    is_website = source.startswith("http")
                    if is_website:
                        score += 0.3  # Website boost
                    elif "chat history" in source:
                        score += 0.1  # Slight boost for chat history
                    
                    # 2. Keyword matching
                    if keywords:
                        matched_keywords = sum(1 for kw in keywords if kw in content_lower)
                        keyword_score = (matched_keywords / len(keywords)) * 0.4
                        score += keyword_score
                    
                    # 3. Term matching (broader than keywords)
                    if query_terms:
                        content_words = set(re.findall(r'\b\w+\b', content_lower))
                        matched_terms = query_terms.intersection(content_words)
                        term_score = (len(matched_terms) / len(query_terms)) * 0.2
                        score += term_score
                    
                    # 4. Content quality (length)
                    content_len = len(d.page_content)
                    if content_len > 100:
                        length_score = min((content_len - 100) / 2000, 0.1)
                        score += length_score
                    
                    # Store score in metadata
                    d.metadata["score"] = score
                    scored_docs.append((d, score))
                
                # Sort by score
                scored_docs.sort(key=lambda x: x[1], reverse=True)
                
                # Apply diversity: limit docs per source
                final_docs = []
                seen_sources = Counter()
                max_per_source = 3
                
                for doc, score in scored_docs:
                    source = doc.metadata.get("source", "Unknown")
                    if seen_sources[source] < max_per_source or len(final_docs) < Config.LLM_K_FINAL:
                        final_docs.append(doc)
                        seen_sources[source] += 1
                    if len(final_docs) >= Config.LLM_K_FINAL:
                        break
                
                return final_docs[:Config.LLM_K_FINAL]

    retriever = AdvancedHybridRetriever(vectorstore, llm=llm, history_retriever=history_retriever, expand_prompt=expand_prompt)


    # 4. Construct the Chain (Matching Octopiai entirely)
    def format_docs(docs):
        # Master formatter matching chatbot/_format_documents
        import sqlite3
        conn = None
        if os.path.exists(Config.CRAWLER_DB):
            try:
                conn = sqlite3.connect(Config.CRAWLER_DB)
            except:
                pass

        def get_url(source):
            if not conn: return "#"
            try:
                cursor = conn.cursor()
                # Try absolute path matching or basename matching
                basename = os.path.basename(source)
                cursor.execute("SELECT url FROM pages WHERE filename = ? OR filename = ? LIMIT 1", (source, basename))
                row = cursor.fetchone()
                return row[0] if row else "#"
            except:
                return "#"

        context_parts = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "?")
            url = get_url(source) if source != "Chat History" else "#"
            
            # Clean "search_document:" prefix if using NomicWrapper
            content = doc.page_content.replace("search_document: ", "")
            
            context_parts.append(
                f"CONTENT: {content}\n"
                f"METADATA: Source: {os.path.basename(source)}, Page: {page}, URL: {url}"
            )
        
        if conn: conn.close()
        return "\n\n---\n\n".join(context_parts)

    def process_history():
        # Octopiai passes history as a distinct variable block rather than prepending 
        history_text = ""
        if conversation_history and len(conversation_history) > 0:
            from app.services.chat_session import format_history_for_prompt
            history_text = format_history_for_prompt(conversation_history)
        return history_text

    # Rephrase Chain
    rephrase_chain = rephrase_prompt | llm | StrOutputParser()

    def get_search_query(q):
        history = process_history()
        if not history.strip():
            return q
        try:
            return rephrase_chain.invoke({"history": history, "question": q})
        except:
            return q

    def process_question(q):
        if is_continuation and last_answer:
            return f"--- CONTINUATION ---\nContinue from:\n{last_answer}\n\nQuestion: {q}"
        return q

    rag_chain = (
        {
            "rephrased_query": RunnableLambda(get_search_query),
            "original_question": RunnablePassthrough()
        }
        | {
            "context": RunnableLambda(lambda x: retriever.invoke(x.get("rephrased_query", "")) if isinstance(x, dict) else retriever.invoke(str(x))) | format_docs, 
            "history": RunnableLambda(lambda _: process_history()),
            "question": RunnableLambda(lambda x: process_question(x.get("rephrased_query", "")) if isinstance(x, dict) else process_question(str(x)))
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def filter_cited_sources(answer: str, all_sources: list) -> list:
    """
    Enhanced citation extraction that validates sources and extracts structured citations.
    Filters the list of all retrieved sources to return only those that are explicitly 
    cited in the answer text.
    """
    import re
    from urllib.parse import urlparse
    
    # Extract citations in multiple formats:
    # 1. Markdown links: [Source Name](URL) or [Source Name (Page X)](URL)
    # 2. Brackets: [Source Name] or [Source Name, Page X]
    # 3. Parentheses: (Source Name) or (Source Name, Page X)
    
    cited_filenames = set()
    cited_urls = set()
    
    # Pattern 1: Markdown links [text](url)
    md_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', answer)
    for text, url in md_links:
        # Extract source name from text (may include page number)
        source_part = text.split('(')[0].strip()  # Remove "(Page X)" if present
        cited_filenames.add(source_part.lower())
        if url and url != "#":
            cited_urls.add(url.lower())
    
    # Pattern 2: Brackets [Source Name] or [Source Name, Page X]
    bracket_citations = re.findall(r'\[([^\]]+)\]', answer)
    for citation in bracket_citations:
        # Skip if it's part of a markdown link (already processed)
        if '(' in citation and ')' in citation:
            continue
        # Take first part before comma
        part = citation.split(',')[0].strip()
        if part and len(part) > 1:  # Ignore single characters
            cited_filenames.add(part.lower())
    
    # Pattern 3: Parentheses citations (Source Name)
    paren_citations = re.findall(r'\(([^)]+)\)', answer)
    for citation in paren_citations:
        # Skip URLs and page numbers
        if citation.startswith('http') or 'page' in citation.lower():
            continue
        part = citation.split(',')[0].strip()
        if part and len(part) > 1:
            cited_filenames.add(part.lower())
    
    final_sources = []
    for src in all_sources:
        src_lower = src.lower()
        basename = os.path.basename(src).lower()
        
        # Check URL match
        if src_lower in cited_urls:
            final_sources.append(src)
            continue
        
        # Check filename/basename match
        if basename in cited_filenames or src_lower in cited_filenames:
            final_sources.append(src)
            continue
        
        # Check if URL domain matches
        if src.startswith('http'):
            parsed = urlparse(src)
            domain = parsed.netloc.lower()
            for url in cited_urls:
                if domain in url or url in domain:
                    final_sources.append(src)
                    break
    
    # If no sources are cited but we have an answer, return all retrieved sources
    # This ensures we don't lose potentially relevant sources
    if not final_sources:
        return sorted(list(set(all_sources)))
    
    return sorted(list(set(final_sources)))

def validate_answer_against_context(answer: str, retrieved_docs: list) -> dict:
    """
    Validate that the answer is supported by the retrieved documents.
    Returns a validation result with confidence score and warnings.
    """
    if not answer or not retrieved_docs:
        return {
            "is_valid": False,
            "confidence": 0.0,
            "warnings": ["No answer or no retrieved documents"]
        }
    
    # Check for "I don't know" responses
    dont_know_phrases = [
        "i cannot answer",
        "i don't know",
        "not in the provided",
        "cannot be determined",
        "not available in"
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in dont_know_phrases):
        return {
            "is_valid": True,
            "confidence": 1.0,
            "warnings": [],
            "reason": "Explicit uncertainty acknowledged"
        }
    
    # Check if answer contains citations
    import re
    citations = re.findall(r'\[([^\]]+)\]', answer)
    has_citations = len(citations) > 0
    
    # Calculate confidence based on:
    # 1. Presence of citations
    # 2. Answer length (too short might indicate incomplete answer)
    # 3. Whether answer references specific details from context
    
    confidence = 0.5  # Base confidence
    warnings = []
    
    if has_citations:
        confidence += 0.3
    else:
        warnings.append("Answer lacks explicit citations")
    
    # Check if answer seems too generic or too short
    if len(answer) < 50:
        confidence -= 0.2
        warnings.append("Answer is very short, may be incomplete")
    
    # Check if answer contains specific details (numbers, dates, names)
    has_specifics = bool(re.search(r'\d+|[A-Z][a-z]+ [A-Z][a-z]+', answer))
    if has_specifics:
        confidence += 0.2
    
    confidence = max(0.0, min(1.0, confidence))  # Clamp between 0 and 1
    
    return {
        "is_valid": confidence >= 0.5,
        "confidence": round(confidence, 2),
        "warnings": warnings,
        "has_citations": has_citations
    }

def answer_question(question: str, deep_thinking: bool = False, is_continuation: bool = False, last_answer: str = "", conversation_history: list = None):
    """
    Entry point to answer a question with performance metrics, source citations, and answer validation.
    """
    start_total = time.time()
    try:
        chain, retriever = get_rag_chain(
            deep_thinking=deep_thinking, 
            is_continuation=is_continuation, 
            last_answer=last_answer,
            conversation_history=conversation_history
        )
        
        # 1. Retrieve documents manually to get metadata for citations
        docs = retriever.invoke(question)
        
        # Debug: Log retrieval results
        print(f"DEBUG: Retrieved {len(docs)} documents for question: '{question[:100]}'")
        if not docs:
            print(f"DEBUG: WARNING - No documents retrieved! This may cause 'cannot answer' response.")
        else:
            sample_content = [doc.page_content[:100] for doc in docs[:3]]
            print(f"DEBUG: Sample document content previews: {sample_content}")
        
        sources = sorted(list(set([doc.metadata.get("source", "Unknown") for doc in docs])))
        print(f"DEBUG: Unique sources: {len(sources)} - {sources[:5]}")
        
        # Capture used docs with content
        used_docs = []
        for doc in docs:
            used_docs.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0),
                "score": doc.metadata.get("score", 0) if isinstance(doc.metadata, dict) else 0
            })

        # 2. Invoke the chain
        start_llm = time.time()
        answer = chain.invoke(question)
        end_time = time.time()
        
        # 3. Validate answer against retrieved context
        validation = validate_answer_against_context(answer, docs)
        
        # 4. Filter sources based on citation
        relevant_sources = filter_cited_sources(answer, sources)
        
        total_time = end_time - start_total
        llm_time = end_time - start_llm
        
        performance = f"Total {total_time:.1f}s | LLM {llm_time:.1f}s"
        
        return {
            "answer": answer,
            "sources": relevant_sources,
            "performance": performance,
            "used_docs": used_docs,
            "validation": validation  # Add validation results
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"DEBUG ERROR: {traceback.format_exc()}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def stream_answer(question: str, deep_thinking: bool = False, is_continuation: bool = False, last_answer: str = "", conversation_history: list = None):
    """
    Entry point to stream chunks of the LLM response.
    Yields newline-delimited JSON strings for client processing.
    conversation_history: List of dicts with 'role' and 'content' keys
    """
    start_total = time.time()
    try:
        chain, retriever = get_rag_chain(
            deep_thinking=deep_thinking, 
            is_continuation=is_continuation, 
            last_answer=last_answer,
            conversation_history=conversation_history
        )
        
        # 1. Pre-retrieve docs just to get sources early (optional but better for UX)
        start_retrieval = time.time()
        docs = retriever.invoke(question)
        end_retrieval = time.time()
        
        sources = sorted(list(set([doc.metadata.get("source", "Unknown") for doc in docs])))
        
        # Capture used docs with content
        used_docs = []
        for doc in docs:
            used_docs.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0)
            })
        
        # Send ALL sources first so frontend can build the link map
        yield json.dumps({"type": "sources", "sources": sources}) + "\n"
        
        # 2. Stream the chain
        start_llm = time.time()
        accumulated_text = ""
        for chunk in chain.stream(question):
            accumulated_text += chunk
            yield json.dumps({"type": "chunk", "content": chunk}) + "\n"
        
        end_time = time.time()
        
        # 3. Validate answer against retrieved context
        validation = validate_answer_against_context(accumulated_text, docs)
        
        # 4. Filter sources for distinct display
        relevant_sources = filter_cited_sources(accumulated_text, sources)
        
        total_time = end_time - start_total
        llm_time = end_time - start_llm
        retrieval_time = end_retrieval - start_retrieval
        
        # Simple token estimation (approx 4 chars per token)
        estimated_tokens = len(accumulated_text) // 4
        
        performance = {
            "total_time": round(total_time, 2),
            "llm_time": round(llm_time, 2),
            "retrieval_time": round(retrieval_time, 2)
        }
        
        current_model = Config.OLLAMA_LLM_MODEL if Config.LLM_PROVIDER == "ollama" else Config.LLM_PROVIDER.upper() # Fallback to provider name
        
        # Prepare metadata for the frontend Trace and Context modals
        yield json.dumps({
            "type": "metadata", 
            "sources": relevant_sources, 
            "performance": performance,
            "tokens": estimated_tokens,
            "model": current_model,
            "chunks": used_docs, 
            "search_query": question, 
            "history": conversation_history or [],
            "validation": validation  # Add validation results
        }) + "\n"
        
        yield json.dumps({"type": "done"}) + "\n"

    except ValueError as e:
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
    except Exception as e:
        print(f"DEBUG ERROR: {traceback.format_exc()}")
        yield json.dumps({"type": "error", "content": f"An unexpected error occurred: {str(e)}"}) + "\n"
