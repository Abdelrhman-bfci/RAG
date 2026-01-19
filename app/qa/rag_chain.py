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
from app.vectorstore.faiss_store import FAISSStore
import time
import traceback
import json

# --- Enhanced Prompt Templates with Chain-of-Thought ---
STRICT_RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a highly precise academic assistant. Follow this systematic process:

STEP 1: ANALYZE THE QUESTION
- Identify the key concepts and requirements
- Determine the question language (English/Arabic) - THIS IS CRITICAL
- Understand what type of answer is needed (factual, list, explanation, etc.)

STEP 2: ASSESS CONTEXT RELEVANCE  
- Review the provided context sources and chunks
- Identify which sources contain relevant information
- Note the number of sources and chunks available

STEP 3: CONSTRUCT ANSWER
{language_instruction}

CRITICAL COMPLIANCE RULES:
1. **Answer ONLY using information from the Context** - No external knowledge
2. **If answer not in Context**: Say "I cannot answer this based on the provided documents" (translate to Arabic if question is Arabic)
3. **MANDATORY INLINE CITATIONS**: Every single fact, claim, or item MUST be followed by [Source, Page]
   - CORRECT: "Software Engineering [CS_Catalog.pdf, Page 12]"
   - WRONG: Listing sources only at the end
4. **COMPLETE LISTS**: When listing items (courses, programs, requirements), provide ALL items found. DO NOT truncate or say "and more"
5. **FULL RESPONSES**: If answer requires long response, provide COMPLETE answer without cutting short

DATABASE & RELATION RULES:
- **Resolved Foreign Keys**: Use `_name` fields (like `department_name`) for human-readable answers
- **Implicit Relations**: Connect related data (e.g., courses → departments → faculties)
- **Schema Awareness**: `institutes` = Faculties/Colleges, `departments` = academic departments

STEP 4: QUALITY CHECK
- Verify all facts are cited with [Source, Page]
- Confirm answer language matches question language
- Ensure completeness (no truncated lists)
- Check that only context information is used"""),
    ("human", """Context (organized by source):
{context}

Question: {question}

Think step-by-step and provide a complete, well-cited answer in the same language as the question.""")
])

DEEP_THINKING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert academic analyst. Follow this systematic approach:

STEP 1: UNDERSTAND THE QUESTION
{language_instruction}
- Identify key concepts and analytical requirements
- Determine the depth of analysis needed
- Note the question language (CRITICAL: English question → English answer, Arabic question → Arabic answer)

STEP 2: ANALYZE AVAILABLE CONTEXT
- Review all {len(sources)} sources provided
- Identify relevant information across sources
- Note connections and relationships between data points
- Identify any gaps in available information

STEP 3: SYNTHESIZE COMPREHENSIVE ANSWER
- Combine information from multiple sources
- Provide analytical insights and patterns
- Structure response with clear sections
- Use professional academic language in the SAME language as question

CRITICAL INSTRUCTIONS:
1. **MANDATORY INLINE CITATIONS**: Every major statement and EVERY item in lists MUST have [Source, Page]
   - FORMAT: [Document.pdf, Page X] or [Table: TableName]
   - Attach citations to each specific point, not grouped at end
2. **COMPLETE INFORMATION**: When listing items, provide ALL items found. Never truncate or summarize
3. **FULL RESPONSES**: Provide complete analysis without cutting short. You have sufficient token capacity
4. **CONTEXT ONLY**: Use only information from provided context
5. **LANGUAGE MATCHING**: 
   - Question in English → Response in English
   - Question in Arabic → Response in Arabic
   - Translate context information to match question language if needed

RELATIONAL DATA LOGIC:
- **Connect the Dots**: Link data across tables (Course → Department → Faculty)
- **Use enriched fields**: Prefer `_name` fields for human-readable output
- **Terminology**: Treat `institutes` as Faculties/Colleges
- **Aggregation**: Summarize trends and groupings where appropriate

STEP 4: VERIFY QUALITY
- All facts cited inline
- Language matches question
- Complete information provided
- Professional academic tone
- Clear structure and organization

If information is missing, clearly state what is unknown in the user's language."""),
    ("human", """Context (organized by source):
{context}

Question: {question}

Provide a comprehensive, analytical answer with complete information and inline citations.""")
])

def get_rag_chain(deep_thinking: bool = False, is_continuation: bool = False, last_answer: str = ""):
    """
    Creates and returns the RAG chain for Question Answering.
    """
    # 1. Initialize Vector Store and Retriever
    faiss_store = FAISSStore()
    vectorstore = faiss_store.load_index()
    
    if not vectorstore:
        raise ValueError("Vector store not found. Please run ingestion first.")

    # Advanced Hybrid Retrieval with Relevance Filtering and MMR
    class AdvancedHybridRetriever:
        def __init__(self, vectorstore):
            self.vectorstore = vectorstore
            
        def _extract_keywords(self, query):
            """Extract keywords from query based on language."""
            import re
            is_arabic = bool(re.search('[\u0600-\u06FF]', query))
            
            if is_arabic:
                keywords = [w.lower() for w in query.split() if len(w) >= 2]
            else:
                keywords = [w.lower() for w in query.split() if len(w) > 3]
            
            return keywords
        
        def _calculate_keyword_score(self, content, keywords):
            """Calculate keyword match score for a document."""
            content_lower = content.lower()
            matches = sum(1 for kw in keywords if kw in content_lower)
            return matches / len(keywords) if keywords else 0
        
        def _apply_mmr(self, docs_with_scores, query_embedding, k=40, lambda_param=0.7):
            """
            Apply Maximum Marginal Relevance to reduce redundancy.
            lambda_param: 0 = max diversity, 1 = max relevance
            """
            if len(docs_with_scores) <= k:
                return [doc for doc, _ in docs_with_scores]
            
            # Separate docs and scores
            docs = [doc for doc, _ in docs_with_scores]
            scores = [score for _, score in docs_with_scores]
            
            # Simple MMR: select diverse documents
            selected = []
            selected_indices = []
            remaining_indices = list(range(len(docs)))
            
            # Start with highest scoring document
            best_idx = scores.index(min(scores))  # Lower L2 distance is better
            selected.append(docs[best_idx])
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)
            
            # Iteratively select documents that are relevant but diverse
            while len(selected) < k and remaining_indices:
                best_score = float('inf')
                best_idx = None
                
                for idx in remaining_indices:
                    # Relevance score (lower is better for L2)
                    relevance = scores[idx]
                    
                    # Diversity: check similarity to already selected docs
                    # Simple diversity: check content overlap
                    diversity = 0
                    for sel_idx in selected_indices:
                        # Simple Jaccard similarity on words
                        words_current = set(docs[idx].page_content.lower().split())
                        words_selected = set(docs[sel_idx].page_content.lower().split())
                        if words_current and words_selected:
                            overlap = len(words_current & words_selected) / len(words_current | words_selected)
                            diversity = max(diversity, overlap)
                    
                    # MMR score: balance relevance and diversity
                    mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                    
                    if mmr_score < best_score:
                        best_score = mmr_score
                        best_idx = idx
                
                if best_idx is not None:
                    selected.append(docs[best_idx])
                    selected_indices.append(best_idx)
                    remaining_indices.remove(best_idx)
                else:
                    break
            
            return selected
            
        def invoke(self, query):
            # Stage 1: Broad retrieval with similarity scores (increased from 300 to 500)
            try:
                docs_with_scores = self.vectorstore.similarity_search_with_score(query, k=500)
            except:
                # Fallback if similarity_search_with_score not available
                docs = self.vectorstore.similarity_search(query, k=500)
                docs_with_scores = [(doc, 0.0) for doc in docs]
            
            # Stage 2: Filter by relevance threshold
            # FAISS L2 distance: lower is better, typical range 0-2 for similar docs
            relevance_threshold = 1.5
            relevant_docs = [(doc, score) for doc, score in docs_with_scores if score < relevance_threshold]
            
            # If too few docs pass threshold, relax it
            if len(relevant_docs) < 20:
                relevance_threshold = 2.0
                relevant_docs = [(doc, score) for doc, score in docs_with_scores if score < relevance_threshold]
            
            # Stage 3: Keyword boosting
            keywords = self._extract_keywords(query)
            boosted_docs = []
            
            for doc, vec_score in relevant_docs:
                keyword_score = self._calculate_keyword_score(doc.page_content, keywords)
                # Combine scores: lower vector score is better, higher keyword score is better
                # Normalize: boost docs with high keyword matches
                combined_score = vec_score * (1 - keyword_score * 0.3)  # Up to 30% boost
                boosted_docs.append((doc, combined_score))
            
            # Sort by combined score (lower is better)
            boosted_docs.sort(key=lambda x: x[1])
            
            # Stage 4: Apply MMR for diversity
            try:
                # Get query embedding for MMR
                query_embedding = self.vectorstore.embeddings.embed_query(query)
                final_docs = self._apply_mmr(boosted_docs, query_embedding, k=40, lambda_param=0.7)
            except:
                # Fallback: just take top 40
                final_docs = [doc for doc, _ in boosted_docs[:40]]
            
            return final_docs

    retriever = AdvancedHybridRetriever(vectorstore)

    # 2. Select Prompt
    prompt = DEEP_THINKING_PROMPT if deep_thinking else STRICT_RAG_PROMPT

    # 3. Initialize LLM
    if Config.LLM_PROVIDER == "ollama":
        llm = ChatOllama(
            base_url=Config.OLLAMA_BASE_URL,
            model=Config.OLLAMA_LLM_MODEL,
            temperature=0.2 if deep_thinking else 0.1,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            num_predict=-1,  # Unlimited tokens - generate until naturally complete
            stop=[],  # Remove default stop sequences to allow complete responses
            repeat_penalty=1.1  # Slight penalty to avoid repetition while maintaining completeness
        )
    elif Config.LLM_PROVIDER == "vllm":
        llm = ChatOpenAI(
            base_url=Config.VLLM_BASE_URL,
            model=Config.VLLM_MODEL,
            temperature=0.2 if deep_thinking else 0.1,
            api_key="none",
            max_tokens=8192  # Maximum tokens to generate in response
        )
    elif Config.LLM_PROVIDER == "openai":
        llm = ChatOpenAI(
            model=Config.OPENAI_LLM_MODEL,
            base_url=Config.OPENAI_BASE_URL,
            temperature=0.2 if deep_thinking else 0.1,
            openai_api_key=Config.OPENAI_API_KEY,
            max_tokens=8192  # Maximum tokens to generate in response
        )
    elif Config.LLM_PROVIDER == "gemini":
        if ChatGoogleGenerativeAI is None:
            raise ImportError("langchain-google-genai is not installed. Please run 'pip install langchain-google-genai'")
        llm = ChatGoogleGenerativeAI(
            model=Config.GEMINI_MODEL,
            google_api_key=Config.GEMINI_API_KEY,
            temperature=0.2 if deep_thinking else 0.1,
            max_output_tokens=8192  # Maximum tokens to generate in response
        )
    else:
        # Default fallback to Ollama
        llm = ChatOllama(
            model=Config.OLLAMA_LLM_MODEL,
            base_url=Config.OLLAMA_BASE_URL,
            temperature=0.2 if deep_thinking else 0.1,
            num_ctx=Config.OLLAMA_CONTEXT_WINDOW,
            num_predict=-1,  # Unlimited tokens - generate until naturally complete
            stop=[],  # Remove default stop sequences to allow complete responses
            repeat_penalty=1.1  # Slight penalty to avoid repetition while maintaining completeness
        )

    # 4. Construct the Chain with Enhanced Context Formatting
    def format_docs(docs):
        """
        Enhanced context formatting with source grouping and relevance indicators.
        """
        # Group documents by source for better context
        grouped = {}
        for i, doc in enumerate(docs):
            source = os.path.basename(doc.metadata.get("source", "Unknown"))
            if source not in grouped:
                grouped[source] = []
            
            page = doc.metadata.get("page", "N/A")
            if isinstance(page, int):
                page = page + 1
            
            grouped[source].append({
                "content": doc.page_content,
                "page": page,
                "index": i + 1
            })
        
        # Format grouped context
        context_parts = []
        context_parts.append("\n" + "="*60)
        context_parts.append(f"RETRIEVED CONTEXT ({len(docs)} chunks from {len(grouped)} sources)")
        context_parts.append("="*60)
        
        for source_idx, (source, chunks) in enumerate(grouped.items(), 1):
            context_parts.append(f"\n{'─'*60}")
            context_parts.append(f"SOURCE #{source_idx}: {source}")
            context_parts.append(f"CHUNKS: {len(chunks)}")
            context_parts.append(f"{'─'*60}")
            
            for chunk in chunks:
                context_parts.append(f"\n[CHUNK #{chunk['index']}] [PAGE: {chunk['page']}]")
                context_parts.append(f"CONTENT: {chunk['content']}")
                context_parts.append("")  # Empty line for readability
        
        return "\n".join(context_parts)

    def detect_language_instruction(question):
        import re
        is_arabic = bool(re.search('[\u0600-\u06FF]', question))
        
        # Continuation Instruction (English or Arabic)
        cont_instr = ""
        if is_continuation:
            if is_arabic:
                cont_instr = "\nتنبيه مهم: أنت الآن تكمل إجابة سابقة تم قطعها. ابدأ مباشرة من حيث انتهيت. لا تكرر أي جملة أو معلومة ذكرتها سابقاً."
            else:
                cont_instr = "\nIMPORTANT: You are continuing a previous response that was cut off. DO NOT REPEAT any information already provided. Start exactly from the next point or sentence."

        if is_arabic:
            return f"""THE USER ASKED IN ARABIC. YOU MUST ANSWER IN ARABIC. {cont_instr}
            قواعد الاستشهاد (MANDATORY CITATIONS): 
            يجب وضع المرجع بجانب كل حقيقة أو دورة تدريبية أو ملف بصيغة [اسم_الملف, صفحة X]. 
            مثال: [Course_Catalog.pdf, صفحة 5].
            لا تذكر كلمة SOURCE أو PAGE داخل القوسين، فقط الاسم والصفحة.
            لا تكتفي بذكر المراجع في النهاية، بل ضعها بجانب كل نقطة."""
        else:
            return f"""THE USER ASKED IN ENGLISH. YOU MUST ANSWER IN ENGLISH. {cont_instr}
            CITATION RULES (MANDATORY): 
            Every single fact or item must have an inline reference like [Document.pdf, Page X]. 
            Example: [CS_Manual.pdf, Page 12].
            Do NOT include labels like 'SOURCE:' or 'PAGE:' inside the brackets, just the filename and page number.
            Do not just list them at the end; place them next to the specific information."""

    def process_question(q):
        if is_continuation and last_answer:
            return f"--- START OF CONTINUATION REQUEST ---\nYOU PREVIOUSLY SAID:\n[[[ {last_answer} ]]]\n\nNOW CONTINUE DIRECTLY from exactly where you left off. DO NOT REPEAT ANY LIST ITEMS OR SENTENCES FROM ABOVE. Provide the remaining missing information for the question: {q}\n--- CONTINUE BELOW ---"
        return f"{q} [MANDATORY: Cite every single course/item in brackets like [Document.pdf, Page X]]"

    rag_chain = (
        {
            "context": RunnableLambda(lambda x: retriever.invoke(x)) | format_docs, 
            "question": RunnableLambda(process_question),
            "language_instruction": RunnableLambda(detect_language_instruction)
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain, retriever

def filter_cited_sources(answer: str, all_sources: list) -> list:
    """
    Filters the list of all retrieved sources to return only those that are explicitly 
    cited in the answer text (e.g., [Document.pdf]).
    """
    import re
    # Extract text in brackets [source, ...]
    # We look for patterns like [Doc.pdf] or [Doc.pdf, Page X]
    citations = re.findall(r'\[(.*?)\]', answer)
    cited_filenames = set()
    for c in citations:
        # Take first part "File.pdf" from "File.pdf, Page 2"
        part = c.split(',')[0].strip()
        cited_filenames.add(part.lower())
    
    final_sources = []
    for src in all_sources:
        # Check if src matches any cited filename
        # We compare basenames to handle paths/URLs
        basename = os.path.basename(src).lower()
        if basename in cited_filenames or src.lower() in cited_filenames:
            final_sources.append(src)
            
    # If no sources are cited but we have an answer, we might strictly want to return nothing,
    # or fallback. The user request implies "just resource that i get information from".
    # If LLM didn't cite, or we want to be generous, we fallback to all retrieved sources.
    if not final_sources:
        return sorted(list(set(all_sources)))
    
    return sorted(list(set(final_sources)))

def answer_question(question: str, deep_thinking: bool = False, is_continuation: bool = False, last_answer: str = ""):
    """
    Entry point to answer a question with performance metrics and source citations.
    """
    start_total = time.time()
    try:
        chain, retriever = get_rag_chain(deep_thinking=deep_thinking, is_continuation=is_continuation, last_answer=last_answer)
        
        # 1. Retrieve documents manually to get metadata for citations
        docs = retriever.invoke(question)
        sources = sorted(list(set([doc.metadata.get("source", "Unknown") for doc in docs])))
        
        # Capture used docs with content
        used_docs = []
        for doc in docs:
            used_docs.append({
                "source": doc.metadata.get("source", "Unknown"),
                "content": doc.page_content,
                "page": doc.metadata.get("page", 0)
            })

        # 2. Invoke the chain
        start_llm = time.time()
        answer = chain.invoke(question)
        end_time = time.time()
        
        # 3. Filter sources based on citation
        relevant_sources = filter_cited_sources(answer, sources)
        
        total_time = end_time - start_total
        llm_time = end_time - start_llm
        
        performance = f"Total {total_time:.1f}s | LLM {llm_time:.1f}s"
        
        return {
            "answer": answer,
            "sources": relevant_sources,
            "performance": performance,
            "used_docs": used_docs 
        }
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        print(f"DEBUG ERROR: {traceback.format_exc()}")
        return {"error": f"An unexpected error occurred: {str(e)}"}

def stream_answer(question: str, deep_thinking: bool = False, is_continuation: bool = False, last_answer: str = ""):
    """
    Entry point to stream chunks of the LLM response.
    Yields JSON strings containing either chunks or final metadata.
    """
    start_total = time.time()
    try:
        chain, retriever = get_rag_chain(deep_thinking=deep_thinking, is_continuation=is_continuation, last_answer=last_answer)
        
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
        
        # 3. Filter sources for distinct display
        relevant_sources = filter_cited_sources(accumulated_text, sources)
        
        total_time = end_time - start_total
        llm_time = end_time - start_llm
        retrieval_time = end_retrieval - start_retrieval
        
        # Simple token estimation (approx 4 chars per token)
        estimated_tokens = len(accumulated_text) // 4
        
        performance = {
            "total": f"{total_time:.1f}s",
            "llm": f"{llm_time:.1f}s",
            "retrieval": f"{retrieval_time:.1f}s"
        }
        
        current_model = Config.OLLAMA_LLM_MODEL if Config.LLM_PROVIDER == "ollama" else Config.LLM_MODEL
        
        yield json.dumps({
            "type": "metadata", 
            "sources": relevant_sources, 
            "performance": performance,
            "tokens": estimated_tokens,
            "model": current_model,
            "used_docs": used_docs
        }) + "\n"
        
        yield json.dumps({"type": "done"}) + "\n"

    except ValueError as e:
        yield json.dumps({"type": "error", "content": str(e)}) + "\n"
    except Exception as e:
        print(f"DEBUG ERROR: {traceback.format_exc()}")
        yield json.dumps({"type": "error", "content": f"An unexpected error occurred: {str(e)}"}) + "\n"
