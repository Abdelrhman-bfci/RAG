# Retrieval Fix: "Cannot Answer" Issue Resolution

## Problem
The system was returning "I cannot answer this based on the provided documents" even when relevant documents existed in ChromaDB.

## Root Causes Identified

1. **Too Strict Adaptive Threshold**: Using 75th percentile was filtering out relevant documents
2. **Insufficient Initial Retrieval**: Only retrieving 100 documents wasn't enough for some queries
3. **Overly Strict Prompt**: Prompt was too conservative, causing LLM to say "cannot answer" even with partial information
4. **Lack of Fallback Safety**: No guarantee that documents would be returned even if scores were low

## Fixes Applied

### 1. More Lenient Adaptive Threshold ✅

**Before:**
- Used 75th percentile (top 25%) with 80% multiplier
- Could filter out relevant documents

**After:**
- Uses 50th percentile (median) with 70% multiplier
- Much more lenient: `max(p50 * 0.7, p75 * 0.5, threshold * 0.5)`
- For small sets (<5 docs): Uses very low threshold (0.05 minimum)

**Impact:** More documents pass the threshold filter

### 2. Increased Initial Retrieval ✅

**Before:**
- With reranker: 100 documents
- Without reranker: 50 documents

**After:**
- With reranker: 150 documents (+50%)
- Without reranker: 80 documents (+60%)
- Increased `fetch_k` for MMR: `k_search * 3` (was `k_search * 2`)

**Impact:** Better coverage of relevant documents in initial retrieval

### 3. Enhanced Fallback Safety ✅

**Before:**
- Fallback only returned top 5 if threshold filtered everything
- Could still return empty results

**After:**
- Fallback returns top `LLM_K_FINAL * 3` documents (was `* 2`)
- **Guaranteed minimum**: Always returns top 5 documents regardless of score
- Better diversity preservation in fallback

**Impact:** Ensures documents are always returned

### 4. More Lenient Prompt ✅

**Before:**
```
If the answer cannot be reasonably derived from the provided Context...
you MUST output exactly: "I cannot answer this based on the provided documents."
```

**After:**
```
Only if the Context is completely empty or contains ZERO relevant information
should you output: "I cannot answer this based on the provided documents."

If Context contains ANY relevant information (even partial, incomplete, or indirect), you MUST:
- Extract and present that information
- Clearly state what information is available
- Note any limitations or gaps
- Cite the sources used
```

**Impact:** LLM will provide partial answers instead of saying "cannot answer"

### 5. Enhanced Debug Logging ✅

Added comprehensive logging:
- Number of documents retrieved at each stage
- Sample sources and content previews
- Threshold calculations with percentiles
- Score distributions
- Fallback triggers

**Impact:** Easier to diagnose retrieval issues

## Testing Recommendations

1. **Test with "Mohamed Sobh" query:**
   - Check debug logs to see how many documents are retrieved
   - Verify threshold calculation
   - Confirm documents are being returned

2. **Monitor Debug Output:**
   - Look for "DEBUG: Retrieved X documents" messages
   - Check "DEBUG: Adaptive threshold calculated" values
   - Verify "DEBUG: Returning X documents after reranking"

3. **Check Threshold Values:**
   - If threshold is too high (>0.5), documents might be filtered
   - If threshold is very low (<0.1), most documents should pass

## Configuration Tuning

If issues persist, consider adjusting:

```env
# Lower threshold for more lenient filtering
RERANKER_THRESHOLD=0.05  # Default: 0.1

# Increase final document count
LLM_K_FINAL=15  # Default: 10

# Disable reranker if it's too strict
USE_RERANKER=false
```

## Expected Behavior After Fix

1. **More Documents Retrieved**: 150 instead of 100 (with reranker)
2. **Lower Threshold**: Median-based instead of 75th percentile
3. **Guaranteed Results**: Always returns at least 5 documents
4. **Partial Answers**: LLM provides information even if incomplete
5. **Better Logging**: Clear visibility into retrieval process

## Debugging Steps

If still seeing "cannot answer":

1. **Check Debug Logs:**
   ```
   DEBUG: Retrieved X documents from MMR search
   DEBUG: Total documents after retrieval and deduplication: X
   DEBUG: Adaptive threshold calculated: X.XXX
   DEBUG: Returning X documents after reranking
   ```

2. **Verify Documents Exist:**
   - Check ChromaDB directly
   - Verify embeddings are correct
   - Check if query matches document content

3. **Test Query Variations:**
   - Try exact name match: "Mohamed Sobh"
   - Try partial: "Sobh"
   - Try with context: "professor Mohamed Sobh"

4. **Check Reranker Scores:**
   - Look at `rerank_score` in document metadata
   - If scores are very low (<0.1), reranker might not be working well
   - Consider disabling reranker temporarily

## Summary

The fixes make the retrieval system:
- **More lenient** in filtering
- **More comprehensive** in initial retrieval
- **More reliable** with guaranteed fallbacks
- **More informative** with better logging
- **More helpful** with partial answers instead of "cannot answer"
