import os
import streamlit as st
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from collections import defaultdict
import time
import re

import faiss
from sentence_transformers import SentenceTransformer
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENROUTER_API_KEY = "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"  # User will add their key
MODEL_NAME = "nvidia/nemotron-nano-9b-v2:free"
ARTIFACTS_DIR = "./"

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="🤖 Agentic TensorFlow Debugger",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .step-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
        transition: transform 0.2s;
    }
    
    .step-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.1);
    }
    
    .metric-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #667eea;
        margin: 0.5rem 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .learning-badge {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
    }
    
    .success-badge {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .failed-badge {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        color: white;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .agent-thinking {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #ff6b6b;
        margin: 1rem 0;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .code-container {
        background: #1e1e1e;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .feedback-card {
        background: white;
        padding: 1.2rem;
        border-radius: 10px;
        margin: 0.8rem 0;
        border-left: 3px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    .tab-content {
        padding: 2rem;
        background: white;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    .progress-ring {
        display: inline-block;
        width: 60px;
        height: 60px;
        border: 4px solid #f3f3f3;
        border-top: 4px solid #667eea;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# OPENROUTER LLM CLIENT
# ============================================================================

class OpenRouterLLM:
    """OpenRouter API client."""
    
    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key if api_key else "dummy_key"
        )
        self.model = model
    
    def invoke(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 2000) -> Dict[str, str]:
        """Invoke the model."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return {"content": response.choices[0].message.content}
        except Exception as e:
            print(f"API Error: {e}")
            return {"content": f"Error: Could not get response from API. {str(e)}"}

# ============================================================================
# REASONING PLANNER WITH REACT
# ============================================================================

class ReasoningPlanner:
    """Plans the reasoning strategy using ReAct framework."""

    def __init__(self, llm):
        self.llm = llm

    def plan(self, query: str) -> Dict[str, Any]:
        """Create reasoning plan using ReAct."""
        messages = [
            {"role": "system", "content": "You are an AI reasoning planner that uses the ReAct framework."},
            {"role": "user", "content": f"""Analyze this error query using ReAct framework:

QUERY: {query}

Use ReAct reasoning:
THOUGHT: Analyze the query complexity and what's needed
ACTION: Determine retrieval strategy and reasoning depth
OBSERVATION: Note key patterns in the error message
FINAL PLAN: Output in this format:

COMPLEXITY: [simple/medium/complex]
RETRIEVAL_STRATEGY: [focused/multi-stage/broad]
RETRIEVAL_K: [2-5]
REASONING_DEPTH: [shallow/medium/deep]
KEY_ENTITIES: [list 3-5 key technical terms from the query]

Be concise and focused."""}
        ]
        
        response = self.llm.invoke(messages)
        content = response["content"]
        
        # Parse response
        plan = {
            "raw_plan": content,
            "complexity": "medium",
            "retrieval_k": 3,
            "reasoning_depth": "medium",
            "key_entities": []
        }
        
        content_lower = content.lower()
        if "complexity: simple" in content_lower:
            plan["complexity"] = "simple"
            plan["retrieval_k"] = 2
        elif "complexity: complex" in content_lower:
            plan["complexity"] = "complex"
            plan["retrieval_k"] = 5
            
        if "retrieval_k:" in content_lower:
            match = re.search(r'retrieval_k:\s*(\d+)', content_lower)
            if match:
                plan["retrieval_k"] = int(match.group(1))
        
        # Extract key entities
        if "key_entities:" in content_lower:
            entities_match = re.search(r'key_entities:(.*?)(?:\n\n|\n[A-Z]|$)', content, re.IGNORECASE | re.DOTALL)
            if entities_match:
                entities_text = entities_match.group(1)
                entities = re.findall(r'[a-zA-Z_]+', entities_text)
                plan["key_entities"] = entities[:5]
        
        return plan

# ============================================================================
# ENHANCED RETRIEVAL
# ============================================================================

class EnhancedRetriever:
    """Multi-stage retrieval system with learning awareness."""

    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.model = artifacts["model"]
        self.index = artifacts["index"]
        self.docs = artifacts["docs"]
        self.graph = artifacts["graph"]

    def retrieve(self, query: str, strategy: str = "focused", top_k: int = 3) -> Dict:
        """Retrieve with different strategies."""
        query_embedding = self.model.encode([query], normalize_embeddings=True)
        distances, indices = self.index.search(query_embedding.astype('float32'), top_k * 2)

        initial_results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx < len(self.docs):
                doc = self.docs[idx]
                initial_results.append({
                    "doc_id": doc.get("id", idx),
                    "title": doc.get("title", "Untitled"),
                    "question": doc.get("question", ""),
                    "answer": doc.get("answer", ""),
                    "link": doc.get("link", "N/A"),
                    "similarity": float(dist),
                    "entities": doc.get("entities", []),
                    "source": doc.get("source", "original"),
                    "timestamp": doc.get("timestamp", "")
                })

        if strategy == "broad":
            expanded = self._expand_via_graph(initial_results, expansion_factor=2)
            return {
                "primary": initial_results[:top_k],
                "expanded": expanded,
                "strategy": strategy
            }
        elif strategy == "focused":
            return {
                "primary": initial_results[:top_k],
                "expanded": [],
                "strategy": strategy
            }
        else:
            expanded = self._expand_via_graph(initial_results, expansion_factor=1)
            return {
                "primary": initial_results[:top_k],
                "expanded": expanded,
                "strategy": strategy
            }

    def _expand_via_graph(self, docs: List[Dict], expansion_factor: int) -> List[Dict]:
        """Expand results using knowledge graph connections."""
        expanded_doc_ids = set()

        for doc in docs[:2]:
            qn = f"q:{doc['doc_id']}"
            if not self.graph.has_node(qn):
                continue

            for entity_node in self.graph.neighbors(qn):
                if entity_node.startswith('e:'):
                    for related_q in self.graph.predecessors(entity_node):
                        if related_q.startswith('q:'):
                            doc_id = int(related_q.split(':')[1])
                            expanded_doc_ids.add(doc_id)
                            if len(expanded_doc_ids) >= expansion_factor:
                                break

        expanded = []
        for doc_id in list(expanded_doc_ids)[:expansion_factor]:
            if doc_id < len(self.docs):
                expanded.append(self.docs[doc_id])

        return expanded

# ============================================================================
# ADVANCED REACT AGENT
# ============================================================================

class AdvancedReActAgent:
    """ReAct agent with enhanced reasoning and reflection."""

    def __init__(self, artifacts, llm):
        self.artifacts = artifacts
        self.llm = llm
        self.planner = ReasoningPlanner(llm)
        self.retriever = EnhancedRetriever(artifacts)

    def analyze(self, error_message: str, progress_container) -> Dict[str, Any]:
        """Main analysis function with ReAct framework."""
        
        # Step 1: Planning
        with progress_container:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("### 🧠 Step 1: ReAct Planning Phase")
            with st.spinner("Agent is analyzing query complexity..."):
                plan = self.planner.plan(error_message)
                time.sleep(0.3)
            
            st.success("✅ Planning Complete")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown('<div class="metric-container"><div class="metric-label">Complexity</div><div class="metric-value">{}</div></div>'.format(plan['complexity'].upper()), unsafe_allow_html=True)
            with col2:
                st.markdown('<div class="metric-container"><div class="metric-label">Retrieval K</div><div class="metric-value">{}</div></div>'.format(plan['retrieval_k']), unsafe_allow_html=True)
            with col3:
                st.markdown('<div class="metric-container"><div class="metric-label">Strategy</div><div class="metric-value" style="font-size:1.2rem;">{}</div></div>'.format(plan.get('retrieval_strategy', 'focused').upper()), unsafe_allow_html=True)
            with col4:
                st.markdown('<div class="metric-container"><div class="metric-label">Depth</div><div class="metric-value" style="font-size:1.5rem;">{}</div></div>'.format(plan['reasoning_depth'].upper()), unsafe_allow_html=True)
            
            if plan.get('key_entities'):
                st.info(f"🎯 **Key Entities Identified:** {', '.join(plan['key_entities'])}")
            
            with st.expander("📋 View Full ReAct Plan"):
                st.code(plan['raw_plan'], language="text")
            st.markdown('</div>', unsafe_allow_html=True)

        # Step 2: Retrieval with ReAct
        with progress_container:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("### 🔍 Step 2: ReAct Retrieval Phase")
            with st.spinner("ACTION: Searching knowledge base..."):
                retrieval_result = self.retriever.retrieve(
                    error_message,
                    strategy="multi-stage" if plan['complexity'] == "complex" else "focused",
                    top_k=plan['retrieval_k']
                )
                time.sleep(0.3)
            
            learned_docs = [d for d in retrieval_result['primary'] if d.get('source') == 'feedback']
            
            st.success(f"OBSERVATION: Retrieved {len(retrieval_result['primary'])} primary documents")
            
            if learned_docs:
                st.markdown(f'<div class="learning-badge">🧠 {len(learned_docs)} LEARNED DOCUMENT(S)</div>', unsafe_allow_html=True)
                st.info(f"💡 **The system has learned from {len(learned_docs)} previous user feedback!**")
            
            with st.expander("📚 View Retrieved Documents"):
                for i, doc in enumerate(retrieval_result['primary'], 1):
                    badge = ' <span class="learning-badge">🧠 LEARNED</span>' if doc.get('source') == 'feedback' else ''
                    st.markdown(f"**Document {i}** (Similarity: {doc['similarity']:.3f}){badge}", unsafe_allow_html=True)
                    st.markdown(f"*{doc['title']}*")
                    if doc['link'] != "N/A":
                        st.markdown(f"[🔗 Source]({doc['link']})")
                    st.divider()
            st.markdown('</div>', unsafe_allow_html=True)

        # Step 3: Knowledge Graph Reasoning
        with progress_container:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("### 🕸️ Step 3: Knowledge Graph Traversal")
            with st.spinner("THOUGHT: Extracting entity relationships..."):
                kg_contexts = []
                for doc in retrieval_result['primary'][:2]:
                    context = self._get_kg_context(doc["doc_id"])
                    if context:
                        kg_contexts.append(context)
                time.sleep(0.3)
            
            st.success(f"OBSERVATION: Extracted context from {len(kg_contexts)} documents")
            
            if kg_contexts:
                with st.expander("🔗 View Entity Network"):
                    for ctx in kg_contexts:
                        entities = [e["name"] for e in ctx["entities"][:5]]
                        st.markdown(f"**Connected Entities:** {', '.join(entities)}")
                        
                        if ctx.get("entity_connections"):
                            st.markdown("**Relationship Graph:**")
                            for conn in ctx["entity_connections"][:3]:
                                related = [f"{r['name']} (w:{r['weight']:.1f})" for r in conn['related_to'][:2]]
                                st.markdown(f"  • `{conn['entity']}` ➔ {', '.join(related)}")
            st.markdown('</div>', unsafe_allow_html=True)

        # Step 4: ReAct Reasoning Loop
        with progress_container:
            st.markdown('<div class="agent-thinking">', unsafe_allow_html=True)
            st.markdown("### 🤖 Step 4: ReAct Reasoning Loop")
            st.markdown("*Agent is performing iterative thought-action-observation cycles...*")
            st.markdown('</div>', unsafe_allow_html=True)
            
            with st.spinner("Processing..."):
                docs_text = self._format_docs(retrieval_result['primary'])
                kg_text = self._format_kg_context(kg_contexts)
                
                messages = [
                    {"role": "system", "content": "You are an expert TensorFlow debugging assistant using ReAct framework."},
                    {"role": "user", "content": f"""Use ReAct framework to solve this error:

ERROR: {error_message}

RETRIEVED KNOWLEDGE:
{docs_text}

KNOWLEDGE GRAPH CONTEXT:
{kg_text}

Apply ReAct reasoning:
THOUGHT 1: Analyze error type and root cause
ACTION 1: Examine retrieved solutions
OBSERVATION 1: Key patterns identified
THOUGHT 2: Synthesize solution approach
ACTION 2: Validate against knowledge graph
OBSERVATION 2: Check solution completeness
REFLECTION: Assess confidence level
FINAL ANSWER: Provide detailed solution with:
1. Root cause analysis
2. Step-by-step fix
3. Code examples if applicable
4. Confidence level (High/Medium/Low)
5. Related issues to watch for

Be thorough and actionable."""}
                ]
                
                response = self.llm.invoke(messages, temperature=0.2)
                time.sleep(0.3)
            
            st.success("✅ ReAct Reasoning Complete")

        # Step 5: Self-Reflection
        with progress_container:
            st.markdown('<div class="step-card">', unsafe_allow_html=True)
            st.markdown("### 🔄 Step 5: Self-Reflection & Quality Check")
            with st.spinner("Agent is evaluating its own response..."):
                reflection = self._reflect(response["content"], docs_text)
                time.sleep(0.3)
            
            st.success("✅ Quality Assurance Complete")
            col1, col2, col3 = st.columns(3)
            with col1:
                quality_color = {"excellent": "🟢", "good": "🟡", "fair": "🟠", "poor": "🔴"}
                st.markdown(f'<div class="metric-container"><div class="metric-label">Quality</div><div class="metric-value" style="font-size:2rem;">{quality_color.get(reflection.get("quality", "fair"), "⚪")}</div><div style="margin-top:0.5rem;">{reflection.get("quality", "N/A").upper()}</div></div>', unsafe_allow_html=True)
            with col2:
                conf_color = {"high": "🟢", "medium": "🟡", "low": "🔴"}
                st.markdown(f'<div class="metric-container"><div class="metric-label">Confidence</div><div class="metric-value" style="font-size:2rem;">{conf_color.get(reflection.get("confidence", "medium"), "⚪")}</div><div style="margin-top:0.5rem;">{reflection.get("confidence", "N/A").upper()}</div></div>', unsafe_allow_html=True)
            with col3:
                completeness = reflection.get("completeness", 75)
                st.markdown(f'<div class="metric-container"><div class="metric-label">Completeness</div><div class="metric-value" style="font-size:2rem;">{completeness}%</div></div>', unsafe_allow_html=True)
            
            if reflection.get("gaps"):
                st.warning(f"⚠️ **Potential Gaps:** {reflection['gaps']}")
            st.markdown('</div>', unsafe_allow_html=True)

        return {
            "query": error_message,
            "plan": plan,
            "retrieved_docs": retrieval_result['primary'],
            "expanded_docs": retrieval_result['expanded'],
            "kg_context": kg_contexts,
            "reasoning": response["content"],
            "reflection": reflection,
            "top_links": [doc["link"] for doc in retrieval_result['primary'][:3] if doc['link'] != "N/A"],
            "timestamp": datetime.now().isoformat(),
            "has_learned_content": len(learned_docs) > 0,
            "learned_doc_count": len(learned_docs)
        }

    def _get_kg_context(self, doc_id: int) -> Optional[Dict]:
        """Extract KG context with enhanced details."""
        G = self.artifacts["graph"]
        qn = f"q:{doc_id}"

        if not G.has_node(qn):
            return None

        context = {"entities": [], "entity_connections": []}

        for neighbor in G.neighbors(qn):
            if neighbor.startswith('e:'):
                entity_data = G.nodes[neighbor]
                context["entities"].append({
                    "name": entity_data.get("name", ""),
                    "importance": entity_data.get("importance", 0),
                    "source": entity_data.get("source", "original")
                })

                related = []
                for rel_entity in G.neighbors(neighbor):
                    if rel_entity.startswith('e:'):
                        edge_data = G.get_edge_data(neighbor, rel_entity) or G.get_edge_data(rel_entity, neighbor)
                        if edge_data and edge_data.get("label") == "RELATED_TO":
                            related.append({
                                "name": G.nodes[rel_entity]["name"],
                                "weight": edge_data.get("weight", 1)
                            })

                if related:
                    context["entity_connections"].append({
                        "entity": entity_data.get("name", ""),
                        "related_to": sorted(related, key=lambda x: x["weight"], reverse=True)[:3]
                    })

        context["entities"] = sorted(context["entities"], key=lambda x: x["importance"], reverse=True)
        return context

    def _reflect(self, answer: str, docs_text: str) -> Dict:
        """Self-reflection using ReAct."""
        messages = [
            {"role": "system", "content": "You are a quality assessment agent using ReAct framework."},
            {"role": "user", "content": f"""Evaluate this debugging answer using ReAct:

ANSWER:
{answer[:1000]}

EVIDENCE:
{docs_text[:500]}

Use ReAct to assess:
THOUGHT: Is the diagnosis well-supported?
ACTION: Check solution completeness
OBSERVATION: Identify gaps or uncertainties
FINAL ASSESSMENT:

QUALITY: [excellent/good/fair/poor]
CONFIDENCE: [high/medium/low]
COMPLETENESS: [0-100%]
GAPS: [list any gaps or None]

Be critical and honest."""}
        ]
        
        response = self.llm.invoke(messages, temperature=0.1)
        content = response["content"].lower()
        
        reflection = {"raw": response["content"]}
        
        if "quality: excellent" in content:
            reflection["quality"] = "excellent"
        elif "quality: good" in content:
            reflection["quality"] = "good"
        elif "quality: poor" in content:
            reflection["quality"] = "poor"
        else:
            reflection["quality"] = "fair"

        if "confidence: high" in content:
            reflection["confidence"] = "high"
        elif "confidence: low" in content:
            reflection["confidence"] = "low"
        else:
            reflection["confidence"] = "medium"
        
        # Extract completeness percentage
        completeness_match = re.search(r'completeness:\s*(\d+)', content)
        if completeness_match:
            reflection["completeness"] = int(completeness_match.group(1))
        else:
            reflection["completeness"] = 75
        
        # Extract gaps
        gaps_match = re.search(r'gaps:(.*?)(?:\n\n|\Z)', content, re.DOTALL)
        if gaps_match:
            gaps_text = gaps_match.group(1).strip()
            reflection["gaps"] = gaps_text if gaps_text.lower() != "none" else None

        return reflection

    def _format_docs(self, docs: List[Dict]) -> str:
        """Format documents with learning indicators."""
        formatted = []
        for i, doc in enumerate(docs, 1):
            source_badge = " [LEARNED FROM FEEDBACK]" if doc.get('source') == 'feedback' else ""
            formatted.append(f"""
Document #{i}{source_badge} (Similarity: {doc['similarity']:.3f})
Title: {doc['title']}
Link: {doc.get('link', 'N/A')}
Question: {doc.get('question', '')[:300]}...
Answer: {doc.get('answer', '')[:300]}...
Entities: {', '.join(doc.get('entities', [])[:5])}
""")
        return "\n".join(formatted)

    def _format_kg_context(self, contexts: List[Dict]) -> str:
        """Format KG context."""
        if not contexts:
            return "No knowledge graph context available."

        formatted = []
        for i, ctx in enumerate(contexts, 1):
            entities = [f"{e['name']} (imp:{e['importance']:.2f})" for e in ctx["entities"][:5]]
            formatted.append(f"Document {i} entities: {', '.join(entities)}")

            if ctx.get("entity_connections"):
                for conn in ctx["entity_connections"][:2]:
                    related = [f"{r['name']}(w:{r['weight']:.1f})" for r in conn["related_to"]]
                    formatted.append(f"  • {conn['entity']} ➔ {', '.join(related)}")

        return "\n".join(formatted)

# ============================================================================
# FEEDBACK LEARNING SYSTEM WITH REACT
# ============================================================================

class FeedbackLearningSystem:
    """Handles user feedback and updates system using ReAct framework."""

    def __init__(self, artifacts, model, llm):
        self.artifacts = artifacts
        self.model = model
        self.llm = llm
        self.feedback_log = []
        self.update_queue = []

    def collect_feedback(self, session_data: Dict, worked: bool, new_error: Optional[str] = None) -> Dict:
        """Collect user feedback."""
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "original_query": session_data["query"],
            "worked": worked,
            "new_error": new_error,
            "retrieved_docs": [d["doc_id"] for d in session_data["retrieved_docs"]],
            "confidence": session_data["reflection"].get("confidence", "unknown"),
            "original_answer": session_data["reasoning"][:200]
        }

        self.feedback_log.append(feedback)

        if not worked and new_error:
            self._queue_for_learning(feedback, new_error)

        return feedback

    def _queue_for_learning(self, feedback: Dict, new_error: str):
        """Queue failed case for system improvement."""
        learning_entry = {
            "original_query": feedback["original_query"],
            "new_error": new_error,
            "timestamp": feedback["timestamp"],
            "retrieved_docs": feedback["retrieved_docs"],
            "original_answer": feedback.get("original_answer", "")
        }
        self.update_queue.append(learning_entry)

    def apply_learning_with_react(self) -> Dict[str, Any]:
        """Apply learning using ReAct framework for intelligent updates."""
        if not self.update_queue:
            return {"updated": False, "message": "No pending updates"}
        
        stats = {
            "docs_added": 0,
            "graph_edges_added": 0,
            "graph_nodes_added": 0,
            "embeddings_updated": False,
            "react_decisions": [],
            "details": []
        }
        
        for entry in self.update_queue:
            try:
                # ReAct Phase 1: Analyze the failed case
                messages = [
                    {"role": "system", "content": "You are a learning agent using ReAct to improve the knowledge base."},
                    {"role": "user", "content": f"""Use ReAct to analyze this failed debugging case:

ORIGINAL QUERY: {entry['original_query']}
NEW ERROR INFO: {entry['new_error']}

THOUGHT: Why did the original answer fail?
ACTION: Extract key technical entities and root cause
OBSERVATION: What should be added to knowledge base?

Output format:
KEY_ENTITIES: [list 5-10 technical terms]
ROOT_CAUSE: [brief explanation]
LEARNING_PRIORITY: [high/medium/low]
SHOULD_ADD: [yes/no - should this be added to KB?]"""}
                ]
                
                analysis = self.llm.invoke(messages, temperature=0.1)
                analysis_content = analysis["content"]
                
                # Parse ReAct decision
                should_add = "should_add: yes" in analysis_content.lower()
                
                if not should_add:
                    stats["react_decisions"].append({
                        "query": entry['original_query'][:50],
                        "decision": "skip",
                        "reason": "Low learning value"
                    })
                    continue
                
                # Extract entities using ReAct
                entities = self._extract_entities_with_react(entry['new_error'], analysis_content)
                
                # Create new document
                new_doc_id = len(self.artifacts["docs"])
                
                new_doc = {
                    "id": new_doc_id,
                    "title": f"🧠 Learned: {entry['original_query'][:60]}...",
                    "question": entry['original_query'],
                    "answer": f"UPDATED SOLUTION (learned from user feedback):\n\n{entry['new_error']}\n\nThis solution was validated by user feedback and added to improve future responses.",
                    "link": "user_feedback",
                    "entities": entities,
                    "source": "feedback",
                    "timestamp": entry['timestamp'],
                    "learning_priority": self._extract_priority(analysis_content)
                }
                
                # Add to docs
                self.artifacts["docs"].append(new_doc)
                stats["docs_added"] += 1
                
                # Update FAISS index with new embedding
                combined_text = f"{entry['original_query']} {entry['new_error']}"
                new_embedding = self.model.encode([combined_text], normalize_embeddings=True)
                self.artifacts["index"].add(new_embedding.astype('float32'))
                stats["embeddings_updated"] = True
                
                # Update Knowledge Graph with ReAct reasoning
                G = self.artifacts["graph"]
                qn = f"q:{new_doc_id}"
                G.add_node(qn, type="question", source="feedback", priority=new_doc["learning_priority"])
                stats["graph_nodes_added"] += 1
                
                # Add answer node
                an = f"a:{new_doc_id}"
                G.add_node(an, type="answer", source="feedback")
                G.add_edge(qn, an, label="HAS_ANSWER", weight=1.0)
                stats["graph_edges_added"] += 1
                stats["graph_nodes_added"] += 1
                
                # Add entity nodes and intelligent connections
                for entity in entities:
                    en = f"e:{entity}"
                    
                    # Create or update entity node
                    if not G.has_node(en):
                        G.add_node(en, name=entity, type="entity", importance=0.2, source="feedback")
                        stats["graph_nodes_added"] += 1
                    else:
                        # Increase importance of existing entity
                        current_importance = G.nodes[en].get('importance', 0.1)
                        G.nodes[en]['importance'] = min(1.0, current_importance + 0.1)
                    
                    # Connect question to entity
                    G.add_edge(qn, en, label="MENTIONS", weight=1.0)
                    stats["graph_edges_added"] += 1
                    
                    # Create entity relationships with co-occurrence weighting
                    for other_entity in entities:
                        if other_entity != entity:
                            other_en = f"e:{other_entity}"
                            if G.has_node(other_en):
                                if G.has_edge(en, other_en):
                                    # Strengthen existing connection
                                    G[en][other_en]['weight'] = G[en][other_en].get('weight', 1.0) + 0.5
                                elif G.has_edge(other_en, en):
                                    G[other_en][en]['weight'] = G[other_en][en].get('weight', 1.0) + 0.5
                                else:
                                    # Create new relationship
                                    G.add_edge(en, other_en, label="RELATED_TO", weight=1.0, learned=True)
                                    stats["graph_edges_added"] += 1
                
                # Find and strengthen connections to related existing documents
                for old_doc_id in entry.get('retrieved_docs', [])[:3]:
                    old_qn = f"q:{old_doc_id}"
                    if G.has_node(old_qn) and old_qn != qn:
                        # Create relationship between old and new knowledge
                        G.add_edge(qn, old_qn, label="REFINES", weight=1.5, learned=True)
                        stats["graph_edges_added"] += 1
                
                stats["react_decisions"].append({
                    "query": entry['original_query'][:50],
                    "decision": "add",
                    "entities_added": len(entities),
                    "priority": new_doc["learning_priority"]
                })
                
                stats["details"].append({
                    "doc_id": new_doc_id,
                    "entities": entities,
                    "edges_added": len(entities) * 2,
                    "priority": new_doc["learning_priority"]
                })
                
            except Exception as e:
                st.error(f"Error processing learning entry: {e}")
                continue
        
        # Clear queue after processing
        self.update_queue.clear()
        
        # Update graph metrics (centrality, PageRank, etc.)
        if stats["graph_edges_added"] > 0:
            self._update_graph_metrics()
        
        return stats
    
    def _extract_entities_with_react(self, text: str, analysis: str) -> List[str]:
        """Extract entities using ReAct analysis."""
        entities = set()
        
        # Extract from ReAct analysis
        if "key_entities:" in analysis.lower():
            entities_match = re.search(r'key_entities:(.*?)(?:\n\n|\n[A-Z]|$)', analysis, re.IGNORECASE | re.DOTALL)
            if entities_match:
                entities_text = entities_match.group(1)
                extracted = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]+', entities_text)
                entities.update(extracted)
        
        # Enhanced pattern matching for TensorFlow/ML entities
        entity_patterns = [
            r'(InvalidArgumentError|ValueError|TypeError|RuntimeError|ResourceExhaustedError|IndexError)',
            r'(embedding|embeddings|tokenizer|vocab_size|input_dim|output_dim|batch_size|seq_length)',
            r'(BERT|GPT|Transformer|LSTM|CNN|RNN|Attention)',
            r'(indices|dimensions|shape|dtype|tensor|array)',
            r'(TensorFlow|Keras|PyTorch|tf\.)',
            r'(layer|model|activation|optimizer|loss)',
            r'(training|inference|prediction|validation)',
        ]
        
        for pattern in entity_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.update([m for m in matches if m and len(m) > 2])
        
        # Common technical keywords
        keywords = [
            "embedding", "tokenizer", "dimensions", "vocab_size", "shape", 
            "model", "layer", "input", "output", "error", "mismatch",
            "tensor", "batch", "sequence", "index", "dtype"
        ]
        
        text_lower = text.lower()
        for keyword in keywords:
            if keyword in text_lower:
                entities.add(keyword)
        
        return list(entities)[:10]  # Limit to 10 most relevant
    
    def _extract_priority(self, analysis: str) -> str:
        """Extract learning priority from ReAct analysis."""
        analysis_lower = analysis.lower()
        if "priority: high" in analysis_lower:
            return "high"
        elif "priority: low" in analysis_lower:
            return "low"
        else:
            return "medium"
    
    def _update_graph_metrics(self):
        """Update graph-wide metrics after learning."""
        G = self.artifacts["graph"]
        
        # Update entity importance using degree centrality
        entity_nodes = [n for n in G.nodes() if n.startswith('e:')]
        
        if len(entity_nodes) > 0:
            try:
                centrality = nx.degree_centrality(G)
                for node in entity_nodes:
                    if node in centrality:
                        G.nodes[node]['importance'] = centrality[node]
            except:
                pass  # Skip if graph is too large or disconnected

    def get_learning_stats(self) -> Dict:
        """Get comprehensive learning statistics."""
        total_feedback = len(self.feedback_log)
        successful = sum(1 for f in self.feedback_log if f['worked'])
        failed = total_feedback - successful
        pending = len(self.update_queue)
        
        # Calculate success rate over time
        recent_feedback = self.feedback_log[-10:] if len(self.feedback_log) >= 10 else self.feedback_log
        recent_success = sum(1 for f in recent_feedback if f['worked'])
        recent_rate = (recent_success / len(recent_feedback) * 100) if recent_feedback else 0
        
        # Count learned documents
        learned_docs = sum(1 for doc in self.artifacts["docs"] if doc.get("source") == "feedback")
        
        return {
            "total_feedback": total_feedback,
            "successful": successful,
            "failed": failed,
            "pending_updates": pending,
            "success_rate": (successful / total_feedback * 100) if total_feedback > 0 else 0,
            "recent_success_rate": recent_rate,
            "learned_documents": learned_docs,
            "knowledge_growth": learned_docs
        }

# ============================================================================
# LEARNING DASHBOARD
# ============================================================================

def show_learning_dashboard():
    """Enhanced learning dashboard with ReAct visualization."""
    st.markdown('<div class="main-header"><h1>🧠 Agentic Learning Dashboard</h1><p>Watch the system evolve through ReAct-driven learning</p></div>', unsafe_allow_html=True)
    
    stats = st.session_state.learning_system.get_learning_stats()
    
    # Top metrics with enhanced styling
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'''<div class="metric-container">
            <div class="metric-label">Total Interactions</div>
            <div class="metric-value">{stats['total_feedback']}</div>
        </div>''', unsafe_allow_html=True)
    with col2:
        delta_color = "🟢" if stats['success_rate'] > 70 else "🟡" if stats['success_rate'] > 50 else "🔴"
        st.markdown(f'''<div class="metric-container">
            <div class="metric-label">Success Rate</div>
            <div class="metric-value">{delta_color} {stats['success_rate']:.1f}%</div>
        </div>''', unsafe_allow_html=True)
    with col3:
        st.markdown(f'''<div class="metric-container">
            <div class="metric-label">Learning Queue</div>
            <div class="metric-value" style="color: {"#f5576c" if stats["pending_updates"] > 0 else "#38ef7d"}">{stats['pending_updates']}</div>
        </div>''', unsafe_allow_html=True)
    with col4:
        st.markdown(f'''<div class="metric-container">
            <div class="metric-label">Learned Docs</div>
            <div class="metric-value" style="color: #667eea">{stats['learned_documents']}</div>
        </div>''', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Knowledge base growth visualization
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("📈 Knowledge Base Growth")
        total_docs = len(st.session_state.artifacts['docs'])
        original_docs = total_docs - stats['learned_documents']
        
        growth_data = {
            "Category": ["Original KB", "Learned from Feedback"],
            "Count": [original_docs, stats['learned_documents']]
        }
        st.bar_chart(growth_data, x="Category", y="Count", color="#667eea")
    
    with col2:
        st.subheader("🕸️ Graph Stats")
        st.metric("Total Nodes", st.session_state.artifacts['graph'].number_of_nodes())
        st.metric("Total Edges", st.session_state.artifacts['graph'].number_of_edges())
        entity_nodes = sum(1 for n in st.session_state.artifacts['graph'].nodes() if n.startswith('e:'))
        st.metric("Entities", entity_nodes)
    
    st.markdown("---")
    
    # Learning action center
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("🚀 Learning Action Center")
    
    if st.session_state.learning_system.update_queue:
        st.warning(f"⚠️ **{len(st.session_state.learning_system.update_queue)} failed cases waiting to be learned**")
        
        # Show preview of pending items
        with st.expander("👀 Preview Pending Learning Items"):
            for i, item in enumerate(st.session_state.learning_system.update_queue[:3], 1):
                st.markdown(f'''<div class="feedback-card">
                    <strong>Item {i}</strong><br>
                    <em>Query:</em> {item['original_query'][:80]}...<br>
                    <em>New Info:</em> {item['new_error'][:100]}...
                </div>''', unsafe_allow_html=True)
            if len(st.session_state.learning_system.update_queue) > 3:
                st.info(f"...and {len(st.session_state.learning_system.update_queue) - 3} more items")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info("💡 **Ready to evolve!** Click the button to apply ReAct-driven learning to improve the system.")
        with col2:
            if st.button("🧠 APPLY REACT LEARNING", type="primary", use_container_width=True):
                with st.spinner("🔄 ReAct Learning in Progress..."):
                    # Show before state
                    before_docs = len(st.session_state.artifacts['docs'])
                    before_nodes = st.session_state.artifacts['graph'].number_of_nodes()
                    before_edges = st.session_state.artifacts['graph'].number_of_edges()
                    
                    # Create progress visualization
                    progress_placeholder = st.empty()
                    
                    with progress_placeholder.container():
                        st.markdown('<div class="agent-thinking">🤖 Agent is using ReAct to analyze failed cases...</div>', unsafe_allow_html=True)
                        time.sleep(1)
                    
                    # Apply learning
                    result = st.session_state.learning_system.apply_learning_with_react()
                    
                    # Show after state
                    after_docs = len(st.session_state.artifacts['docs'])
                    after_nodes = st.session_state.artifacts['graph'].number_of_nodes()
                    after_edges = st.session_state.artifacts['graph'].number_of_edges()
                    
                    progress_placeholder.empty()
                    
                    st.success("✅ ReAct Learning Applied Successfully!")
                    st.balloons()
                    
                    # Show impact metrics
                    st.markdown("### 📊 Learning Impact")
                    impact_col1, impact_col2, impact_col3 = st.columns(3)
                    with impact_col1:
                        st.markdown(f'''<div class="metric-container">
                            <div class="metric-label">Documents Added</div>
                            <div class="metric-value" style="color: #38ef7d">+{after_docs - before_docs}</div>
                        </div>''', unsafe_allow_html=True)
                    with impact_col2:
                        st.markdown(f'''<div class="metric-container">
                            <div class="metric-label">Graph Nodes Added</div>
                            <div class="metric-value" style="color: #38ef7d">+{after_nodes - before_nodes}</div>
                        </div>''', unsafe_allow_html=True)
                    with impact_col3:
                        st.markdown(f'''<div class="metric-container">
                            <div class="metric-label">Graph Edges Added</div>
                            <div class="metric-value" style="color: #38ef7d">+{after_edges - before_edges}</div>
                        </div>''', unsafe_allow_html=True)
                    
                    # Show ReAct decisions
                    if result.get("react_decisions"):
                        st.markdown("### 🤖 ReAct Decisions Made")
                        for decision in result["react_decisions"]:
                            if decision["decision"] == "add":
                                st.markdown(f'''<div class="feedback-card">
                                    <span class="success-badge">✅ ADDED</span> 
                                    <strong>{decision['query']}</strong><br>
                                    Priority: {decision.get('priority', 'medium').upper()} | 
                                    Entities: {decision.get('entities_added', 0)}
                                </div>''', unsafe_allow_html=True)
                            else:
                                st.markdown(f'''<div class="feedback-card">
                                    <span class="failed-badge">⏭️ SKIPPED</span> 
                                    <strong>{decision['query']}</strong><br>
                                    Reason: {decision.get('reason', 'N/A')}
                                </div>''', unsafe_allow_html=True)
                    
                    time.sleep(2)
                    st.rerun()
    else:
        st.success("✨ **System is fully up-to-date!** No pending learning items.")
        st.info("The system will automatically queue failed queries for learning when users provide feedback.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Feedback history with enhanced UI
    st.markdown('<div class="tab-content">', unsafe_allow_html=True)
    st.subheader("📜 Feedback History")
    
    if st.session_state.learning_system.feedback_log:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_type = st.selectbox("Filter by Result", ["All", "Successful ✅", "Failed ❌"])
        with col2:
            sort_order = st.selectbox("Sort by", ["Newest First", "Oldest First"])
        with col3:
            limit = st.slider("Show entries", 5, 50, 15)
        
        # Apply filters
        filtered = st.session_state.learning_system.feedback_log.copy()
        if filter_type == "Successful ✅":
            filtered = [f for f in filtered if f['worked']]
        elif filter_type == "Failed ❌":
            filtered = [f for f in filtered if not f['worked']]
        
        if sort_order == "Oldest First":
            filtered = filtered[:limit]
        else:
            filtered = list(reversed(filtered))[:limit]
        
        # Display feedback items
        for i, feedback in enumerate(filtered, 1):
            status_badge = '<span class="success-badge">✅ WORKED</span>' if feedback['worked'] else '<span class="failed-badge">❌ FAILED</span>'
            confidence_badge = f'<span style="background:#667eea;color:white;padding:0.3rem 0.6rem;border-radius:12px;font-size:0.7rem;">Confidence: {feedback["confidence"].upper()}</span>'
            
            with st.expander(f"Feedback #{len(st.session_state.learning_system.feedback_log) - i + 1} | {feedback['timestamp'][:19]}"):
                st.markdown(f'''<div class="feedback-card">
                    {status_badge} {confidence_badge}<br><br>
                    <strong>Query:</strong> {feedback['original_query']}<br>
                    <strong>Timestamp:</strong> {feedback['timestamp']}
                </div>''', unsafe_allow_html=True)
                
                if feedback.get('new_error'):
                    st.markdown("**Additional Information Provided:**")
                    st.code(feedback['new_error'][:300], language="text")
                
                if not feedback['worked']:
                    st.warning("⚠️ This case was queued for learning")
    else:
        st.info("No feedback collected yet. Start debugging to see history!")
    
    st.markdown('</div>', unsafe_allow_html=True)

# ============================================================================
# STREAMLIT APP
# ============================================================================

@st.cache_resource
def load_all_artifacts():
    """Load artifacts with caching."""
    with st.spinner("📦 Loading AI models and knowledge base..."):
        try:
            embeddings = np.load(os.path.join(ARTIFACTS_DIR, "embeddings.npy"))
            index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "faiss_index.index"))

            with open(os.path.join(ARTIFACTS_DIR, "processed_docs.json"), 'r') as f:
                docs = json.load(f)

            with open(os.path.join(ARTIFACTS_DIR, "kg_networkx.gpickle"), 'rb') as f:
                G = pickle.load(f)

            model = SentenceTransformer('all-MiniLM-L6-v2')

            return {
                "embeddings": embeddings,
                "index": index,
                "docs": docs,
                "graph": G,
                "model": model
            }
        except Exception as e:
            st.error(f"Error loading artifacts: {e}")
            return None

def main():
    """Main Streamlit application."""
    
    # Header with gradient
    st.markdown('<div class="main-header"><h1>🤖 Agentic TensorFlow Debugger</h1><p>AI-Powered Debugging with ReAct Reasoning & Self-Learning</p></div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        
        api_key = st.text_input("OpenRouter API Key", type="password", value=OPENROUTER_API_KEY, 
                                help="Get your free API key from https://openrouter.ai/")
        
        if st.button("💾 Save API Key"):
            if api_key:
                st.session_state.api_key = api_key
                st.success("API Key saved!")
            else:
                st.error("Please enter an API key")
        
        st.divider()
        
        st.markdown("### ℹ️ About")
        st.markdown("""
        This intelligent debugger uses:
        - **ReAct Reasoning** 🧠
        - **Knowledge Graph** 🕸️
        - **Self-Reflection** 🔄
        - **Continuous Learning** 📈
        - **Agentic Updates** 🤖
        """)
        
        st.divider()
        
        st.markdown("### 📊 System Stats")
        if 'artifacts' in st.session_state and st.session_state.artifacts:
            artifacts = st.session_state.artifacts
            st.metric("Documents", len(artifacts['docs']))
            st.metric("Graph Nodes", artifacts['graph'].number_of_nodes())
            st.metric("Graph Edges", artifacts['graph'].number_of_edges())
            
            if 'learning_system' in st.session_state:
                stats = st.session_state.learning_system.get_learning_stats()
                st.metric("Learned Docs", stats['learned_documents'])
        
        st.divider()
        
        st.markdown("### 🔧 Model")
        st.code(MODEL_NAME, language="text")
        st.caption("Powered by NVIDIA Nemotron via OpenRouter")
    
    # Initialize session state
    if 'artifacts' not in st.session_state:
        artifacts = load_all_artifacts()
        if artifacts:
            st.session_state.artifacts = artifacts
            api_key = st.session_state.get('api_key', OPENROUTER_API_KEY)
            llm = OpenRouterLLM(api_key, MODEL_NAME)
            st.session_state.llm = llm
            st.session_state.agent = AdvancedReActAgent(artifacts, llm)
            st.session_state.learning_system = FeedbackLearningSystem(
                artifacts, 
                artifacts["model"],
                llm
            )
            st.session_state.chat_history = []
            st.success("✅ System initialized successfully!")
        else:
            st.error("Failed to load artifacts. Please check the ARTIFACTS_DIR path.")
            st.stop()
    
    # Tabs
    tab1, tab2 = st.tabs(["💬 Debug Assistant", "🧠 Learning Dashboard"])
    
    with tab1:
        st.markdown('<div class="tab-content">', unsafe_allow_html=True)
        
        # Display chat history
        for message in st.session_state.chat_history:
            if message["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(message["content"])
            else:
                with st.chat_message("assistant"):
                    if "result" in message:
                        result = message["result"]
                        
                        # Show if learned content was used
                        if result.get("has_learned_content"):
                            st.markdown(f'<div class="learning-badge">🧠 Using {result["learned_doc_count"]} learned document(s)</div>', unsafe_allow_html=True)
                        
                        st.markdown("### 💡 Solution")
                        st.markdown(result["reasoning"])
                        
                        if result["top_links"]:
                            with st.expander("🔗 Reference Links"):
                                for i, link in enumerate(result["top_links"], 1):
                                    st.markdown(f"{i}. [{link}]({link})")
        
        # Chat input
        user_input = st.chat_input("Enter your TensorFlow error message or question...")
        
        if user_input:
            # Check API key
            if not st.session_state.get('api_key') and not OPENROUTER_API_KEY:
                st.error("⚠️ Please enter your OpenRouter API key in the sidebar first!")
                st.stop()
            
            # Add user message
            st.session_state.chat_history.append({
                "role": "user",
                "content": user_input
            })
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Display assistant response
            with st.chat_message("assistant"):
                st.markdown("## 🔍 Analyzing with ReAct Framework...")
                
                progress_container = st.container()
                
                try:
                    result = st.session_state.agent.analyze(user_input, progress_container)
                    
                    st.markdown("---")
                    
                    # Show if learned content was used
                    if result.get("has_learned_content"):
                        st.markdown(f'<div class="learning-badge">🧠 Using {result["learned_doc_count"]} learned document(s) from previous feedback!</div>', unsafe_allow_html=True)
                    
                    st.markdown("### 💡 Solution & Analysis")
                    st.markdown(result["reasoning"])
                    
                    if result["top_links"]:
                        st.markdown("### 🔗 Reference Links")
                        for i, link in enumerate(result["top_links"], 1):
                            st.markdown(f"{i}. [{link}]({link})")
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "result": result
                    })
                    
                    # Feedback section
                    st.markdown("---")
                    st.markdown("### 📝 Was this solution helpful?")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if st.button("✅ Yes, it worked!", key=f"worked_{len(st.session_state.chat_history)}", use_container_width=True):
                            feedback = st.session_state.learning_system.collect_feedback(result, True)
                            st.success("Thank you! Your feedback helps improve the system! 🎉")
                            time.sleep(1)
                            st.rerun()
                    
                    with col2:
                        if st.button("❌ No, still having issues", key=f"failed_{len(st.session_state.chat_history)}", use_container_width=True):
                            st.session_state.show_feedback_form = True
                            st.rerun()
                    
                    if st.session_state.get('show_feedback_form', False):
                        st.markdown("#### 📝 Help us learn")
                        new_error = st.text_area(
                            "What happened? Please describe the issue or paste the new error:",
                            placeholder="E.g., 'The solution didn't fix the error. The actual issue was...'",
                            height=100
                        )
                        
                        col_submit, col_cancel = st.columns([1, 1])
                        with col_submit:
                            if st.button("🚀 Submit Feedback", use_container_width=True):
                                if new_error:
                                    feedback = st.session_state.learning_system.collect_feedback(result, False, new_error)
                                    st.success("Feedback recorded! The system will learn from this. 🧠")
                                    st.info("💡 Go to the 'Learning Dashboard' tab to apply the learning!")
                                    st.session_state.show_feedback_form = False
                                    time.sleep(2)
                                    st.rerun()
                                else:
                                    st.warning("Please provide some details to help the system learn.")
                        
                        with col_cancel:
                            if st.button("Cancel", use_container_width=True):
                                st.session_state.show_feedback_form = False
                                st.rerun()
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {e}")
                    st.exception(e)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        show_learning_dashboard()

if __name__ == "__main__":
    main()