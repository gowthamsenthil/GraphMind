import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

import faiss
from sentence_transformers import SentenceTransformer
import networkx as nx
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

OPENROUTER_API_KEY = "PASTE YOUR API KEY HERE"
MODEL_NAME = "nvidia/nvidia-nemotron-nano-9b-v2"
ARTIFACTS_DIR = "/Users/gowtham/Downloads/NVIDIA-Agentic"

# ============================================================================
# OPENROUTER LLM CLIENT
# ============================================================================

class OpenRouterLLM:
    """OpenRouter API client."""
    
    def __init__(self, api_key: str, model: str = MODEL_NAME):
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
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

    def analyze(self, error_message: str) -> Dict[str, Any]:
        """Main analysis function with ReAct framework."""
        
        # Step 1: Planning
        print("Step 1: Planning...")
        plan = self.planner.plan(error_message)
        
        # Step 2: Retrieval with ReAct
        print("Step 2: Retrieval...")
        retrieval_result = self.retriever.retrieve(
            error_message,
            strategy="multi-stage" if plan['complexity'] == "complex" else "focused",
            top_k=plan['retrieval_k']
        )
        
        learned_docs = [d for d in retrieval_result['primary'] if d.get('source') == 'feedback']
        
        # Step 3: Knowledge Graph Reasoning
        print("Step 3: Knowledge Graph Traversal...")
        kg_contexts = []
        for doc in retrieval_result['primary'][:2]:
            context = self._get_kg_context(doc["doc_id"])
            if context:
                kg_contexts.append(context)
        
        # Step 4: ReAct Reasoning Loop
        print("Step 4: ReAct Reasoning...")
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
        
        # Step 5: Self-Reflection
        print("Step 5: Self-Reflection...")
        reflection = self._reflect(response["content"], docs_text)

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
                    formatted.append(f"  • {conn['entity']} → {', '.join(related)}")

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
                print(f"Error processing learning entry: {e}")
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
# UTILITY FUNCTIONS
# ============================================================================

def load_all_artifacts():
    """Load artifacts."""
    try:
        print("Loading artifacts...")
        embeddings = np.load(os.path.join(ARTIFACTS_DIR, "embeddings.npy"))
        index = faiss.read_index(os.path.join(ARTIFACTS_DIR, "faiss_index.index"))

        with open(os.path.join(ARTIFACTS_DIR, "processed_docs.json"), 'r') as f:
            docs = json.load(f)

        with open(os.path.join(ARTIFACTS_DIR, "kg_networkx.gpickle"), 'rb') as f:
            G = pickle.load(f)

        model = SentenceTransformer('all-MiniLM-L6-v2')

        print("Artifacts loaded successfully!")
        return {
            "embeddings": embeddings,
            "index": index,
            "docs": docs,
            "graph": G,
            "model": model
        }
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return None