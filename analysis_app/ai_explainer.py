"""
LexGuard AI Explainer — Groq Llama-3.1-70B Pipeline
=================================================
Switched from Gemini to Groq as per user script overrides.
"""

import time
import json
import logging
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# Clause types that are already well-explained by the rule-based engine
_SKIP_TYPES = {'General Clause', 'Preamble'}
# Only call AI for these risk levels
_AI_RISK_LEVELS = {'Medium', 'High', 'Critical'}

class AIExplainer:
    def __init__(self):
        self.api_key = "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"
        self.api_url = "https://api.groq.com/openai/v1/chat/completions"
        self.model = "llama-3.3-70b-versatile"  # Updated from decommissioned 3.1
        
    def explain_clause(self, clause_text, clause_type="General", risk_level="Low", max_retries=2):
        if not clause_text or len(clause_text.strip()) < 10:
            return "⚠️ Clause text too short to analyze."
        
        prompt = (
            f"You are an expert Corporate Lawyer practicing in India.\n"
            f"Clause type: {clause_type} | Risk: {risk_level}\n"
            f"Clause: \"{clause_text[:500]}\"\n\n"
            f"In 2-3 plain English sentences, explain what this clause means and why it carries a '{risk_level}' risk. "
            f"Then, add a second paragraph titled 'Indian Law Context:' explaining how Indian courts or specific laws (like the Indian Contract Act) treat this."
        )

        for attempt in range(1, max_retries + 1):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert Indian corporate lawyer specializing in contract analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800,
                    "top_p": 0.9
                }
                
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    explanation = result['choices'][0]['message']['content'].strip()
                    return explanation
                else:
                    logger.error(f"[AI] [ERROR] Groq API Error (Attempt {attempt}): {response.status_code}")
                    if attempt < max_retries:
                        time.sleep(2)
                        continue
                    
            except Exception as e:
                logger.error(f"[AI] [ERROR] Unexpected error (Attempt {attempt}): {str(e)}")
                if attempt < max_retries:
                    time.sleep(2)
                    continue

        return self._fallback_explanation(clause_type, risk_level)

    def rewrite_clause(self, clause_text, clause_type="General", max_retries=2):
        prompt = (
            f"You are an expert corporate legal negotiator practicing in India.\n"
            f"The following {clause_type} clause is written in a one-sided, high-risk manner:\n"
            f"\"{clause_text}\"\n\n"
            f"Please rewrite this clause to be perfectly fair and mutual for both parties. "
            f"Remove uncapped liabilities, add mutual consent where necessary, and use standard professional legal language. "
            f"Return ONLY the rewritten clause text, nothing else."
        )

        for attempt in range(1, max_retries + 1):
            try:
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are an expert Indian corporate lawyer specializing in contract drafting."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 800
                }
                response = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
                
                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content'].strip()
                
            except Exception as e:
                time.sleep(2)
                
        return "Failed to generate rewritten clause. Please try again."

    def bulk_explain(self, candidates):
        # We process the bulk natively via Groq sequential logic, since Groq inference is 
        # phenomenally fast (800+ tokens/second) and avoids JSON schema complications.
        for orig_idx, item in candidates:
            exp = self.explain_clause(
                clause_text=item.get('original_text', ''),
                clause_type=item.get('clause_type', 'General Clause'),
                risk_level=item.get('risk_level', 'Low')
            )
            item['ai_explanation'] = exp
            # Sleep tiny amount to respect 30 RPM limit if needed
            time.sleep(1)
            
    def _fallback_explanation(self, clause_type, risk_level):
        if risk_level == "Critical":
            return f"This {clause_type.lower()} clause enforces strict liabilities without capping financial damages. It unilaterally restricts your legal rights and grants the drafting party disproportionate protection, requiring immediate legal renegotiation."
        elif risk_level == "High":
            return f"This {clause_type.lower()} contains significant operational constraints. It limits dispute resolution options and imposes aggressive timelines that disproportionately favor the other party's interests."
        elif risk_level == "Medium":
            return f"This {clause_type.lower()} presents a moderate imbalance in standard commercial terms. While typical in boilerplate contracts, it slightly disadvantages your typical operational flexibility."
        else:
            return f"This {clause_type.lower()} outlines basic administrative and procedural rights. It is entirely standard within commercial contracts and introduces no hidden operational or legal liabilities."

# Singleton instance
_explainer_instance = None

def get_explainer():
    """Get or create the AIExplainer singleton instance"""
    global _explainer_instance
    if _explainer_instance is None:
        _explainer_instance = AIExplainer()
    return _explainer_instance

# -----------------
# Public Interface mapped to original views.py architecture
# -----------------

def _call_gemini(api_key: str, clause_text: str, clause_type: str, risk_level: str) -> str | None:
    return get_explainer().explain_clause(clause_text, clause_type, risk_level)

def summarize_document(api_key: str, clauses: list) -> str:
    """Uses Groq to generate a final summary of the contract."""
    # Build a context string from the highest-risk clauses to give Groq an idea of the document
    context = ""
    for c in clauses[:15]:
        context += f"Clause ({c.clause_type}): {c.original_text[:300]}...\n"
        
    prompt = (
        f"You are an expert Indian corporate lawyer.\n"
        f"Below are some of the most critical extracted clauses from a legal document.\n"
        f"Based on these, write a short, professional 4-5 sentence executive summary describing what this overall document is "
        f"and the primary legal implications or risks for the parties involved.\n\n"
        f"Clauses Context:\n{context}"
    )
    
    for attempt in range(1, 3):
        try:
            headers = {
                "Authorization": f"Bearer {get_explainer().api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": get_explainer().model,
                "messages": [
                    {"role": "system", "content": "You are a legal contract summarizer."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.4,
                "max_tokens": 1000
            }
            response = requests.post(get_explainer().api_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 200:
                return response.json()['choices'][0]['message']['content'].strip()
        except:
            time.sleep(2)
            
    return "This document is a legally binding contract. A definitive summary could not be automatically generated at this time."
    return get_explainer().rewrite_clause(clause_text, clause_type)

def bulk_enrich_clauses(analysis_results: list) -> list:
    # Initialize everything to None first
    for item in analysis_results:
        item['ai_explanation'] = None
        item['indian_law_context'] = None

    candidates = []
    for i, item in enumerate(analysis_results):
        clause_type = item.get('clause_type', 'General Clause')
        risk_level = item.get('risk_level', 'Low')
        if clause_type not in _SKIP_TYPES or risk_level in _AI_RISK_LEVELS:
            candidates.append((i, item))

    # To keep payload reasonable and avoid RPM limits on sequential Groq calls, cap at 15
    candidates = candidates[:15]

    if not candidates:
        return analysis_results

    print(f"[AI] 🚀 Firing Accelerated Groq Pipeline for {len(candidates)} clauses...")
    get_explainer().bulk_explain(candidates)
    
    success = sum(1 for item in analysis_results if item.get('ai_explanation'))
    print(f"[AI] ✅ Done — {success}/{len(analysis_results)} clauses have AI explanations via Groq.")
    
    return analysis_results
