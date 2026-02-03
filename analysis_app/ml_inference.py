import os
import re
import time
import logging
from django.conf import settings

logger = logging.getLogger(__name__)

# Try importing ML libraries. If they fail, fall back to the advanced rule-based engine.
try:
    import torch
    import joblib
    from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
    ML_AVAILABLE = True
    logger.info("PyTorch and Transformers loaded successfully. Real ML inference is available.")
except ImportError as e:
    logger.warning(f"ML libraries not fully available ({e}). Using rule-based legal NLP engine.")
    ML_AVAILABLE = False


ML_MODELS_DIR = os.path.join(settings.BASE_DIR, 'ml_models')

# ────────────────────────────────────────────────────────────────────
#  Keyword pattern library for the rule-based NLP fallback engine
#  Each entry has: patterns (regex), risk_level, clause_type, and
#  rich legal explanation fields.
# ────────────────────────────────────────────────────────────────────
CLAUSE_PATTERNS = [
    # ── CRITICAL patterns ──────────────────────────────────────────
    {
        "patterns": [
            r"terminat[e|ion|ed].{0,60}(without cause|at any time|immediately|with.{0,10}notice)",
            r"without\s+(prior\s+)?notice.{0,40}terminat",
            r"unilateral(ly)?.{0,40}terminat",
        ],
        "risk_level": "Critical",
        "clause_type": "Termination",
        "simplified_english": "Either party can end this contract immediately or with very short notice, with no obligation to provide a reason. This heavily favors the party who wishes to exit.",
        "what_this_means": "This grants one or both parties the unilateral power to terminate the agreement, potentially leaving the other party with no recourse, compensation, or wind-down period.",
        "why_risky": "Unilateral termination without cause or notice can result in sudden loss of business, incomplete deliverables, and no legal recourse for the impacted party. It is a hallmark of one-sided agreements.",
        "consequences": "Abrupt contract termination, potential loss of income, sunk costs with no recovery, and possible difficulty pursuing damages without a specific breach.",
        "red_flags": "No cure period, no reason for termination required, asymmetric termination rights favoring one party.",
        "recommendations": "Negotiate for a minimum notice period (30-90 days), mutual termination rights, and a termination-for-cause standard with defined cure periods. Include provisions for work-in-progress payment.",
        "base_score": 88,
    },
    {
        "patterns": [
            r"liquidat[ed|es|ing]\s+damages",
            r"penalty.{0,30}(breach|default|failure)",
            r"punitive\s+(damages|penalty)",
        ],
        "risk_level": "Critical",
        "clause_type": "Penalty",
        "simplified_english": "The contract sets predefined monetary penalties for specific failures, which may be excessive and difficult to challenge in court.",
        "what_this_means": "Pre-agreed damages apply if you breach specific terms, regardless of the actual loss suffered by the other party.",
        "why_risky": "Liquidated damages clauses can impose penalties that far exceed the actual harm caused. Indian courts have historically upheld these if they are a genuine pre-estimate of loss, making disputes costly.",
        "consequences": "Heavy financial penalties even for minor or technical breaches, potentially disproportionate to the value of the contract.",
        "red_flags": "No upper cap on penalties, penalties apply to minor or non-material breaches, pre-set amounts seem disproportionate.",
        "recommendations": "Seek to cap liquidated damages at a percentage of contract value (e.g., 10-15%). Ensure penalties only apply to material breaches. Request a grace/cure period before penalties trigger.",
        "base_score": 85,
    },
    {
        "patterns": [
            r"sole\s+(discretion|decision|authority).{0,60}(amend|modify|terminate|change)",
            r"(amend|modify|change).{0,40}without.{0,40}(consent|notice|approval)",
            r"unilaterally.{0,60}(amend|modify|update|change)",
        ],
        "risk_level": "Critical",
        "clause_type": "Unilateral Amendment",
        "simplified_english": "One party has the power to change the terms of this agreement without your agreement or prior notice.",
        "what_this_means": "The other party can modify obligations, pricing, deliverables, or terms at will with no obligation to inform or get consent from you.",
        "why_risky": "This essentially renders the contract terms unenforceable as a protection for you — the other party can shift goalposts at any time.",
        "consequences": "Your rights and entitlements under the contract may be changed unilaterally, significantly impacting your position with no legal recourse.",
        "red_flags": "No requirement for mutual consent, no notice period, 'sole discretion' language.",
        "recommendations": "Insist on a mutual amendment clause requiring written consent from both parties for any changes. If any unilateral rights are given, demand prior written notice and a right to exit.",
        "base_score": 82,
    },
    # ── HIGH patterns ──────────────────────────────────────────────
    {
        "patterns": [
            r"indemnif[y|ies|ied|ication].{0,80}(any|all|indirect|consequential|third.party)",
            r"hold\s+harmless.{0,60}(any|all|indirect|third.party)",
            r"client.{0,30}shall.{0,30}indemnif",
        ],
        "risk_level": "High",
        "clause_type": "Indemnification",
        "simplified_english": "You may be financially responsible for a very wide range of losses, including damages from lawsuits involving third parties, even if they are only indirectly related to your actions.",
        "what_this_means": "Indemnification requires you to protect the other party from financial losses. Broad indemnification clauses can extend to indirect and consequential damages, third-party claims, and losses beyond your direct control.",
        "why_risky": "Without a cap on liability, the financial exposure can be significant and unpredictable, potentially exceeding the total contract value.",
        "consequences": "Potential liability for significant third-party claims, legal defense costs, and indirect damages not directly caused by you.",
        "red_flags": "Uncapped indemnification, inclusion of indirect and consequential damages, third-party liability with no carve-outs.",
        "recommendations": "Negotiate a mutual indemnification clause. Cap total indemnification liability at the contract value. Exclude indirect, consequential, and incidental damages. Add carve-outs for third-party IP infringement caused by the other party.",
        "base_score": 72,
    },
    {
        "patterns": [
            r"not\s+be\s+(held\s+)?liable",
            r"no\s+liability.{0,40}(any|indirect|consequential|loss|damage)",
            r"limitation\s+of\s+liability",
            r"shall\s+not\s+be\s+responsible.{0,60}loss",
        ],
        "risk_level": "High",
        "clause_type": "Liability Limitation",
        "simplified_english": "The other party has limited or excluded its financial responsibility for losses, damages, or failures — but your liability may remain uncapped.",
        "what_this_means": "While the other party protects itself from financial consequences, the agreement may not offer you reciprocal protection, creating an asymmetric liability structure.",
        "why_risky": "If the limitation only protects one party, you bear all financial risk from failures, even those caused by the other side.",
        "consequences": "You may not be able to recover losses caused by the other party, while remaining exposed to unlimited claims from their side.",
        "red_flags": "One-sided liability limitations, no corresponding cap protecting you, exclusion of lost profits or consequential damages for one party only.",
        "recommendations": "Ensure liability limitations are mutual. Push for a minimum guaranteed liability of at least the contract value for material breaches. Preserve your right to claim direct damages.",
        "base_score": 65,
    },
    {
        "patterns": [
            r"(non.compete|non.solicitation|restraint\s+of\s+trade)",
            r"shall\s+not.{0,60}(compete|solicit|approach|engage).{0,60}(employee|client|customer)",
        ],
        "risk_level": "High",
        "clause_type": "Restrictive Covenant",
        "simplified_english": "You are restricted from competing, offering similar services, or hiring from the other party's pool, which can significantly limit your future business opportunities.",
        "what_this_means": "These restrictions apply during and after the contract period, potentially limiting where you can work, who you can hire, and whom you can sell to.",
        "why_risky": "Overly broad non-compete and non-solicitation clauses can cripple your business operations long after the contract ends. Indian courts do not always enforce these, but the litigation cost is high.",
        "consequences": "Legal action for alleged violations, injunctions stopping business activities, and costly litigation to challenge the clauses.",
        "red_flags": "Broad geographic scope, long duration (> 1 year), wide definition of 'competing' that covers natural business growth.",
        "recommendations": "Negotiate narrowly scoped restrictions with clear definitions, a maximum 12-month duration, and geographic limits. Push for a carve-out for existing clients and employees you bring to the engagement.",
        "base_score": 68,
    },
    {
        "patterns": [
            r"intellectua[l]?\s+property.{0,60}(assign|transfer|vest|belong)\s+to\s+(client|company|employer)",
            r"all\s+(ip|intellectual\s+property|inventions|work\s+product).{0,60}(owned|assigned\s+to|vest)",
            r"work\s+for\s+hire",
        ],
        "risk_level": "High",
        "clause_type": "IP Assignment",
        "simplified_english": "All intellectual property you create during this engagement is automatically assigned to the other party, including pre-existing tools or methods you may use.",
        "what_this_means": "Any innovation, code, writing, design, or invention created in the course of this contract is owned by the other party from the moment of creation.",
        "why_risky": "Broad IP assignment clauses can strip you of ownership over pre-existing work, general methodologies, and background IP critical to your business.",
        "consequences": "Loss of ownership over valuable IP, inability to use your own work in future projects, and potential claims if you reuse similar material.",
        "red_flags": "No background IP carve-out, work-for-hire language, assignment of pre-existing proprietary tools or frameworks.",
        "recommendations": "Negotiate an explicit carve-out for background IP. Ensure only foreground IP (created specifically for this project) is assigned. Retain a license to use your own pre-existing tools.",
        "base_score": 70,
    },
    # ── MEDIUM patterns ────────────────────────────────────────────
    {
        "patterns": [
            r"payment.{0,60}(within|due\s+in|net)\s+\d+\s+days",
            r"invoice.{0,60}(days|due|payable)",
            r"late\s+payment.{0,40}(interest|fee|penalty)",
        ],
        "risk_level": "Medium",
        "clause_type": "Payment Terms",
        "simplified_english": "This clause defines when and how payments are made. Delays in payment or unclear invoice terms can create cash-flow and dispute risks.",
        "what_this_means": "Sets the payment schedule, invoice processing timelines, and consequences for late payment. Extended net payment terms (e.g., Net-90) can create cash flow challenges.",
        "why_risky": "Long payment cycles or unclear dispute resolution for invoices can severely impact your working capital and operational capacity.",
        "consequences": "Delayed cash receipts, potential disputes over invoice acceptance, and insufficient remedy for continued late payment.",
        "red_flags": "Net-60 or Net-90 payment terms, no interest on late payments, unilateral right to dispute invoices without defined timelines.",
        "recommendations": "Negotiate for Net-30 payment terms. Include an automatic interest clause (e.g., 18% p.a.) for delayed payments. Define a clear invoice dispute resolution window.",
        "base_score": 45,
    },
    {
        "patterns": [
            r"confidential(ity|information|data).{0,80}(shall\s+not|must\s+not|not\s+disclose|protect)",
            r"non.disclosure",
            r"proprietary\s+information.{0,40}(secret|confidential|protect)",
        ],
        "risk_level": "Medium",
        "clause_type": "Confidentiality",
        "simplified_english": "You are legally bound to keep sensitive information private, which is standard but carries responsibility, especially for digital data.",
        "what_this_means": "This imposes a legal obligation to safeguard the other party's business information, trade secrets, and data, often without a defined end date.",
        "why_risky": "Perpetual confidentiality obligations are difficult to comply with indefinitely. Accidental disclosure can lead to injunctions or damage claims.",
        "consequences": "Legal action for breach of confidentiality, injunctions, and potentially significant damage awards for data leaks.",
        "red_flags": "Perpetual confidentiality duration, overly broad definition of 'confidential information', no exceptions for publicly known information.",
        "recommendations": "Insist on a 3-5 year confidentiality period. Define 'confidential information' precisely with standard carve-outs (publicly known, independently developed). Request mutual confidentiality obligations.",
        "base_score": 42,
    },
    {
        "patterns": [
            r"governing\s+law.{0,60}(India|Indian|jurisdiction)",
            r"disputes?.{0,60}arbitration",
            r"arbitration.{0,60}(new\s+delhi|mumbai|bangalore|chennai)",
            r"courts?.{0,40}(exclusive|jurisdiction)",
        ],
        "risk_level": "Medium",
        "clause_type": "Governing Law & Disputes",
        "simplified_english": "This defines where and how disputes are resolved. Exclusive jurisdiction clauses and mandatory arbitration in distant cities can make seeking remedies expensive.",
        "what_this_means": "Establishes the legal framework and forum for resolving any future contract disputes. The choice of jurisdiction significantly impacts the practicality and cost of seeking remedies.",
        "why_risky": "Mandatory arbitration in a distant city, or exclusive jurisdiction in a court unfavorable to your position, can deter you from pursuing legitimate claims.",
        "consequences": "High litigation costs for resolving disputes in inconvenient forums, potential bias toward the party who drafted the agreement.",
        "red_flags": "Single-sided jurisdiction clause, mandatory arbitration without mutually agreed rules, no interim relief provisions.",
        "recommendations": "Negotiate for arbitration under SIAC, ICC, or domestic rules (LCIA India) with neutral seat. Include provisions for emergency interim relief. Ensure the governing law is fair to both parties.",
        "base_score": 40,
    },
    # ── LOW patterns ───────────────────────────────────────────────
    {
        "patterns": [
            r"this\s+agreement.{0,30}(entered|made|executed).{0,40}between",
            r"hereinafter\s+(referred\s+to\s+as|called)",
            r"whereas.{0,60}(party|parties|client|vendor|company)",
            r"recital",
        ],
        "risk_level": "Low",
        "clause_type": "Preamble",
        "simplified_english": "This introductory section identifies the contracting parties. It is standard and carries no legal risk on its own.",
        "what_this_means": "Sets the legal context, identifies parties, and provides background on why the agreement is being executed.",
        "why_risky": "Generally low risk, but ensure names, entity types (e.g., Pvt. Ltd.), and addresses are accurate to avoid enforceability gaps.",
        "consequences": "Incorrect party identification can create enforceability challenges.",
        "red_flags": "Misspelled entity names, incorrect CIN numbers, wrong address details.",
        "recommendations": "Verify all company names, registration numbers, and addresses match official records from the Ministry of Corporate Affairs.",
        "base_score": 12,
    },
    {
        "patterns": [
            r"force\s+majeure",
            r"act\s+of\s+god",
            r"(epidemic|pandemic|war|strike|flood|earthquake).{0,60}(excuse|exempt|suspend|relieve)",
        ],
        "risk_level": "Low",
        "clause_type": "Force Majeure",
        "simplified_english": "This protects both parties if unforeseeable catastrophic events (like natural disasters or pandemics) make it impossible to fulfill contract obligations.",
        "what_this_means": "A standard protective clause that suspends obligations during extraordinary events beyond either party's control.",
        "why_risky": "Broadly worded force majeure clauses can be misused to excuse non-performance for events that were foreseeable or preventable.",
        "consequences": "One party may declare force majeure without genuine grounds, potentially delaying your project or payment indefinitely.",
        "red_flags": "No defined duration limit, no obligation to mitigate, no right to terminate if force majeure persists too long.",
        "recommendations": "Include a maximum force majeure period (e.g., 60-90 days) after which either party can terminate. Add a mitigation obligation and notification requirement within 48 hours.",
        "base_score": 18,
    },
    {
        "patterns": [
            r"entire\s+agreement",
            r"supersede[s]?\s+all\s+(prior|previous|earlier)",
            r"no\s+(oral|verbal|prior)\s+agreement",
        ],
        "risk_level": "Low",
        "clause_type": "Entire Agreement",
        "simplified_english": "This document is the complete and final agreement, replacing all previous verbal or written understandings.",
        "what_this_means": "Ensures that only what's written in this contract is legally enforceable, providing certainty and clarity.",
        "why_risky": "Any oral promises or side agreements made before signing become void. Ensure all agreed terms are written into this contract before signing.",
        "consequences": "Previous representations or discussions that were part of your understanding are unenforceable unless explicitly written here.",
        "red_flags": "Any agreed term not included in the written contract.",
        "recommendations": "Before signing, list all terms discussed verbally and confirm they are reflected in the contract. Have a legal review to ensure no oral commitment is missing.",
        "base_score": 10,
    },
]


def _rule_based_analysis(clauses_text: list) -> list:
    """
    A sophisticated rule-based legal NLP engine that analyzes clauses
    using keyword pattern matching to produce rich, accurate explanations.
    This is the primary engine when ML models are not available.
    """
    results = []
    
    for text in clauses_text:
        text_lower = text.lower()
        best_match = None
        highest_score = 0

        for pattern_set in CLAUSE_PATTERNS:
            for pattern in pattern_set["patterns"]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    score = pattern_set["base_score"]
                    if score > highest_score:
                        highest_score = score
                        best_match = pattern_set
                    break

        if best_match:
            # Add minor variance based on text length as a heuristic
            score = best_match["base_score"] + min(5, len(text) // 50)
            score = min(score, 99)
            results.append({
                "original_text": text,
                "clause_type": best_match["clause_type"],
                "risk_score": score,
                "risk_level": best_match["risk_level"],
                "simplified_english": best_match["simplified_english"],
                "what_this_means": best_match["what_this_means"],
                "why_risky": best_match["why_risky"],
                "consequences": best_match["consequences"],
                "red_flags": best_match["red_flags"],
                "recommendations": best_match["recommendations"],
            })
        else:
            # Uncategorized clause — give a generic low-risk assessment
            results.append({
                "original_text": text,
                "clause_type": "General Clause",
                "risk_score": 15,
                "risk_level": "Low",
                "simplified_english": "This section contains standard contractual language that does not match any high-risk legal patterns.",
                "what_this_means": "This clause defines general rights, obligations, or procedures that are standard in commercial agreements.",
                "why_risky": "No significant risk pattern detected. This clause appears to be routine and standard.",
                "consequences": "Low risk if standard terms are maintained and both parties act in good faith.",
                "red_flags": "None detected by automated analysis. Manual review still recommended.",
                "recommendations": "Review manually to ensure the clause aligns with your expectations and does not introduce any specific operational risks unique to your situation.",
            })
    return results


def _cpu_load(filepath):
    """
    Loads a joblib/pickle file that was saved with CUDA tensors,
    safely remapping all tensors to CPU. This is needed when the
    model was trained on Colab GPU but loaded on a CPU-only machine.
    """
    import io as _io
    import pickle as _pickle

    class _CPUUnpickler(_pickle.Unpickler):
        def find_class(self, module, name):
            if module == 'torch.storage' and name == '_load_from_bytes':
                return lambda b: torch.load(_io.BytesIO(b), map_location='cpu', weights_only=False)
            return super().find_class(module, name)

    try:
        with open(filepath, 'rb') as f:
            return _CPUUnpickler(f).load()
    except Exception:
        # Last-resort: standard joblib (works if file was already CPU)
        return joblib.load(filepath)


class LexGuardInference:
    def __init__(self):
        # Prefer the newly trained model (1) first, fall back to the older cpu variant
        new_model_path = os.path.join(ML_MODELS_DIR, 'english_risk_model (1).pkl')
        old_model_path = os.path.join(ML_MODELS_DIR, 'english_risk_model_cpu.pkl')
        self.english_model_path = new_model_path if os.path.exists(new_model_path) else old_model_path
        self.english_pipeline = None
        self._load_models()

    def _load_models(self):
        """
        Load the user-trained PyTorch model bundle (contains model_state, config, tokenizer)
        """
        if not ML_AVAILABLE or not os.path.exists(self.english_model_path):
            logger.warning(f"ML model file not found at {self.english_model_path}. Using rule-based engine.")
            self.english_model = None
            return

        try:
            logger.info(f"Loading custom risk model from {self.english_model_path} ...")
            bundle = _cpu_load(self.english_model_path)
            
            self.english_tokenizer = bundle['tokenizer']
            self.english_model = AutoModelForSequenceClassification.from_config(bundle['config'])
            
            # The user's colab model was trained and saved with state_dict
            self.english_model.load_state_dict(bundle['model_state'], strict=False)
            self.english_model.eval()
            
            # If a GPU is available, move it
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.english_model.to(self.device)
            
            logger.info("Successfully loaded PyTorch risk model!")
        except Exception as e:
            logger.error(f"Failed to load PyTorch risk model ({e}). Using rule-based engine.")
            self.english_model = None


    def analyze_document(self, text: str, language: str = 'English') -> list:
        """
        Main entry point. Segments the document into clauses and runs analysis.
        Prefers ML inference, falls back to the rule-based engine.
        """
        # Smart clause segmentation: split on newlines so every physical paragraph
        # from the PDF is extracted as its own distinct clause, preventing bundling.
        raw_splits = [p for p in text.split('\n') if p.strip()]
        clauses = [s.strip() for s in raw_splits if len(s.strip()) > 25]

        if not clauses:
            clauses = [text]

        if ML_AVAILABLE and getattr(self, 'english_model', None) is not None:
            return self._real_inference(clauses)

        return _rule_based_analysis(clauses)

    def _real_inference(self, clauses_text: list) -> list:
        """Perform real inference using the loaded PyTorch model."""
        analyzed_clauses = []

        LEVEL_MAP = {
            0: {"level": "Low",      "score_range": (10, 25)},
            1: {"level": "Medium",   "score_range": (26, 50)},
            2: {"level": "High",     "score_range": (51, 75)},
            3: {"level": "Critical", "score_range": (76, 99)},
        }

        for text in clauses_text:
            try:
                # Tokenize clause
                inputs = self.english_tokenizer(
                    text, return_tensors="pt", truncation=True, max_length=512
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.english_model(**inputs)
                    logits = outputs.logits
                    probs = torch.nn.functional.softmax(logits, dim=-1)[0]
                    label_idx = torch.argmax(probs).item()
                    confidence = probs[label_idx].item()

                lvl_info = LEVEL_MAP.get(label_idx, LEVEL_MAP[0])
                level = lvl_info["level"]
                lo, hi = lvl_info["score_range"]
                risk_score = int(lo + (confidence * (hi - lo)))

                # Enrich with rule-based context knowledge for explanations
                rule_results = _rule_based_analysis([text])
                rule_data = rule_results[0] if rule_results else {}

                clause_type = rule_data.get("clause_type", "General Clause")
                if clause_type == "General Clause" and label_idx > 0:
                    clause_type = f"Non-Standard Clause"

                what = rule_data.get("what_this_means", "")
                risky = rule_data.get("why_risky", "")
                flags = rule_data.get("red_flags", "")
                recs = rule_data.get("recommendations", "")

                if "General Clause" in clause_type or "Non-Standard" in clause_type:
                    if level == "Critical":
                        what = "This clause imposes massive one-sided obligations, severe penalties, or strips you of fundamental legal rights."
                        risky = "It exposes you to uncapped liability, immediate termination without cause, or severe data/privacy loss."
                        flags = "Uncapped liability; Unilateral termination; Broad indemnification."
                        recs = "Do NOT sign this before consulting a lawyer. Push for mutual protections."
                    elif level == "High":
                        what = "This clause contains strict conditions that heavily favor the drafting party."
                        risky = "It may lead to significant financial loss or restrict your operational freedom."
                        flags = "Strict timelines; High penalties; One-sided dispute resolution."
                        recs = "Push back on these terms to ensure they are mutual or capped."
                    elif level == "Medium":
                        what = "This clause deviates slightly from standard balanced legal practices."
                        risky = "While not a dealbreaker, it introduces minor unfavorable conditions."
                        flags = "Vague language; Slight imbalance in rights."
                        recs = "Consider asking for clarification or minor amendments."
                    else:
                        what = "This clause defines general rights, obligations, or procedures that are standard in commercial agreements."
                        risky = "No significant risk pattern detected. This clause appears to be routine and standard."
                        flags = "None identified."
                        recs = "Review generally to ensure it aligns with your commercial understanding."

                analyzed_clauses.append({
                    "original_text": text,
                    "clause_type": clause_type,
                    "risk_score": risk_score,
                    "risk_level": level,
                    "simplified_english": rule_data.get("simplified_english", ""),
                    "what_this_means": what,
                    "why_risky": risky,
                    "consequences": rule_data.get("consequences", ""),
                    "red_flags": flags,
                    "recommendations": recs,
                })

            except Exception as e:
                logger.error(f"ML prediction error for clause, falling back to rule engine: {e}")
                analyzed_clauses.extend(_rule_based_analysis([text]))

        return analyzed_clauses
