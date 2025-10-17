"""
NLP4Health MEDICAL RAG SYSTEM
========================
A production-grade multilingual medical Q&A system with:
- Confidence-based retrieval
- Hallucination prevention
- Safety guardrails
- Comprehensive logging
- Answer validation
"""
# ============================================================================
# SECTION 1: IMPORTS AND DEPENDENCIES
# ============================================================================
from google.colab import drive
drive.mount('/content/drive')
# Install required packages
!pip install -q sentence-transformers transformers langdetect accelerate faiss-cpu rouge-score bert-score
import os
import json
import datetime
import re
import logging
import hashlib
import numpy as np
import torch
import faiss
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, CrossEncoder, util
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langdetect import detect, LangDetectException
from sklearn.metrics.pairwise import cosine_similarity
print("✅ All imports successful")


# ============================================================================
# SECTION 2 Configuration of model
# ============================================================================
@dataclass
class Config:
    """Central configuration for the RAG system"""
    # Paths
    BASE_DIR: str = "/content/drive/MyDrive/NLP_A14_Health/combined_qns"
    FAISS_DIR: str = "/content/faiss_indices/"
    RESULTS_DIR: str = "/content/drive/MyDrive/NLP_A14_Health/results/"
    LOG_DIR: str = "/content/drive/MyDrive/NLP_A14_Health/logs/"
    # Models
    # EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    # GENERATION_MODEL: str = "CLARA-MeD/mt5-small"
    # RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    # Models
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    GENERATION_MODEL: str = ""
    RERANKER_MODEL: str = ""
    # Retrieval parameters
    TOP_K_RETRIEVAL: int = 10
    TOP_K_RERANK: int = 3
    RELEVANCE_THRESHOLD: float = 0.45  # Minimum similarity score
    # Generation parameters
    MAX_NEW_TOKENS: int = 256
    TEMPERATURE: float = 0.7
    NUM_BEAMS: int = 5
    # Cache settings
    CACHE_SIZE: int = 1000
    # Safety thresholds
    MIN_CONFIDENCE: float = 0.5
    HALLUCINATION_THRESHOLD: float = 0.3
# Initialize configuration
config = Config()
# Make sure directories exist
for directory in [config.FAISS_DIR, config.RESULTS_DIR, config.LOG_DIR]:
    os.makedirs(directory, exist_ok=True)
print("✅ Configuration initialized")


# ============================================================================
# SECTION 3: LOGGING SYSTEM
# ============================================================================
class RAGLogger:
    """Comprehensive logging system for monitoring and debugging"""
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.logger = logging.getLogger("MedicalRAG")
        self.logger.setLevel(logging.INFO)
        # File handler with date
        log_file = os.path.join(
            log_dir,
            f"rag_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        )
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)
        self.logger.info("=" * 80)
        self.logger.info("RAG SYSTEM INITIALIZED")
        self.logger.info("=" * 80)
    def log_query(self, query: str, lang: str, status: str, **kwargs):
        """Log incoming query with metadata"""
        self.logger.info(
            f"QUERY | Lang: {lang} | Status: {status} | "
            f"Query: {query[:100]}... | {kwargs}"
        )
    def log_retrieval(self, lang: str, num_retrieved: int, scores: List[float]):
        """Log retrieval results"""
        avg_score = np.mean(scores) if scores else 0
        self.logger.info(
            f"RETRIEVAL | Lang: {lang} | Retrieved: {num_retrieved} | "
            f"Avg Score: {avg_score:.3f}"
        )
    def log_generation(self, answer_length: int, confidence: float):
        """Log answer generation"""
        self.logger.info(
            f"GENERATION | Length: {answer_length} chars | "
            f"Confidence: {confidence:.3f}"
        )
    def log_safety(self, query: str, flag_type: str, reason: str):
        """Log safety flags"""
        self.logger.warning(
            f"SAFETY FLAG | Type: {flag_type} | Reason: {reason} | "
            f"Query: {query[:50]}..."
        )
    def log_error(self, error_type: str, details: str, query: str = ""):
        """Log errors"""
        self.logger.error(
            f"ERROR | Type: {error_type} | Details: {details} | "
            f"Query: {query[:50] if query else 'N/A'}"
        )
    def log_cache(self, action: str, query: str):
        """Log cache operations"""
        self.logger.debug(f"CACHE | Action: {action} | Query: {query[:50]}...")
# Initialize logger
logger = RAGLogger(config.LOG_DIR)
print("✅ Logging system initialized")

# ============================================================================
# SECTION 4: SAFETY AND VALIDATION MODULES
# ============================================================================
class SafetyGuard:
    """Safety checks for medical queries and answers"""
    # Harmful query patterns
    HARMFUL_PATTERNS = [
        r"how to (kill|harm|suicide|die)",
        r"(overdose|lethal dose|fatal amount)",
        r"without (doctor|prescription|medical help)",
        r"(hide|fake|forge) (symptoms|test results|prescription)",
        r"(abortion|terminate pregnancy) at home",
        r"how to get high",
        r"self (harm|mutilate|injury)",
        r"poison someone",
    ]
    # Unsafe advice patterns
    UNSAFE_ADVICE = [
        "stop taking your medication",
        "ignore your doctor",
        "don't see a doctor",
        "avoid medical help",
        "perform surgery",
        "inject yourself",
        "increase dosage significantly",
    ]
    # Emergency keywords
    EMERGENCY_KEYWORDS = [
        "severe pain", "can't breathe", "chest pain", "unconscious",
        "heavy bleeding", "overdose", "poisoning", "stroke symptoms",
        "heart attack", "severe allergic reaction", "anaphylaxis"
    ]
    @staticmethod
    def is_safe_query(query: str) -> Tuple[bool, str]:
        """Check if query is safe to process"""
        query_lower = query.lower()
        # Check harmful patterns
        for pattern in SafetyGuard.HARMFUL_PATTERNS:
            if re.search(pattern, query_lower):
                return False, "HARMFUL_INTENT"
        return True, "SAFE"
    @staticmethod
    def is_emergency(query: str) -> bool:
        """Detect emergency situations"""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in SafetyGuard.EMERGENCY_KEYWORDS)
    @staticmethod
    def filter_dangerous_advice(answer: str, query: str) -> Tuple[str, bool]:
        """Filter out dangerous medical advice"""
        answer_lower = answer.lower()
        # Check for unsafe advice
        for phrase in SafetyGuard.UNSAFE_ADVICE:
            if phrase in answer_lower:
                return (
                    "I cannot provide this advice. Please consult your healthcare provider immediately.",
                    True
                )
        # Check if emergency query got non-urgent answer
        if SafetyGuard.is_emergency(query) and "emergency" not in answer_lower and "immediately" not in answer_lower:
            return (
                "🚨 EMERGENCY: This appears to be a medical emergency. "
                "Please call emergency services (ambulance) immediately or go to the nearest emergency room. "
                "Do not wait for medical information.",
                True
            )
        return answer, False
    @staticmethod
    def get_emergency_response() -> str:
        """Standard emergency response"""
        return (
            "🚨 MEDICAL EMERGENCY DETECTED\n\n"
            "This appears to be a medical emergency. Please:\n"
            "1. Call emergency services (ambulance) IMMEDIATELY\n"
            "2. Go to the nearest emergency room\n"
            "3. Do NOT wait for online medical advice\n\n"
            "If someone is unconscious or not breathing, start CPR if trained."
        )
class DisclaimerManager:
    """Manages medical disclaimers for different query types"""
    DISCLAIMERS = {
        "medication": (
            "\n\n⚠️ MEDICATION INFORMATION: This information is for educational purposes only. "
            "Always consult a doctor or pharmacist before starting, stopping, or changing medications. "
            "Never adjust medication dosages without professional guidance."
        ),
        "diagnosis": (
            "\n\n⚠️ DIAGNOSTIC INFORMATION: This is not a medical diagnosis. "
            "Only a qualified healthcare professional can diagnose medical conditions. "
            "If you're experiencing symptoms, please consult a doctor."
        ),
        "emergency": (
            "\n\n🚨 EMERGENCY: If this is an emergency, call emergency services immediately."
        ),
        "treatment": (
            "\n\n⚠️ TREATMENT INFORMATION: Treatment plans should be personalized by healthcare professionals. "
            "Do not self-treat based on general information. Consult your doctor."
        ),
        "general": (
            "\n\nℹ️ MEDICAL DISCLAIMER: This information is based on general medical knowledge "
            "and should not replace professional medical advice. Always consult a qualified "
            "healthcare provider for personalized medical guidance."
        )
    }
    @staticmethod
    def detect_category(query: str) -> str:
        """Detect query category for appropriate disclaimer"""
        query_lower = query.lower()
        if re.search(r"\b(medicine|drug|pill|prescription|medication|tablet)\b", query_lower):
            return "medication"
        if re.search(r"\b(diagnose|do i have|symptoms of|is it|could it be)\b", query_lower):
            return "diagnosis"
        if SafetyGuard.is_emergency(query):
            return "emergency"
        if re.search(r"\b(treat|treatment|cure|therapy|procedure)\b", query_lower):
            return "treatment"
        return "general"
    @staticmethod
    def add_disclaimer(answer: str, query: str) -> str:
        """Add appropriate disclaimer to answer"""
        category = DisclaimerManager.detect_category(query)
        disclaimer = DisclaimerManager.DISCLAIMERS.get(category, DisclaimerManager.DISCLAIMERS["general"])
        return answer + disclaimer
print("✅ Safety modules initialized")

# ============================================================================
# SECTION 5: LANGUAGE DETECTION
# ============================================================================
class LanguageDetector:
    """Automatic language detection with fallback"""
    # Language code mapping
    LANG_MAP = {
        'en': 'English',
        'hi': 'Hindi',
        'bn': 'Bengali',
        'te': 'Telugu',
        'ta': 'Tamil',
        'mr': 'Marathi',
        'gu': 'Gujarati',
        'kn': 'Kannada',
        'ml': 'Malayalam',
        'pa': 'Punjabi',
        'as': 'Assamese',
        'or': 'Odia',
        'ur': 'Urdu',
    }
    @staticmethod
    def detect(query: str, available_languages: List[str]) -> str:
        """Detect language with fallback to English"""
        try:
            lang_code = detect(query)
            lang_name = LanguageDetector.LANG_MAP.get(lang_code, 'English')
            # Check if detected language is available
            if lang_name in available_languages:
                logger.log_query(query, lang_name, "LANG_DETECTED")
                return lang_name
            else:
                logger.log_query(query, "English", "LANG_FALLBACK")
                return 'English'
        except LangDetectException:
            logger.log_error("LANG_DETECTION", "Failed to detect language", query)
            return 'English'
print("✅ Language detector initialized")

# ============================================================================
# SECTION 6: CACHING SYSTEM
# ============================================================================
class QueryCache:
    """LRU cache for query results"""
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size
        logger.logger.info(f"Cache initialized with max_size={max_size}")
    def _get_cache_key(self, query: str, lang: str) -> str:
        """Generate cache key from query and language"""
        content = f"{query.lower().strip()}_{lang}".encode('utf-8')
        return hashlib.md5(content).hexdigest()
    def get(self, query: str, lang: str) -> Optional[Dict]:
        """Retrieve cached result"""
        key = self._get_cache_key(query, lang)
        if key in self.cache:
            # Update access order (LRU)
            self.access_order.remove(key)
            self.access_order.append(key)
            logger.log_cache("HIT", query)
            return self.cache[key]
        logger.log_cache("MISS", query)
        return None
    def set(self, query: str, lang: str, result: Dict):
        """Store result in cache"""
        key = self._get_cache_key(query, lang)
        # Evict oldest if cache is full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        self.cache[key] = result
        if key not in self.access_order:
            self.access_order.append(key)
        logger.log_cache("SET", query)
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.access_order.clear()
        logger.logger.info("Cache cleared")
# Initialize cache
query_cache = QueryCache(max_size=config.CACHE_SIZE)
print("✅ Cache system initialized")


#Importing already precoumputed FAISS All languages index
import shutil
import os
# Source and destination directories
source_dir = '/content/drive/MyDrive/faiss_backup'
dest_dir = '/content/faiss_indices'
# Create destination folder if it doesn't exist
os.makedirs(dest_dir, exist_ok=True)
# Copy all files and folders recursively
shutil.copytree(source_dir, dest_dir, dirs_exist_ok=True)
print(f"Files copied from {source_dir} to {dest_dir}")
# List all files recursively
faiss_dir = '/content/faiss_indices'
print(f"Listing all files in {faiss_dir}:\n")
for root, dirs, files in os.walk(faiss_dir):
    for file in files:
        file_path = os.path.join(root, file)
        print(file_path)


# =============================================================================
# MULTILINGUAL FAISS RETRIEVER
# =============================================================================
import os
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from langdetect import detect
from typing import List, Tuple
# --- CONFIG ---
FAISS_DIR = "/content/faiss_indices"
TOP_K_RETRIEVAL = 5
RELEVANCE_THRESHOLD = 0.3
device = "cuda" if torch.cuda.is_available() else "cpu"
# --- EMBEDDER ---
embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device=device
)
# --- AVAILABLE LANGUAGES ---
# map language code to FAISS index and map file
LANGUAGE_INDEX_MAP = {
    "en": ("English_index.faiss", "English_map.npy"),
    "hi": ("Hindi_index.faiss", "Hindi_map.npy"),
    "bn": ("Bangla_index.faiss", "Bangla_map.npy"),
    "ta": ("Tamil_index.faiss", "Tamil_map.npy"),
    "te": ("Telugu_index.faiss", "Telugu_map.npy"),
    "kn": ("Kannada_index.faiss", "Kannada_map.npy"),
    "gu": ("Gujarati_index.faiss", "Gujarati_map.npy"),
    "mr": ("Marathi_index.faiss", "Marathi_map.npy"),
    "as": ("Assamese_index.faiss", "Assamese_map.npy"),
    "doi": ("Dogri_index.faiss", "Dogri_map.npy"),
}
# --- RETRIEVER CLASS ---
class MultilingualRetriever:
    def __init__(self, top_k=TOP_K_RETRIEVAL, threshold=RELEVANCE_THRESHOLD):
        self.top_k = top_k
        self.threshold = threshold
        self.index_cache = {}
        self.text_cache = {}
    def _load_index_and_texts(self, lang: str):
        """Load FAISS index and text map for a given language"""
        if lang not in LANGUAGE_INDEX_MAP:
            raise ValueError(f"No index available for language '{lang}'")
        if lang in self.index_cache:
            return self.index_cache[lang], self.text_cache[lang]
        index_file, map_file = LANGUAGE_INDEX_MAP[lang]
        index_path = os.path.join(FAISS_DIR, index_file)
        map_path = os.path.join(FAISS_DIR, map_file)
        if not os.path.exists(index_path) or not os.path.exists(map_path):
            raise FileNotFoundError(f"Files missing for language '{lang}'")
        # Load FAISS index
        if index_path.endswith(".faiss"):
            index = faiss.read_index(index_path)
        else:
            index = faiss.read_index(index_path)
        # Load texts
        texts = np.load(map_path, allow_pickle=True)
        self.index_cache[lang] = index
        self.text_cache[lang] = texts
        return index, texts
    def retrieve(self, query: str, top_k=None) -> Tuple[List[dict], List[float], str, str]:
        # Auto-detect language
        try:
            lang = detect(query)
        except:
            lang = "en"  # fallback
        if top_k is None:
            top_k = self.top_k
        index, texts = self._load_index_and_texts(lang)
        # Embed query
        q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        D, I = index.search(q_emb, top_k)
        D, I = D[0], I[0]
        # Collect results above threshold
        contexts, scores = [], []
        for idx, score in zip(I, D):
            if score >= self.threshold:
                ctx = texts[idx]
                if isinstance(ctx, str):
                    ctx = {"question": "", "answer": ctx}
                contexts.append(ctx)
                scores.append(float(score))
        # Confidence
        if len(contexts) == 0:
            status = "LOW_CONFIDENCE"
        elif len(contexts) >= 3 and np.mean(scores) >= 0.6:
            status = "HIGH_CONFIDENCE"
        else:
            status = "MEDIUM_CONFIDENCE"
        return contexts, scores, status, lang
# --- INITIALIZE ---
retriever = MultilingualRetriever()
print("✅ Multilingual retriever initialized")


# =============================================================================
# SECTION 9: ROBUST MULTILINGUAL ANSWER GENERATION SYSTEM
# =============================================================================
import re
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from typing import List, Dict, Tuple
class AnswerGenerator:
    """Robust multilingual answer generator using google/mt5-small and context-QA grounding."""
    def __init__(self, generation_model: str = "google/mt5-small", max_tokens: int = 256, num_beams: int = 4, temperature: float = 0.0):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.generation_model = generation_model
        self.max_tokens = max_tokens
        self.num_beams = num_beams
        self.temperature = temperature
        # Load mT5-small
        print(f"🔹 Loading generation model: {self.generation_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.generation_model)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.generation_model).to(self.device)
        self.model.config.decoder_start_token_id = self.tokenizer.pad_token_id
        print(f"✅ Generator loaded on {self.device}")
    def _build_prompt(self, query: str, contexts: List[Dict], lang: str) -> str:
        """Build structured prompt with Q&A context and strict safety instructions."""
        if not contexts:
            return query  # Fallback: just query
        context_text = "\n\n".join([
            f"[Source {i+1}]\nQuestion: {ctx.get('question', '')}\nAnswer: {ctx.get('answer', '')}"
            for i, ctx in enumerate(contexts)
        ])
        prompt = (
            f"You are a medical information assistant. Provide accurate information STRICTLY based on the provided context.\n\n"
            f"CRITICAL RULES:\n"
            f"1. Answer ONLY using information from the context below.\n"
            f"2. If context doesn't contain enough info, respond: \"I don't have sufficient information in my knowledge base to answer this question accurately.\"\n"
            f"3. NEVER invent or assume medical information.\n"
            f"4. Stay factual, clear, concise.\n"
            f"5. Respond in {lang} language.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            f"Answer (following all rules above):"
        )
        return prompt
    def generate(self, query: str, contexts: List[Dict], lang: str) -> Tuple[str, Dict]:
        """Generate a context-grounded answer."""
        if not contexts:
            return self._generate_no_context_response(), {}
        prompt = self._build_prompt(query, contexts, lang)
        # Tokenize prompt
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=1024
        ).to(self.device)
        # Generate output
        start_time = datetime.datetime.now()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                num_beams=self.num_beams,
                temperature=self.temperature,
                early_stopping=True,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        generation_time = (datetime.datetime.now() - start_time).total_seconds()
        # Decode and clean answer
        full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = full_output.split("Answer (following all rules above):")[-1].strip()
        answer = self._clean_answer(answer)
        metadata = {
            "generation_time": generation_time,
            "num_tokens": len(outputs[0]),
            "num_contexts": len(contexts)
        }
        return answer, metadata
    def _clean_answer(self, answer: str) -> str:
        """Clean artifacts and incomplete endings."""
        answer = re.sub(r"^(Answer|Response|A:)\s*:?\s*", "", answer, flags=re.IGNORECASE)
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        if len(sentences) > 1 and len(sentences[-1]) < 10:
            sentences = sentences[:-1]
        return '. '.join(sentences).strip() + '.'
    def _generate_no_context_response(self) -> str:
        """Fallback if no context is available."""
        return (
            "I don't have sufficient information in my knowledge base to answer this specific question accurately.\n\n"
            "Recommendations:\n"
            "1. Consult a qualified healthcare professional.\n"
            "2. Check reputable medical resources.\n"
            "3. Call medical helplines if needed.\n"
            "For emergencies, contact emergency services immediately."
        )
    def validate_answer(self, answer: str, contexts: List[Dict]) -> Tuple[bool, str, float]:
        """Heuristic validation for context grounding and minimal length."""
        if len(answer.split()) < 10:
            return False, "TOO_SHORT", 0.0
        refusal_patterns = [
            "i don't have", "cannot answer", "insufficient information",
            "not enough information", "unable to provide"
        ]
        if any(pat in answer.lower() for pat in refusal_patterns):
            return True, "VALID_REFUSAL", 0.5
        # Context overlap (basic keyword check)
        context_keywords = set()
        for ctx in contexts:
            words = re.findall(r'\w+', ctx.get('answer', '').lower())
            context_keywords.update([w for w in words if len(w) > 4])
        answer_keywords = set(re.findall(r'\w+', answer.lower()))
        answer_keywords = {w for w in answer_keywords if len(w) > 4}
        overlap = len(context_keywords & answer_keywords)
        overlap_ratio = overlap / max(len(answer_keywords), 1)
        if overlap < 3:
            return False, "NO_CONTEXT_OVERLAP", 0.2
        confidence = min(overlap_ratio * 1.5, 1.0)
        return True, "VALID", confidence
# =========================
# Example usage:
# generator = AnswerGenerator()
# answer, metadata = generator.generate(query="क्या खांसी का इलाज क्या है?", contexts=top_contexts, lang="hi")
# print(answer, metadata)
# =========================


# ============================================================================
# SECTION 10: CONFIDENCE SCORING
# ============================================================================
class ConfidenceScorer:
    """Calculate multi-factor confidence scores"""
    @staticmethod
    def calculate_confidence(
        retrieval_scores: List[float],
        answer: str,
        contexts: List[Dict],
        validation_confidence: float
    ) -> float:
        """
        Calculate overall confidence score
        Factors:
        1. Retrieval quality (40%)
        2. Answer length appropriateness (15%)
        3. Context overlap (25%)
        4. Validation confidence (20%)
        """
        if len(retrieval_scores) == 0:
            return 0.0
        # Factor 1: Retrieval quality
        avg_retrieval_score = np.mean(retrieval_scores)
        retrieval_factor = min(avg_retrieval_score / 0.8, 1.0)  # Normalize to 0.8 as max
        # Factor 2: Answer length (optimal range: 50-300 words)
        word_count = len(answer.split())
        if 50 <= word_count <= 300:
            length_factor = 1.0
        elif word_count < 50:
            length_factor = word_count / 50
        else:
            length_factor = max(0.5, 1.0 - (word_count - 300) / 500)
        # Factor 3: Context overlap (keyword-based)
        context_keywords = set()
        for ctx in contexts:
            words = re.findall(r'\w+', ctx['answer'].lower())
            context_keywords.update([w for w in words if len(w) > 4])
        answer_keywords = set(re.findall(r'\w+', answer.lower()))
        answer_keywords = {w for w in answer_keywords if len(w) > 4}
        if len(answer_keywords) > 0:
            overlap_ratio = len(context_keywords & answer_keywords) / len(answer_keywords)
        else:
            overlap_ratio = 0.0
        # Factor 4: Validation confidence
        validation_factor = validation_confidence
        # Weighted combination
        confidence = (
            0.40 * retrieval_factor +
            0.15 * length_factor +
            0.25 * overlap_ratio +
            0.20 * validation_factor
        )
        return round(confidence, 3)
    @staticmethod
    def interpret_confidence(confidence: float) -> str:
        """Interpret confidence level"""
        if confidence >= 0.7:
            return "HIGH"
        elif confidence >= 0.5:
            return "MEDIUM"
        else:
            return "LOW"
print("✅ Confidence scorer initialized")

# =============================================================================
# SECTION 11: MAIN RAG PIPELINE (UPDATED WITH ROBUST ANSWER GENERATOR)
# =============================================================================
import os
import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
# ------------------- INITIALIZE GENERATOR -------------------
generator = AnswerGenerator(generation_model="google/mt5-small")
class RobustMedicalRAG:
    """
    Main RAG pipeline with safety, validation, multilingual support, and robust answer generation.
    """
    def __init__(self):
        self.retriever = retriever              # Must be previously initialized
        self.generator = generator              # Our robust multilingual generator
        self.cache = query_cache                # Your caching system
        self.safety = SafetyGuard()             # Safety filters
        self.disclaimer_mgr = DisclaimerManager()
        self.lang_detector = LanguageDetector()
        self.confidence_scorer = ConfidenceScorer()
        logger.logger.info("=" * 80)
        logger.logger.info("ROBUST MEDICAL RAG SYSTEM READY")
        logger.logger.info("=" * 80)
    # ------------------- ANSWER METHOD -------------------
    def answer(
        self,
        query: str,
        lang: str = None,
        skip_cache: bool = False
    ) -> Dict:
        """
        Main entry point for answering queries
        """
        start_time = datetime.datetime.now()
        try:
            # ========== STEP 1: SAFETY CHECK ==========
            is_safe, reason = self.safety.is_safe_query(query)
            if not is_safe:
                logger.log_safety(query, reason, "Harmful query detected")
                return self._create_response(
                    answer="I cannot provide information on this topic. Please contact a healthcare professional or emergency services if needed.",
                    confidence=0.0,
                    status="BLOCKED_UNSAFE",
                    metadata={"block_reason": reason}
                )
            # Check for emergency
            if self.safety.is_emergency(query):
                logger.log_safety(query, "EMERGENCY", "Emergency query detected")
                return self._create_response(
                    answer=self.safety.get_emergency_response(),
                    confidence=1.0,
                    status="EMERGENCY_RESPONSE",
                    metadata={}
                )
            # ========== STEP 2: LANGUAGE DETECTION ==========
            if lang is None:
                available_langs = list(self.retriever.indices.keys())
                lang = self.lang_detector.detect(query, available_langs)
            logger.log_query(query, lang, "PROCESSING")
            # ========== STEP 3: CACHE CHECK ==========
            if not skip_cache:
                cached_result = self.cache.get(query, lang)
                if cached_result:
                    logger.log_query(query, lang, "CACHE_HIT")
                    cached_result['metadata']['from_cache'] = True
                    return cached_result
            # ========== STEP 4: RETRIEVAL ==========
            contexts, retrieval_scores, retrieval_status = self.retriever.retrieve_with_confidence(
                query, lang
            )
            if retrieval_status == "LOW_CONFIDENCE" or len(contexts) == 0:
                logger.log_query(query, lang, "LOW_RETRIEVAL_CONFIDENCE")
                answer = self.generator._generate_no_context_response()
                return self._create_response(
                    answer=answer,
                    confidence=0.3,
                    status="NO_RELEVANT_CONTEXT",
                    metadata={"retrieval_status": retrieval_status}
                )
            # ========== STEP 5: DEDUPLICATION ==========
            contexts = self.retriever.deduplicate_contexts(contexts)
            # ========== STEP 6: RE-RANKING ==========
            contexts = self.retriever.rerank_contexts(query, contexts)
            # ========== STEP 7: ANSWER GENERATION ==========
            answer, gen_metadata = self.generator.generate(query, contexts, lang)
            # ========== STEP 8: ANSWER VALIDATION ==========
            is_valid, validation_reason, validation_confidence = self.generator.validate_answer(
                answer, contexts
            )
            if not is_valid and validation_reason != "VALID_REFUSAL":
                logger.log_query(query, lang, f"VALIDATION_FAILED: {validation_reason}")
                answer = self.generator._generate_no_context_response()
                validation_confidence = 0.3
            # ========== STEP 9: SAFETY FILTER ==========
            answer, was_filtered = self.safety.filter_dangerous_advice(answer, query)
            if was_filtered:
                logger.log_safety(query, "DANGEROUS_ADVICE", "Answer filtered")
            # ========== STEP 10: ADD DISCLAIMER ==========
            answer = self.disclaimer_mgr.add_disclaimer(answer, query)
            # ========== STEP 11: CONFIDENCE CALCULATION ==========
            confidence = self.confidence_scorer.calculate_confidence(
                retrieval_scores,
                answer,
                contexts,
                validation_confidence
            )
            confidence_level = self.confidence_scorer.interpret_confidence(confidence)
            # ========== STEP 12: CREATE RESPONSE ==========
            total_time = (datetime.datetime.now() - start_time).total_seconds()
            response = self._create_response(
                answer=answer,
                confidence=confidence,
                confidence_level=confidence_level,
                status="SUCCESS",
                sources=contexts,
                metadata={
                    "retrieval_status": retrieval_status,
                    "num_contexts_retrieved": len(contexts),
                    "retrieval_scores": retrieval_scores,
                    "validation_reason": validation_reason,
                    "generation_time": gen_metadata.get("generation_time", 0),
                    "total_time": total_time,
                    "language": lang,
                    "was_safety_filtered": was_filtered,
                    "from_cache": False
                }
            )
            # ========== STEP 13: CACHE RESULT ==========
            self.cache.set(query, lang, response)
            # ========== STEP 14: LOG SUCCESS ==========
            logger.log_query(
                query, lang, "SUCCESS",
                confidence=confidence,
                time=total_time
            )
            return response
        except Exception as e:
            logger.log_error("PIPELINE_ERROR", str(e), query)
            return self._create_response(
                answer="I encountered an error processing your question. Please try rephrasing or contact support.",
                confidence=0.0,
                status="ERROR",
                metadata={"error": str(e)}
            )
    # ------------------- CREATE RESPONSE -------------------
    def _create_response(
        self,
        answer: str,
        confidence: float,
        status: str,
        confidence_level: str = None,
        sources: List[Dict] = None,
        metadata: Dict = None
    ) -> Dict:
        """Standardized response dictionary"""
        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_level": confidence_level or self.confidence_scorer.interpret_confidence(confidence),
            "status": status,
            "sources": sources or [],
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now().isoformat()
        }
    # ------------------- BATCH ANSWER -------------------
    def batch_answer(self, queries: List[Tuple[str, str]]) -> List[Dict]:
        results = []
        for query, lang in tqdm(queries, desc="Processing batch"):
            results.append(self.answer(query, lang))
        return results
    # ------------------- LIST FAISS INDICES -------------------
    def list_faiss_indices(self, lang: str = None) -> None:
        if lang:
            langs_to_check = [lang] if lang in self.retriever.indices else []
            if not langs_to_check:
                print(f"No indices found for language '{lang}'")
                return
        else:
            langs_to_check = list(self.retriever.indices.keys())
        print("Listing FAISS index files:")
        for l in langs_to_check:
            index_path = self.retriever.indices[l]
            print(f"\nLanguage: {l}\nIndex Path: {index_path}")
            if os.path.exists(index_path):
                for root, dirs, files in os.walk(index_path):
                    for file in files:
                        print(os.path.join(root, file))
            else:
                print("  ❌ Index path does not exist")
# ------------------- INITIALIZE MAIN RAG SYSTEM -------------------
rag_system = RobustMedicalRAG()
print("✅ Main RAG system initialized")


# =========================
# INTERACTIVE Q&A SESSION
# =========================
print("🩺 Robust Medical RAG Interactive Q&A")
print("Type 'exit' to quit.\n")
while True:
    query = input("❓ Enter your medical question: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting interactive session. Stay healthy!")
        break
    # Optional: specify language, or leave None for auto-detect
    lang = None
    # Get answer from RAG
    response = rag_system.answer(query, lang=lang)
    # Display results
    print("\n💬 Answer:")
    print(response["answer"])
    print(f"\n📊 Confidence: {response['confidence']:.2f} ({response.get('confidence_level', 'N/A')})")
    print(f"📝 Status: {response['status']}")
    print(f"📚 Number of sources used: {len(response['sources'])}")
    print("-" * 60, "\n")


# ============================================================================
# SECTION 11: MAIN RAG PIPELINE
# ============================================================================
import os
import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
class RobustMedicalRAG:
    """
    Main RAG pipeline with all safety and quality features
    """
    def __init__(self):
        self.retriever = retriever
        self.generator = generator
        self.cache = query_cache
        self.safety = SafetyGuard()
        self.disclaimer_mgr = DisclaimerManager()
        self.lang_detector = LanguageDetector()
        self.confidence_scorer = ConfidenceScorer()
        logger.logger.info("=" * 80)
        logger.logger.info("ROBUST MEDICAL RAG SYSTEM READY")
        logger.logger.info("=" * 80)
    # ------------------- ANSWER METHOD -------------------
    def answer(
        self,
        query: str,
        lang: str = None,
        skip_cache: bool = False
    ) -> Dict:
        """
        Main entry point for answering queries
        """
        start_time = datetime.datetime.now()
        try:
            # ========== STEP 1: SAFETY CHECK ==========
            is_safe, reason = self.safety.is_safe_query(query)
            if not is_safe:
                logger.log_safety(query, reason, "Harmful query detected")
                return self._create_response(
                    answer="I cannot provide information on this topic. Please contact a healthcare professional or emergency services if needed.",
                    confidence=0.0,
                    status="BLOCKED_UNSAFE",
                    metadata={"block_reason": reason}
                )
            # Check for emergency
            if self.safety.is_emergency(query):
                logger.log_safety(query, "EMERGENCY", "Emergency query detected")
                return self._create_response(
                    answer=self.safety.get_emergency_response(),
                    confidence=1.0,
                    status="EMERGENCY_RESPONSE",
                    metadata={}
                )
            # ========== STEP 2: LANGUAGE DETECTION ==========
            if lang is None:
                available_langs = list(self.retriever.indices.keys())
                lang = self.lang_detector.detect(query, available_langs)
            logger.log_query(query, lang, "PROCESSING")
            # ========== STEP 3: CACHE CHECK ==========
            if not skip_cache:
                cached_result = self.cache.get(query, lang)
                if cached_result:
                    logger.log_query(query, lang, "CACHE_HIT")
                    cached_result['metadata']['from_cache'] = True
                    return cached_result
            # ========== STEP 4: RETRIEVAL ==========
            contexts, retrieval_scores, retrieval_status = self.retriever.retrieve_with_confidence(
                query, lang
            )
            if retrieval_status == "LOW_CONFIDENCE" or len(contexts) == 0:
                logger.log_query(query, lang, "LOW_RETRIEVAL_CONFIDENCE")
                answer = self.generator._generate_no_context_response(query)
                return self._create_response(
                    answer=answer,
                    confidence=0.3,
                    status="NO_RELEVANT_CONTEXT",
                    metadata={"retrieval_status": retrieval_status}
                )
            # ========== STEP 5: DEDUPLICATION ==========
            contexts = self.retriever.deduplicate_contexts(contexts)
            # ========== STEP 6: RE-RANKING ==========
            contexts = self.retriever.rerank_contexts(query, contexts)
            # ========== STEP 7: ANSWER GENERATION ==========
            answer, gen_metadata = self.generator.generate(query, contexts, lang)
            # ========== STEP 8: ANSWER VALIDATION ==========
            is_valid, validation_reason, validation_confidence = self.generator.validate_answer(
                answer, contexts, query
            )
            if not is_valid and validation_reason != "VALID_REFUSAL":
                logger.log_query(query, lang, f"VALIDATION_FAILED: {validation_reason}")
                answer = self.generator._generate_no_context_response(query)
                validation_confidence = 0.3
            # ========== STEP 9: SAFETY FILTER ==========
            answer, was_filtered = self.safety.filter_dangerous_advice(answer, query)
            if was_filtered:
                logger.log_safety(query, "DANGEROUS_ADVICE", "Answer filtered")
            # ========== STEP 10: ADD DISCLAIMER ==========
            answer = self.disclaimer_mgr.add_disclaimer(answer, query)
            # ========== STEP 11: CONFIDENCE CALCULATION ==========
            confidence = self.confidence_scorer.calculate_confidence(
                retrieval_scores,
                answer,
                contexts,
                validation_confidence
            )
            confidence_level = self.confidence_scorer.interpret_confidence(confidence)
            # ========== STEP 12: CREATE RESPONSE ==========
            total_time = (datetime.datetime.now() - start_time).total_seconds()
            response = self._create_response(
                answer=answer,
                confidence=confidence,
                confidence_level=confidence_level,
                status="SUCCESS",
                sources=contexts,
                metadata={
                    "retrieval_status": retrieval_status,
                    "num_contexts_retrieved": len(contexts),
                    "retrieval_scores": retrieval_scores,
                    "validation_reason": validation_reason,
                    "generation_time": gen_metadata.get("generation_time", 0),
                    "total_time": total_time,
                    "language": lang,
                    "was_safety_filtered": was_filtered,
                    "from_cache": False
                }
            )
            # ========== STEP 13: CACHE RESULT ==========
            self.cache.set(query, lang, response)
            # ========== STEP 14: LOG SUCCESS ==========
            logger.log_query(
                query, lang, "SUCCESS",
                confidence=confidence,
                time=total_time
            )
            return response
        except Exception as e:
            logger.log_error("PIPELINE_ERROR", str(e), query)
            return self._create_response(
                answer="I encountered an error processing your question. Please try rephrasing or contact support.",
                confidence=0.0,
                status="ERROR",
                metadata={"error": str(e)}
            )
    # ------------------- CREATE RESPONSE -------------------
    def _create_response(
        self,
        answer: str,
        confidence: float,
        status: str,
        confidence_level: str = None,
        sources: List[Dict] = None,
        metadata: Dict = None
    ) -> Dict:
        """Create standardized response dictionary"""
        return {
            "answer": answer,
            "confidence": confidence,
            "confidence_level": confidence_level or self.confidence_scorer.interpret_confidence(confidence),
            "status": status,
            "sources": sources or [],
            "metadata": metadata or {},
            "timestamp": datetime.datetime.now().isoformat()
        }
    # ------------------- BATCH ANSWER -------------------
    def batch_answer(self, queries: List[Tuple[str, str]]) -> List[Dict]:
        """Process multiple queries efficiently"""
        results = []
        for query, lang in tqdm(queries, desc="Processing batch"):
            result = self.answer(query, lang)
            results.append(result)
        return results
    # ------------------- LIST FAISS INDICES -------------------
    def list_faiss_indices(self, lang: str = None) -> None:
        """
        Print all FAISS index files for available languages or a specific language.
        Args:
            lang: Optional, specific language to inspect. If None, lists all.
        """
        if lang:
            langs_to_check = [lang] if lang in self.retriever.indices else []
            if not langs_to_check:
                print(f"No indices found for language '{lang}'")
                return
        else:
            langs_to_check = list(self.retriever.indices.keys())
        print("Listing FAISS index files:")
        for l in langs_to_check:
            index_path = self.retriever.indices[l]
            print(f"\nLanguage: {l}\nIndex Path: {index_path}")
            if os.path.exists(index_path):
                for root, dirs, files in os.walk(index_path):
                    for file in files:
                        print(os.path.join(root, file))
            else:
                print("  ❌ Index path does not exist")
# Initialize main RAG system
rag_system = RobustMedicalRAG()
print("✅ Main RAG system initialized")
# Example usage:
# rag_system.list_faiss_indices()        # Lists all languages
# rag_system.list_faiss_indices("en")    # Lists English indices only


# ============================================================================
# SECTION 12: RESULT PERSISTENCE
# ============================================================================
class ResultManager:
    """Manage saving and loading of results"""
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    def save_result(self, response: Dict, query: str, lang: str):
        """Save individual query result"""
        result = {
            "query": query,
            "language": lang,
            "answer": response["answer"],
            "confidence": response["confidence"],
            "confidence_level": response["confidence_level"],
            "status": response["status"],
            "sources": response["sources"],
            "metadata": response["metadata"],
            "timestamp": response["timestamp"]
        }
        # Save to language-specific file
        filepath = os.path.join(self.results_dir, f"{lang}_results.jsonl")
        with open(filepath, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
        logger.logger.info(f"Saved result for {lang} query")
    def save_batch_results(self, results: List[Dict], queries: List[Tuple[str, str]]):
        """Save batch results"""
        for result, (query, lang) in zip(results, queries):
            self.save_result(result, query, lang)
    def load_results(self, lang: str) -> List[Dict]:
        """Load all results for a language"""
        filepath = os.path.join(self.results_dir, f"{lang}_results.jsonl")
        if not os.path.exists(filepath):
            return []
        results = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        return results
    def get_statistics(self) -> Dict:
        """Get system statistics"""
        stats = {
            "total_queries": 0,
            "by_language": {},
            "by_status": {},
            "avg_confidence": 0.0,
            "confidence_distribution": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
        }
        all_confidences = []
        for lang_file in os.listdir(self.results_dir):
            if not lang_file.endswith("_results.jsonl"):
                continue
            lang = lang_file.replace("_results.jsonl", "")
            results = self.load_results(lang)
            stats["by_language"][lang] = len(results)
            stats["total_queries"] += len(results)
            for result in results:
                status = result.get("status", "UNKNOWN")
                stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
                confidence = result.get("confidence", 0.0)
                all_confidences.append(confidence)
                conf_level = result.get("confidence_level", "LOW")
                stats["confidence_distribution"][conf_level] += 1
        if all_confidences:
            stats["avg_confidence"] = round(np.mean(all_confidences), 3)
        return stats
# Initialize result manager
result_manager = ResultManager(config.RESULTS_DIR)
print("✅ Result manager initialized")

# ============================================================================
# SECTION 13: EVALUATION METRICS
# ============================================================================
class EvaluationMetrics:
    """Evaluate answer quality"""
    @staticmethod
    def calculate_metrics(generated: str, reference: str) -> Dict:
        """
        Calculate quality metrics
        Note: Requires ground truth for proper evaluation
        """
        try:
            from rouge_score import rouge_scorer
            scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            scores = scorer.score(reference, generated)
            return {
                "rouge1_f": scores['rouge1'].fmeasure,
                "rouge2_f": scores['rouge2'].fmeasure,
                "rougeL_f": scores['rougeL'].fmeasure
            }
        except ImportError:
            logger.log_error("METRICS", "rouge_score not installed")
            return {}
    @staticmethod
    def evaluate_faithfulness(answer: str, contexts: List[Dict]) -> float:
        """
        Measure how much of the answer is supported by context
        Simple keyword-based approach
        """
        if not contexts:
            return 0.0
        # Extract keywords from answer
        answer_words = set(re.findall(r'\w+', answer.lower()))
        answer_words = {w for w in answer_words if len(w) > 4}
        # Extract keywords from contexts
        context_words = set()
        for ctx in contexts:
            words = re.findall(r'\w+', ctx['answer'].lower())
            context_words.update([w for w in words if len(w) > 4])
        if not answer_words:
            return 0.0
        # Calculate overlap
        overlap = len(answer_words & context_words)
        faithfulness = overlap / len(answer_words)
        return round(faithfulness, 3)
print("✅ Evaluation metrics ready")

# ============================================================================
# SECTION 14: UTILITY FUNCTIONS
# ============================================================================
def print_response(response: Dict, show_sources: bool = True):
    """Pretty print a response"""
    print("\n" + "="*80)
    print("MEDICAL RAG SYSTEM RESPONSE")
    print("="*80)
    print(f"\n📊 Status: {response['status']}")
    print(f"🎯 Confidence: {response['confidence']:.3f} ({response['confidence_level']})")
    print(f"\n💬 Answer:")
    print("-" * 80)
    print(response['answer'])
    print("-" * 80)
    if show_sources and response['sources']:
        print(f"\n📚 Sources ({len(response['sources'])}):")
        for i, source in enumerate(response['sources'][:3], 1):  # Show top 3
            print(f"\n[{i}] Q: {source['question'][:100]}...")
            print(f"    A: {source['answer'][:150]}...")
    if response['metadata']:
        print(f"\n⏱️  Metadata:")
        print(f"   - Total time: {response['metadata'].get('total_time', 0):.2f}s")
        print(f"   - Language: {response['metadata'].get('language', 'N/A')}")
        print(f"   - Contexts retrieved: {response['metadata'].get('num_contexts_retrieved', 0)}")
    print("\n" + "="*80 + "\n")
def interactive_mode():
    """Interactive query mode"""
    print("\n🏥 Medical RAG System - Interactive Mode")
    print("Type 'quit' to exit, 'stats' for statistics, 'clear' to clear cache\n")
    while True:
        try:
            query = input("\n❓ Your question: ").strip()
            if query.lower() == 'quit':
                print("Goodbye!")
                break
            if query.lower() == 'stats':
                stats = result_manager.get_statistics()
                print(json.dumps(stats, indent=2))
                continue
            if query.lower() == 'clear':
                rag_system.cache.clear()
                print("✅ Cache cleared")
                continue
            if not query:
                continue
            # Process query
            response = rag_system.answer(query)
            # Save result
            result_manager.save_result(
                response,
                query,
                response['metadata'].get('language', 'English')
            )
            # Print response
            print_response(response)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")


"""
FAISS INDEX CREATION ON ALL NLP Dataset
"""
# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')
# --- Imports ---
import os, json, datetime, re
import numpy as np
import torch
from tqdm import tqdm
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- Paths ---
BASE_DIR = "/content/drive/MyDrive/NLP_A14_Health/combined_qns/"
FAISS_DIR = "/content/faiss_indices/"
RESULTS_DIR = "/content/drive/MyDrive/NLP_A14_Health/results/"
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# --- Device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# --- Load Embeddings Model ---
embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device=device
)
# --- Load Generator Model (LLM) ---
GEN_MODEL = "bigscience/bloomz-560m"  # instruction-tuned multilingual
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
generator = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(device)
# --- Helper: Check Yes/No Question ---
def is_yes_no_question(question):
    yes_no_starters = ["is","are","do","does","did","can","could","will","would","should","may","have"]
    question = question.strip().lower()
    if any(question.startswith(word + " ") for word in yes_no_starters):
        if not re.search(r"\b(how|why|explain|tell|describe)\b", question):
            return True
    return False
# --- Build / Load FAISS Indices (once) ---
def build_or_load_indices():
    indices, data_maps, lang_data = {}, {}, {}
    for file in os.listdir(BASE_DIR):
        if not file.endswith(".json"):
            continue
        lang = file.replace("_combined_qns.json","")
        print(f"🔹 Processing language: {lang}")
        json_path = os.path.join(BASE_DIR, file)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        lang_data[lang] = data
        texts = [qa["question"] + " " + qa["answer"] for qa in data]
        n_items = len(texts)
        # --- Create embeddings ---
        embeddings = embedder.encode(
            texts, batch_size=64, show_progress_bar=True,
            convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")
        dim = embeddings.shape[1]
        # --- Create / Load FAISS index ---
        index_file = os.path.join(FAISS_DIR, f"{lang}_index.faiss")
        map_file   = os.path.join(FAISS_DIR, f"{lang}_map.npy")
        if os.path.exists(index_file) and os.path.exists(map_file):
            print(f"   ⚡ Loading existing index for {lang}")
            index = faiss.read_index(index_file)
            data_map = np.load(map_file)
        else:
            print(f"   ⚙️ Building new FAISS index for {lang}")
            if n_items < 1000:
                index = faiss.IndexFlatIP(dim)
            else:
                nlist = 256 if n_items < 50000 else 1024
                quantizer = faiss.IndexFlatIP(dim)
                index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
                index.train(embeddings)
            index.add(embeddings)
            faiss.write_index(index, index_file)
            np.save(map_file, np.arange(n_items))
            data_map = np.arange(n_items)
        indices[lang] = index
        data_maps[lang] = data_map
        print(f"✅ {lang} index ready ({n_items} items)")
    return indices, data_maps, lang_data
indices, data_maps, lang_data = build_or_load_indices()
# --- Retrieve top-K contexts ---
def retrieve_context(query, lang, top_k=5):
    if lang not in indices:
        raise ValueError(f"No FAISS index for language {lang}")
    index = indices[lang]
    data_map = data_maps[lang]
    data = lang_data[lang]
    # --- Embed query ---
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    # --- Check dimension ---
    if q_emb.shape[1] != index.d:
        raise ValueError(f"Query embedding dim {q_emb.shape[1]} != index dim {index.d}")
    top_k = min(top_k, len(data))
    D, I = index.search(q_emb, top_k)
    context_items = [data[int(data_map[i])] for i in I[0] if i < len(data_map)]
    return context_items
# --- Generate answer ---
def generate_answer(query, lang, context_items, max_tokens=200):
    if is_yes_no_question(query):
        context_text = " ".join([c["answer"] for c in context_items])
        if re.search(r"\b(no|not|never|avoid|don\'t|cannot)\b", context_text.lower()):
            return "No"
        else:
            return "Yes"
    # LLM answer
    context_text = "\n".join([f"Q: {c['question']} A: {c['answer']}" for c in context_items])
    prompt = f"You are a helpful medical assistant. Answer in {lang} using the context below.\n\nContext:\n{context_text}\n\nQuestion:\n{query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    outputs = generator.generate(
        inputs['input_ids'], max_new_tokens=max_tokens, num_beams=5, early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
# --- Save results ---
def save_results(lang, query, answer, context_items):
    result = {
        "language": lang,
        "query": query,
        "generated_answer": answer,
        "retrieved_contexts": context_items,
        "timestamp": str(datetime.datetime.now())
    }
    paths = [f"./{lang}_results.json", os.path.join(RESULTS_DIR, f"{lang}_results.json")]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(result)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved results for {lang} to local + Drive")
# --- Print contexts ---
def print_contexts(context_items):
    print("\n📄 Retrieved Contexts:")
    for i, c in enumerate(context_items, 1):
        print(f"[{i}] Q: {c['question']}\n    A: {c['answer']}\n")
# --- Example Queries ---
example_queries = [
    ("Is honey okay to try for mucosal healing if my sugar's under control?", "English"),
    ("ডায়বিটিসে শहद খাওয়া নিরাপদ কি?", "Hindi"),
    ("I have severe chest pain and can't breathe properly", "English"),
    ("How to perform surgery at home without doctor?", "English"),
]
for q, lang in example_queries:
    print("="*80)
    print(f"Query ({lang}): {q}")
    ctx = retrieve_context(q, lang, top_k=5)
    print_contexts(ctx)
    answer = generate_answer(q, lang, ctx)
    print(f"💬 Generated Answer:\n{answer}")
    save_results(lang, q, answer, ctx)


"""
Load existing FAISS indices + RAG System (Multilingual Health QA)
"""
# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')
# --- Imports ---
import os, json, datetime, re
import numpy as np
import torch
import faiss
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- Paths ---
BASE_DIR = "/content/drive/MyDrive/NLP_A14_Health/combined_qns/"
FAISS_DIR = "/content/faiss_indices/"
RESULTS_DIR = "/content/drive/MyDrive/NLP_A14_Health/results/"
os.makedirs(FAISS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
# --- Device ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
# --- Load Embeddings Model ---
embedder = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    device=device
)
# --- Load Generator Model (LLM) ---
GEN_MODEL = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
generator = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(device)
# --- Helper: Check Yes/No Question ---
def is_yes_no_question(question):
    yes_no_starters = ["is","are","do","does","did","can","could","will","would","should","may","have"]
    question = question.strip().lower()
    if any(question.startswith(word + " ") for word in yes_no_starters):
        if not re.search(r"\b(how|why|explain|tell|describe)\b", question):
            return True
    return False
# --- Load FAISS indices and data ---
def load_indices_only():
    indices, data_maps, lang_data = {}, {}, {}
    for file in os.listdir(FAISS_DIR):
        if not file.endswith("_index.faiss"):
            continue
        lang = file.replace("_index.faiss", "")
        print(f"🔹 Loading index for language: {lang}")
        index_file = os.path.join(FAISS_DIR, f"{lang}_index.faiss")
        map_file   = os.path.join(FAISS_DIR, f"{lang}_map.npy")
        json_file  = os.path.join(BASE_DIR, f"{lang}_combined_qns.json")
        if not os.path.exists(index_file) or not os.path.exists(map_file) or not os.path.exists(json_file):
            print(f"⚠️ Missing files for {lang}, skipping...")
            continue
        # Load FAISS index
        index = faiss.read_index(index_file)
        data_map = np.load(map_file)
        # Load QA data
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        indices[lang] = index
        data_maps[lang] = data_map
        lang_data[lang] = data
        print(f"✅ {lang} loaded ({len(data)} items)")
    return indices, data_maps, lang_data
indices, data_maps, lang_data = load_indices_only()
# --- Retrieve top-K contexts ---
def retrieve_context(query, lang, top_k=5):
    if lang not in indices:
        raise ValueError(f"No FAISS index for language {lang}")
    index = indices[lang]
    data_map = data_maps[lang]
    data = lang_data[lang]
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    if q_emb.shape[1] != index.d:
        raise ValueError(f"Query embedding dim {q_emb.shape[1]} != index dim {index.d}")
    top_k = min(top_k, len(data))
    _, I = index.search(q_emb, top_k)
    return [data[int(data_map[i])] for i in I[0] if i < len(data_map)]
# --- Generate answer ---
def generate_answer(query, lang, context_items, max_tokens=200):
    if is_yes_no_question(query):
        context_text = " ".join([c["answer"] for c in context_items])
        return "No" if re.search(r"\b(no|not|never|avoid|don\'t|cannot)\b", context_text.lower()) else "Yes"
    context_text = "\n".join([f"Q: {c['question']} A: {c['answer']}" for c in context_items])
    prompt = f"You are a helpful medical assistant. Answer in {lang} using the context below.\n\nContext:\n{context_text}\n\nQuestion:\n{query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(device)
    outputs = generator.generate(inputs['input_ids'], max_new_tokens=max_tokens, num_beams=5, early_stopping=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
# --- Save results ---
def save_results(lang, query, answer, context_items):
    result = {
        "language": lang,
        "query": query,
        "generated_answer": answer,
        "retrieved_contexts": context_items,
        "timestamp": str(datetime.datetime.now())
    }
    paths = [os.path.join(RESULTS_DIR, f"{lang}_results.json"), f"./{lang}_results.json"]
    for path in paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = []
        data.append(result)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"💾 Saved results for {lang}")
# --- Print contexts ---
def print_contexts(context_items):
    print("\n📄 Retrieved Contexts:")
    for i, c in enumerate(context_items, 1):
        print(f"[{i}] Q: {c['question']}\n    A: {c['answer']}\n")
# --- Example Queries ---
example_queries = [
    ("Is honey okay to try for mucosal healing if my sugar's under control?", "English"),
    ("ডায়বিটিসে শहদ খাওয়া নিরাপদ কি?", "Hindi"),
    ("I have severe chest pain and can't breathe properly", "English"),
    ("How to perform surgery at home without doctor?", "English"),
]
for q, lang in example_queries:
    print("="*80)
    print(f"Query ({lang}): {q}")
    ctx = retrieve_context(q, lang, top_k=5)
    print_contexts(ctx)
    answer = generate_answer(q, lang, ctx)
    print(f"💬 Generated Answer:\n{answer}")
    save_results(lang, q, answer, ctx)


# ============================================================================
# SECTION 16: MAIN EXECUTION
# ============================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("ROBUST MEDICAL RAG SYSTEM")
    print("Version 2.0 - Production Ready")
    print("="*80)
    # Check if indices exist
    if not os.listdir(config.FAISS_DIR):
        print("\n⚠️  No FAISS indices found!")
        print("Please run build_faiss_indices() first to create indices.")
        print("\nUncomment the following line to build indices:")
        print("# build_faiss_indices()")
    else:
        print("\n✅ All systems initialized successfully!")
        print("\nAvailable modes:")
        print("1. run_examples() - Run example queries")
        print("2. interactive_mode() - Interactive Q&A mode")
        print("3. Custom usage - Call rag_system.answer(query, lang)")
        # Uncomment to run examples
        # run_examples()
        # Uncomment for interactive mode
        # interactive_mode()
print("\n✅ RAG system fully loaded and ready!")


"""
Multilingual FAISS RAG Loader + Retriever
"""
import os
import shutil
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
# ===========================
# 1️⃣ Copy FAISS backup from Drive to local
# ===========================
DRIVE_FAISS_DIR = "/content/drive/MyDrive/faiss_backup"
LOCAL_FAISS_DIR = "/content/faiss_indices"
os.makedirs(LOCAL_FAISS_DIR, exist_ok=True)
print(f"📦 Copying FAISS backup from Drive ({DRIVE_FAISS_DIR}) to local ({LOCAL_FAISS_DIR})...")
for item in os.listdir(DRIVE_FAISS_DIR):
    s = os.path.join(DRIVE_FAISS_DIR, item)
    d = os.path.join(LOCAL_FAISS_DIR, item)
    if os.path.isdir(s):
        shutil.copytree(s, d, dirs_exist_ok=True)
    else:
        shutil.copy2(s, d)
print("✅ Copy complete.\n")
# ===========================
# 2️⃣ Setup device + embedder
# ===========================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")
embedder = SentenceTransformer(
    "intfloat/e5-large",
    device=device
)
# ===========================
# 3️⃣ Load all indices and maps
# ===========================
indices = {}
data_maps = {}
lang_data = {}
BASE_JSON_DIR = "/content/drive/MyDrive/NLP_A14_Health/combined_qns"
for file in os.listdir(LOCAL_FAISS_DIR):
    if file.endswith("_index.faiss"):
        lang = file.replace("_index.faiss", "")
        print(f"🔹 Loading language: {lang}")
        index_path = os.path.join(LOCAL_FAISS_DIR, f"{lang}_index.faiss")
        map_path = os.path.join(LOCAL_FAISS_DIR, f"{lang}_map.npy")
        json_path = os.path.join(BASE_JSON_DIR, f"{lang}_combined_qns.json")
        # Load FAISS index
        indices[lang] = faiss.read_index(index_path)
        # Load mapping
        data_maps[lang] = np.load(map_path, allow_pickle=True)
        # Load original Q&A data
        with open(json_path, "r", encoding="utf-8") as f:
            lang_data[lang] = json.load(f)
print(f"\n✅ Loaded {len(indices)} languages successfully.\n")
# ===========================
# 4️⃣ Retrieval function
# ===========================
def retrieve(query, lang, top_k=5):
    """
    Retrieve top-K Q&A for a query in a given language
    """
    if lang not in indices:
        raise ValueError(f"No index found for language '{lang}'")
    index = indices[lang]
    data_map = data_maps[lang]
    data = lang_data[lang]
    # Embed query
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    # Search
    D, I = index.search(q_emb, top_k)
    # Retrieve original items
    results = [data[int(data_map[i])] for i in I[0]]
    scores = D[0].tolist()
    return results, scores
# ===========================
# 5️⃣ Example usage
# ===========================
query = "Is honey okay to try for mucosal healing if my sugar's under control, or should I skip home remedies altogether?"
lang = "English"
top_k = 5
contexts, scores = retrieve(query, lang, top_k)
print(f"\n🧠 Query: {query}\n")
for i, ctx in enumerate(contexts, 1):
    print(f"[{i}] Q: {ctx['question']}")
    print(f"    A: {ctx['answer']}")
    print(f"    Score: {scores[i-1]:.4f}\n")


# -*- coding: utf-8 -*-
"""
Multilingual Health RAG System (FAISS + Bloomz)
Interactive tester with explanatory answers
"""
# --- Mount Google Drive ---
from google.colab import drive
drive.mount('/content/drive')
# --- Imports ---
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- Paths ---
LOCAL_FAISS_DIR = "/content/faiss_indices"
BASE_JSON_DIR = "/content/drive/MyDrive/NLP_A14_Health/combined_qns"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
# --- Load Embeddings and Generator Model ---
embedder = SentenceTransformer(
    "intfloat/e5-large",
    device=DEVICE
)
GEN_MODEL = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
generator = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(DEVICE)
# --- Load all FAISS indices, maps, and data ---
indices = {}
data_maps = {}
lang_data = {}
for file in os.listdir(LOCAL_FAISS_DIR):
    if file.endswith("_index.faiss"):
        lang = file.replace("_index.faiss", "")
        print(f"🔹 Loading language: {lang}")
        indices[lang] = faiss.read_index(os.path.join(LOCAL_FAISS_DIR, f"{lang}_index.faiss"))
        data_maps[lang] = np.load(os.path.join(LOCAL_FAISS_DIR, f"{lang}_map.npy"), allow_pickle=True)
        json_path = os.path.join(BASE_JSON_DIR, f"{lang}_combined_qns.json")
        with open(json_path, "r", encoding="utf-8") as f:
            lang_data[lang] = json.load(f)
print(f"\n✅ Loaded {len(indices)} languages successfully.\n")
# --- Retrieval function ---
def retrieve(query, lang, top_k=5):
    if lang not in indices:
        raise ValueError(f"No index found for language '{lang}'")
    index = indices[lang]
    data_map = data_maps[lang]
    data = lang_data[lang]
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = [data[int(data_map[i])] for i in I[0]]
    scores = D[0].tolist()
    return results, scores
# --- Explanatory answer generation ---
def generate_explanatory_answer(query, lang, context_items, max_tokens=300):
    # Detect yes/no type question
    yes_no_starters = ["is","are","do","does","did","can","could","will","would","should","may","have"]
    question_lower = query.strip().lower()
    is_yes_no = any(question_lower.startswith(w + " ") for w in yes_no_starters) \
                and not any(word in question_lower for word in ["how","why","explain","describe","tell"])
    # Compose context text
    context_text = "\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in context_items])
    if is_yes_no:
        prompt = f"""You are a helpful medical assistant.
Answer the yes/no question in {lang} using the context below.
Explain your reasoning with reference to the retrieved documents.
Context:
{context_text}
Question:
{query}
Answer with explanation:"""
    else:
        prompt = f"""You are a helpful medical assistant.
Answer the question in {lang} using the context below.
Provide a detailed explanation referencing the retrieved documents.
Context:
{context_text}
Question:
{query}
Answer:"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(DEVICE)
    outputs = generator.generate(
        inputs['input_ids'],
        max_new_tokens=max_tokens,
        num_beams=5,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
# --- Interactive loop ---
print("📌 Paste your questions. Type 'exit' to quit.\n")
while True:
    query = input("Enter question: ").strip()
    if query.lower() == "exit":
        break
    try:
        detected_lang = detect(query)
    except:
        detected_lang = "English"  # fallback
    print(f"🧾 Detected language: {detected_lang}")
    # Find closest matching language available
    matched_lang = None
    for lang in indices.keys():
        if lang.lower() == detected_lang.lower():
            matched_lang = lang
            break
    if not matched_lang:
        matched_lang = "English"  # default
    ctx_items, scores = retrieve(query, matched_lang, top_k=5)
    print("\n📄 Retrieved Contexts:")
    for i, c in enumerate(ctx_items, 1):
        print(f"[{i}] Q: {c['question']}")
        print(f"    A: {c['answer']}")
        print(f"    Score: {scores[i-1]:.4f}\n")
    answer = generate_explanatory_answer(query, matched_lang, ctx_items)
    print(f"💬 Generated Answer:\n{answer}\n")


# -*- coding: utf-8 -*-
"""
Multilingual Health RAG System (FAISS + Bloomz)
Exact match returns document answer; approximate match summarizes top contexts with elaboration
"""
# --- Mount Drive ---
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
# --- Imports ---
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- Paths & device ---
LOCAL_FAISS_DIR = "/content/faiss_indices"
BASE_JSON_DIR = "/content/drive/MyDrive/NLP_A14_Health/combined_qns"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
# --- Load embeddings & LLM ---
embedder = SentenceTransformer(
    "intfloat/e5-large",
    device=DEVICE
)
# GEN_MODEL = "bigscience/bloomz-560m"
# tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
# generator = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(DEVICE)
# --- Load FAISS indices and data ---
indices, data_maps, lang_data = {}, {}, {}
for file in os.listdir(LOCAL_FAISS_DIR):
    if file.endswith("_index.faiss"):
        lang = file.replace("_index.faiss", "")
        print(f"🔹 Loading language: {lang}")
        indices[lang] = faiss.read_index(os.path.join(LOCAL_FAISS_DIR, f"{lang}_index.faiss"))
        data_maps[lang] = np.load(os.path.join(LOCAL_FAISS_DIR, f"{lang}_map.npy"), allow_pickle=True)
        json_path = os.path.join(BASE_JSON_DIR, f"{lang}_combined_qns.json")
        with open(json_path, "r", encoding="utf-8") as f:
            lang_data[lang] = json.load(f)
print(f"\n✅ Loaded {len(indices)} languages successfully.\n")
# --- Retrieve top-K contexts ---
def retrieve(query, lang, top_k=5):
    if lang not in indices:
        return [], []
    index = indices[lang]
    data_map = data_maps[lang]
    data = lang_data[lang]
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = [data[int(data_map[i])] for i in I[0] if i < len(data_map)]
    scores = D[0].tolist()
    return results, scores
# --- Generate answer with elaboration ---
def generate_answer(query, lang, context_items, sim_threshold=0.85):
    if not context_items:
        return "Sorry, I don't have information on this topic in the documents."
    # 1️⃣ Exact match
    for c in context_items:
        if query.strip().lower() == c['question'].strip().lower():
            return c['answer']
    # 2️⃣ Similarity check for approximate match
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    answers = []
    for c in context_items:
        c_emb = embedder.encode([c['question']], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        sim = np.dot(query_emb, c_emb.T)[0][0]
        if sim >= sim_threshold:
            answers.append((sim, c['answer'], c['question']))
    # 3️⃣ Prepare prompt for elaboration
    if answers:
        answers = sorted(answers, key=lambda x: x[0], reverse=True)
        top_text = "\n".join([f"Q: {q}\nA: {a}" for _, a, q in answers[:5]])
    else:
        top_text = "\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in context_items[:5]])
    prompt = f"""
You are a knowledgeable medical assistant. Read the following context carefully.
Generate a detailed, explanatory answer to the question below strictly based on the context.
Use full, coherent sentences, elaborating clearly on the reasoning.
Each sentence should be informative, and try to make 30-50 word explanations if possible.
Do not hallucinate. If the context does not contain relevant info, say you don't have sufficient information.
Context:
{top_text}
Question:
{query}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(DEVICE)
    outputs = generator.generate(
        inputs['input_ids'],
        max_new_tokens=400,
        num_beams=5,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer
# --- Interactive loop ---
print("📌 Paste your questions. Type 'exit' to quit.\n")
while True:
    query = input("Enter question: ").strip()
    if query.lower() == "exit":
        break
    try:
        detected_lang = detect(query)
    except:
        detected_lang = "English"
    print(f"🧾 Detected language: {detected_lang}")
    # Match available language
    matched_lang = None
    for lang in indices.keys():
        if lang.lower() == detected_lang.lower():
            matched_lang = lang
            break
    if not matched_lang:
        matched_lang = "English"
    context_items, scores = retrieve(query, matched_lang, top_k=5)
    print("\n📄 Retrieved Contexts:")
    if not context_items:
        print("No relevant contexts found.")
    for i, c in enumerate(context_items, 1):
        print(f"[{i}] Q: {c['question']}")
        print(f"    A: {c['answer']}")
        print(f"    Score: {scores[i-1]:.4f}\n")
    answer = generate_answer(query, matched_lang, context_items)
    print(f"💬 Elaborated Answer:\n{answer}\n")


# -*- coding: utf-8 -*-
"""
Multilingual Health RAG System (FAISS + Bloomz)
Generates context-grounded explanatory answers of at least 30 words
"""
# --- Mount Drive ---
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
# --- Imports ---
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# --- Paths & device ---
LOCAL_FAISS_DIR = "/content/faiss_indices"
BASE_JSON_DIR = "/content/drive/MyDrive/NLP_A14_Health/combined_qns"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
# --- Load embeddings & LLM ---
embedder = SentenceTransformer(
    "intfloat/e5-large",
    device=DEVICE
)
GEN_MODEL = "bigscience/bloomz-560m"
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
generator = AutoModelForCausalLM.from_pretrained(GEN_MODEL).to(DEVICE)
# --- Load FAISS indices and data ---
indices, data_maps, lang_data = {}, {}, {}
for file in os.listdir(LOCAL_FAISS_DIR):
    if file.endswith("_index.faiss"):
        lang = file.replace("_index.faiss", "")
        print(f"🔹 Loading language: {lang}")
        indices[lang] = faiss.read_index(os.path.join(LOCAL_FAISS_DIR, f"{lang}_index.faiss"))
        data_maps[lang] = np.load(os.path.join(LOCAL_FAISS_DIR, f"{lang}_map.npy"), allow_pickle=True)
        json_path = os.path.join(BASE_JSON_DIR, f"{lang}_combined_qns.json")
        with open(json_path, "r", encoding="utf-8") as f:
            lang_data[lang] = json.load(f)
print(f"\n✅ Loaded {len(indices)} languages successfully.\n")
# --- Retrieve top-K contexts ---
def retrieve(query, lang, top_k=5):
    if lang not in indices:
        return [], []
    index = indices[lang]
    data_map = data_maps[lang]
    data = lang_data[lang]
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = [data[int(data_map[i])] for i in I[0] if i < len(data_map)]
    scores = D[0].tolist()
    return results, scores
# --- Generate elaborated answer ---
def generate_answer(query, lang, context_items, min_words=30, sim_threshold=0.85):
    if not context_items:
        return "Sorry, I don't have sufficient information on this topic in the documents."
    # Exact match check
    for c in context_items:
        if query.strip().lower() == c['question'].strip().lower():
            base_answer = c['answer']
            words = base_answer.split()
            if len(words) >= min_words:
                return base_answer
            else:
                # If too short, expand with other contexts
                additional = " ".join([ci['answer'] for ci in context_items if ci != c])
                elaborated = base_answer + " " + additional
                return " ".join(elaborated.split()[:min_words])
    # Approximate match
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    answers = []
    for c in context_items:
        c_emb = embedder.encode([c['question']], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        sim = np.dot(query_emb, c_emb.T)[0][0]
        if sim >= sim_threshold:
            answers.append((sim, c['answer'], c['question']))
    if answers:
        answers = sorted(answers, key=lambda x: x[0], reverse=True)
        top_text = "\n".join([f"Q: {q}\nA: {a}" for _, a, q in answers[:5]])
    else:
        top_text = "\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in context_items[:5]])
    prompt = f"""
You are a knowledgeable medical assistant. Read the following context carefully.
Generate a detailed, explanatory answer of at least {min_words} words to the question below strictly based on the context.
Each sentence should be coherent, informative, and elaborate clearly using the retrieved documents.
Do not hallucinate. If the context does not contain relevant info, say you don't have sufficient information.
Context:
{top_text}
Question:
{query}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(DEVICE)
    outputs = generator.generate(
        inputs['input_ids'],
        max_new_tokens=400,
        num_beams=5,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Ensure minimum word count
    words = answer.split()
    if len(words) < min_words:
        # append top contexts to reach minimum length
        addition = " ".join([c['answer'] for c in context_items])
        answer = answer + " " + addition
        words = answer.split()
        if len(words) > min_words:
            answer = " ".join(words[:min_words])
    return answer
# --- Interactive loop ---
print("📌 Paste your questions. Type 'exit' to quit.\n")
while True:
    query = input("Enter question: ").strip()
    if query.lower() == "exit":
        break
    try:
        detected_lang = detect(query)
    except:
        detected_lang = "English"
    print(f"🧾 Detected language: {detected_lang}")
    # Match available language
    matched_lang = None
    for lang in indices.keys():
        if lang.lower() == detected_lang.lower():
            matched_lang = lang
            break
    if not matched_lang:
        matched_lang = "English"
    context_items, scores = retrieve(query, matched_lang, top_k=5)
    print("\n📄 Retrieved Contexts:")
    if not context_items:
        print("No relevant contexts found.")
    for i, c in enumerate(context_items, 1):
        print(f"[{i}] Q: {c['question']}")
        print(f"    A: {c['answer']}")
        print(f"    Score: {scores[i-1]:.4f}\n")
    answer = generate_answer(query, matched_lang, context_items, min_words=30)
    print(f"💬 Elaborated Answer:\n{answer}\n")


# -*- coding: utf-8 -*-
"""
Multilingual Health RAG System (FAISS + mT5-small)
Strict context-based, 30+ word explanatory answers
"""
# --- Mount Drive ---
from google.colab import drive
drive.mount('/content/drive', force_remount=False)
# --- Imports ---
import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# --- Paths & device ---
LOCAL_FAISS_DIR = "/content/faiss_indices"
BASE_JSON_DIR = "/content/drive/MyDrive/NLP_A14_Health/combined_qns"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")
# --- Load embeddings & LLM ---
embedder = SentenceTransformer(
    "intfloat/e5-large",
    device=DEVICE
)
GEN_MODEL = "google/mt5-small"
tokenizer = AutoTokenizer.from_pretrained(GEN_MODEL)
generator = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL).to(DEVICE)
# --- Load FAISS indices and data ---
indices, data_maps, lang_data = {}, {}, {}
for file in os.listdir(LOCAL_FAISS_DIR):
    if file.endswith("_index.faiss"):
        lang = file.split("_")[0].capitalize()  # e.g., dogri_index.faiss -> dogri
        print(f"🔹 Loading language: {lang}")
        indices[lang] = faiss.read_index(os.path.join(LOCAL_FAISS_DIR, file))
        map_file = file.replace("_index.faiss", "_map.npy")
        data_maps[lang] = np.load(os.path.join(LOCAL_FAISS_DIR, map_file), allow_pickle=True)
        json_file = os.path.join(BASE_JSON_DIR, f"{lang}_combined_qns.json")
        with open(json_file, "r", encoding="utf-8") as f:
            lang_data[lang] = json.load(f)
print(f"\n✅ Loaded {len(indices)} languages successfully.\n")
# --- Retrieve top-K contexts ---
def retrieve(query, lang, top_k=5):
    if lang not in indices:
        return [], []
    index = indices[lang]
    data_map = data_maps[lang]
    data = lang_data[lang]
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    D, I = index.search(q_emb, top_k)
    results = [data[int(data_map[i])] for i in I[0] if i < len(data_map)]
    scores = D[0].tolist()
    return results, scores
# --- Generate elaborated answer ---
def generate_answer(query, context_items, min_words=30, sim_threshold=0.85):
    if not context_items:
        return "Sorry, I don't have sufficient information on this topic in the documents."
    # Exact match check
    for c in context_items:
        if query.strip().lower() == c['question'].strip().lower():
            base_answer = c['answer']
            words = base_answer.split()
            if len(words) >= min_words:
                return base_answer
            else:
                additional = " ".join([ci['answer'] for ci in context_items if ci != c])
                elaborated = base_answer + " " + additional
                return " ".join(elaborated.split()[:min_words])
    # Approximate match
    query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
    answers = []
    for c in context_items:
        c_emb = embedder.encode([c['question']], convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        sim = np.dot(query_emb, c_emb.T)[0][0]
        if sim >= sim_threshold:
            answers.append((sim, c['answer'], c['question']))
    if answers:
        answers = sorted(answers, key=lambda x: x[0], reverse=True)
        top_text = "\n".join([f"Q: {q}\nA: {a}" for _, a, q in answers[:5]])
    else:
        top_text = "\n".join([f"Q: {c['question']}\nA: {c['answer']}" for c in context_items[:5]])
    prompt = f"""
You are a knowledgeable medical assistant. Read the following context carefully.
Generate a detailed, explanatory answer of at least {min_words} words to the question below strictly based on the context.
Each sentence should be coherent, informative, and elaborate clearly using the retrieved documents.
Do not hallucinate. If the context does not contain relevant info, say you don't have sufficient information.
Context:
{top_text}
Question:
{query}
Answer:
"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(DEVICE)
    outputs = generator.generate(
        inputs['input_ids'],
        max_new_tokens=400,
        num_beams=5,
        early_stopping=True
    )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    words = answer.split()
    if len(words) < min_words:
        addition = " ".join([c['answer'] for c in context_items])
        answer = answer + " " + addition
        words = answer.split()
        if len(words) > min_words:
            answer = " ".join(words[:min_words])
    return answer
# --- Interactive loop ---
print("📌 Paste your questions. Type 'exit' to quit.\n")
while True:
    query = input("Enter question: ").strip()
    if query.lower() == "exit":
        break
    try:
        detected_lang = detect(query)
    except:
        detected_lang = "english"
    print(f"🧾 Detected language: {detected_lang}")
    # Map detected language to available FAISS index
    matched_lang = None
    for lang in indices.keys():
        if lang.lower() == detected_lang.lower():
            matched_lang = lang
            break
    # If Dogri detected (or input contains Dogri script) and index exists
    if "dogri" in indices and any("\u0900" <= c <= "\u097F" for c in query):
        matched_lang = "dogri"
    if not matched_lang:
        matched_lang = "english"
    context_items, scores = retrieve(query, matched_lang, top_k=5)
    print("\n📄 Retrieved Contexts:")
    if not context_items:
        print("No relevant contexts found.")
    for i, c in enumerate(context_items, 1):
        print(f"[{i}] Q: {c['question']}")
        print(f"    A: {c['answer']}")
        print(f"    Score: {scores[i-1]:.4f}\n")
    answer = generate_answer(query, context_items, min_words=30)
    print(f"💬 Elaborated Answer:\n{answer}\n")


# Step 1: Install gdown (if not already installed)
!pip install gdown
# Step 2: Import required libraries
import gdown
import zipfile
import os
# Step 3: Download the file from Google Drive
file_id = '1Qm1CPaaYHZu-PCeBwx-OrAFTkNQNZzHS'
url = f'https://drive.google.com/uc?id={file_id}'
output_zip = 'test_data_release.zip'
gdown.download(url, output_zip, quiet=False)
# Step 4: Extract the ZIP file
extract_dir = 'test_data_release'
os.makedirs(extract_dir, exist_ok=True)
with zipfile.ZipFile(output_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)


import os
import json
import re
import shutil
import gc
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from langdetect import detect
# ==================== PATH SETTINGS ====================
EXTRACT_PATH = "/content/test_data_release/test_data_release"
LOCAL_OUTPUT = "/content/FINAL_QNS_FOR_SUBMISSION"
DRIVE_OUTPUT = "/content/drive/MyDrive/Final_QNS_Real_answers"
os.makedirs(LOCAL_OUTPUT, exist_ok=True)
os.makedirs(DRIVE_OUTPUT, exist_ok=True)
# ==================== MODEL SETUP (Optimized for L4 GPU) ====================
SARVAM_MODEL_NAME = "sarvamai/sarvam-1"
print("🚀 Loading Sarvam model optimized for NVIDIA L4 GPU...")
torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(SARVAM_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    SARVAM_MODEL_NAME,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    device_map="auto"
)
# Compile for L4 optimization
if device == "cuda":
    try:
        model = torch.compile(model)
        print("✅ Model compiled successfully for L4 (bfloat16).")
    except Exception as e:
        print(f"⚠ Compilation skipped: {e}")
# ==================== CONFIG ====================
class Config:
    MAX_NEW_TOKENS = 400
    MIN_NEW_TOKENS = 40
    TEMPERATURE = 0.55
    TOP_P = 0.9
    REPETITION_PENALTY = 1.2
    BATCH_SIZE = 6
config = Config()
# ==================== HELPERS ====================
def build_prompt(user_query: str) -> str:
    """
    Build medical prompt dynamically based on detected language.
    """
 
   lang = detect(user_query)
    # Strict medical rules in English (universal)
    rules = (
        "- Understand the patient's question clearly and concisely.\n"
        "- Provide accurate, evidence-based medical information only.\n"
        "- Avoid hallucinations. If unsure, explicitly state 'I don't know'.\n"
        "- Include tests, diagnosis, treatment, medication, side effects, precautions.\n"
        "- Do not repeat the question, avoid unnecessary symbols or extra words.\n"
        "- Keep language clear, simple, and patient-friendly."
- Answer the question in {lang} only 
    )
    prompt = f"""
{prefix}
Patient question:
{user_query}
Instructions:
{rules}
Answer in the same language as the question:
"""
    return prompt
def clean_text(text: str) -> str:
    """
    Remove junk, repeated sentences, unnecessary symbols, ensure safe medical output.
    """
    text = text.replace('\\"', '"').replace("\\'", "'").replace("\\n", " ").replace("\\", " ")
    text = re.sub(r'["“”‘’]+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # Deduplicate lines
    lines = []
    seen = set()
    for line in re.split(r'(?<=[।!?])\s+', text):
        l = line.strip()
        if len(l) > 4 and l not in seen:
            seen.add(l)
            lines.append(l)
    text = " ".join(lines)
    # Limit excessively long outputs
    if len(text) > 600:
        text = text[:600].rsplit(" ", 1)[0]
    # Fallback for unclear answers
    if len(text) < 3:
        return "I don't know."
    return text.strip()
def generate_answer(user_query: str) -> str:
    """
    Generate answer with dynamic language detection and safe medical rules.
    """
    prompt = build_prompt(user_query)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(device)
    with torch.no_grad():
        try:
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                min_new_tokens=config.MIN_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                repetition_penalty=config.REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
        except torch.cuda.OutOfMemoryError:
            print("⚠ GPU OOM — switching to CPU temporarily...")
            model.to("cpu")
            torch.cuda.empty_cache()
            outputs = model.generate(
                **inputs,
                max_new_tokens=config.MAX_NEW_TOKENS,
                min_new_tokens=config.MIN_NEW_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                repetition_penalty=config.REPETITION_PENALTY,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            model.to(device)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Remove echoes of prompt
    decoded = re.sub(r'^(.*?Answer in the same language as the question:)', '', decoded, flags=re.DOTALL|re.IGNORECASE)
    return clean_text(decoded)
def cleanup():
    torch.cuda.empty_cache()
    gc.collect()
# ==================== MAIN PROCESS ====================
print(f"\n🔹 Processing questions from: {EXTRACT_PATH}")
for lang_folder in os.listdir(EXTRACT_PATH):
    lang_path = os.path.join(EXTRACT_PATH, lang_folder)
    if not os.path.isdir(lang_path):
        continue
    qna_root = os.path.join(lang_path, "QnA")
    if not os.path.exists(qna_root):
        print(f"⚠ No QnA folder in {lang_folder}, skipping...")
        continue
    for root, _, files in os.walk(qna_root):
        for f in files:
            if not f.endswith(".json"):
                continue
            json_path = os.path.join(root, f)
            rel_path = os.path.relpath(root, EXTRACT_PATH)
            drive_dir = os.path.join(DRIVE_OUTPUT, rel_path)
            drive_file = os.path.join(drive_dir, f)
            if os.path.exists(drive_file):
                print(f"✅ Already processed: {drive_file}")
                continue
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
            questions = data.get("questions", [])
            for i in tqdm(range(0, len(questions), config.BATCH_SIZE), desc=f"🧠 Processing {f}", leave=False):
                batch = questions[i:i + config.BATCH_SIZE]
                for q in batch:
                    query = q.get("question", "").strip()
                    if query:
                        try:
                            ans = generate_answer(query)
                            q["answer"] = ans
                        except Exception as e:
                            print(f"⚠ Error generating answer: {e}")
                            q["answer"] = "I don't know."
                    else:
                        q["answer"] = "I don't know."
                cleanup()
            # Save locally
            local_dir = os.path.join(LOCAL_OUTPUT, rel_path)
            os.makedirs(local_dir, exist_ok=True)
            local_file = os.path.join(local_dir, f)
            with open(local_file, "w", encoding="utf-8") as out_f:
                json.dump(data, out_f, ensure_ascii=False, indent=2)
            # Copy to Drive
            os.makedirs(drive_dir, exist_ok=True)
            shutil.copy2(local_file, drive_file)
            cleanup()
print("\n✅✅ All questions processed — outputs are clean, multilingual, safe, and ready in Drive.")



# ==================== MAIN LOOP (Only process remaining languages) ====================
already_trained = ['English', 'Assamese', 'Bangla', 'Gujarati', 'Kannada', 'Tamil']
remaining_languages = [lang for lang in languages if lang not in already_trained]
for lang in remaining_languages:
    print(f"\n🔹 Processing language: {lang}")
    qna_path = os.path.join(EXTRACT_PATH, lang, "QnA")
    if not os.path.exists(qna_path):
        print(f"⚠️ No QnA folder for {lang}, skipping...")
        continue
    json_files = [f for f in os.listdir(qna_path) if f.endswith(".json")]
    for jf in tqdm(json_files, desc=f"{lang} JSONs"):
        file_path = os.path.join(qna_path, jf)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        questions_with_answers = []
        for q in data.get("questions", []):
            query_text = q.get("question", "")
            context_items = retrieve_context(query_text, lang, top_k=5)
            answer_text = generate_answer(query_text, lang, context_items)
            questions_with_answers.append({
                "question": query_text,
                "answer": answer_text
            })
        save_answer(lang, jf, questions_with_answers)
print("\n✅ Remaining languages processed and answers saved in output/<lang>/QnA/")


from google.colab import drive
from pathlib import Path
import shutil
# === Step 1: Mount Google Drive ===
drive.mount('/content/drive')
# === Step 2: Define source files/folders in Colab ===
local_items = [
    "faiss_indices",
    "sample_data",
    "test_data_release",
    "English_results.json",
    "Hindi_results.json",
    "faiss_indices.zip",
    "test_data_release.zip"
]
# Destination folder in Google Drive
drive_folder = Path("/content/drive/MyDrive/NLPQNARAG")
drive_folder.mkdir(parents=True, exist_ok=True)
# === Step 3: Copy files/folders ===
for item in local_items:
    src = Path("/content") / item
    dst = drive_folder / item
    if src.exists():
        if src.is_dir():
            # If folder exists, copytree
            if dst.exists():
                shutil.rmtree(dst)  # Remove existing folder in Drive
            shutil.copytree(src, dst)
            print(f"📂 Folder copied: {item}")
        else:
            # If file, copy
            shutil.copy2(src, dst)
            print(f"📄 File copied: {item}")
    else:
        print(f"⚠️ Not found: {item}")
print("\n✅ All files and folders copied to Google Drive under NLPQNARAG")




# # FULL CODE
# """
# ROBUST MEDICAL RAG SYSTEM
# ========================
# A production-grade multilingual medical Q&A system with:
# - Confidence-based retrieval
# - Hallucination prevention
# - Safety guardrails
# - Comprehensive logging
# - Answer validation
# """
# # ============================================================================
# # SECTION 1: IMPORTS AND DEPENDENCIES
# # ============================================================================
# from google.colab import drive
# drive.mount('/content/drive')
# # Install required packages
# !pip install -q sentence-transformers transformers langdetect accelerate faiss-cpu rouge-score bert-score
# import os
# import json
# import datetime
# import re
# import logging
# import hashlib
# import numpy as np
# import torch
# import faiss
# from typing import Dict, List, Tuple, Optional
# from dataclasses import dataclass
# from tqdm import tqdm
# from sentence_transformers import SentenceTransformer, CrossEncoder, util
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from langdetect import detect, LangDetectException
# from sklearn.metrics.pairwise import cosine_similarity
# print("✅ All imports successful")
# # ============================================================================
# # SECTION 2: CONFIGURATION AND CONSTANTS
# # ============================================================================
# @dataclass
# class Config:
#     """Central configuration for the RAG system"""
#     # Paths
#     BASE_DIR: str = "/content/drive/MyDrive/NLP_A14_Health/combined_qns"
#     FAISS_DIR: str = "/content/faiss_indices/"
#     RESULTS_DIR: str = "/content/drive/MyDrive/NLP_A14_Health/results/"
#     LOG_DIR: str = "/content/drive/MyDrive/NLP_A14_Health/logs/"
#     # Models
#     EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
#     GENERATION_MODEL: str = "bigscience/bloomz-3b"  # Upgraded from 560m
#     RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#     # Retrieval parameters
#     TOP_K_RETRIEVAL: int = 10
#     TOP_K_RERANK: int = 3
#     RELEVANCE_THRESHOLD: float = 0.45  # Minimum similarity score
#     # Generation parameters
#     MAX_NEW_TOKENS: int = 256
#     TEMPERATURE: float = 0.7
#     NUM_BEAMS: int = 5
#     # Cache settings
#     CACHE_SIZE: int = 1000
#     # Safety thresholds
#     MIN_CONFIDENCE: float = 0.5
#     HALLUCINATION_THRESHOLD: float = 0.3
# # Initialize configuration
# config = Config()
# # Create directories
# for directory in [config.FAISS_DIR, config.RESULTS_DIR, config.LOG_DIR]:
#     os.makedirs(directory, exist_ok=True)
# print("✅ Configuration initialized")
# # ============================================================================
# # SECTION 3: LOGGING SYSTEM
# # ============================================================================
# class RAGLogger:
#     """Comprehensive logging system for monitoring and debugging"""
#     def __init__(self, log_dir: str):
#         self.log_dir = log_dir
#         self.logger = logging.getLogger("MedicalRAG")
#         self.logger.setLevel(logging.INFO)
#         # File handler with date
#         log_file = os.path.join(
#             log_dir,
#             f"rag_{datetime.datetime.now().strftime('%Y%m%d')}.log"
#         )
#         fh = logging.FileHandler(log_file, encoding='utf-8')
#         fh.setLevel(logging.INFO)
#         # Console handler
#         ch = logging.StreamHandler()
#         ch.setLevel(logging.WARNING)
#         # Formatter
#         formatter = logging.Formatter(
#             '%(asctime)s | %(levelname)s | %(message)s',
#             datefmt='%Y-%m-%d %H:%M:%S'
#         )
#         fh.setFormatter(formatter)
#         ch.setFormatter(formatter)
#         self.logger.addHandler(fh)
#         self.logger.addHandler(ch)
#         self.logger.info("=" * 80)
#         self.logger.info("RAG SYSTEM INITIALIZED")
#         self.logger.info("=" * 80)
#     def log_query(self, query: str, lang: str, status: str, **kwargs):
#         """Log incoming query with metadata"""
#         self.logger.info(
#             f"QUERY | Lang: {lang} | Status: {status} | "
#             f"Query: {query[:100]}... | {kwargs}"
#         )
#     def log_retrieval(self, lang: str, num_retrieved: int, scores: List[float]):
#         """Log retrieval results"""
#         avg_score = np.mean(scores) if scores else 0
#         self.logger.info(
#             f"RETRIEVAL | Lang: {lang} | Retrieved: {num_retrieved} | "
#             f"Avg Score: {avg_score:.3f}"
#         )
#     def log_generation(self, answer_length: int, confidence: float):
#         """Log answer generation"""
#         self.logger.info(
#             f"GENERATION | Length: {answer_length} chars | "
#             f"Confidence: {confidence:.3f}"
#         )
#     def log_safety(self, query: str, flag_type: str, reason: str):
#         """Log safety flags"""
#         self.logger.warning(
#             f"SAFETY FLAG | Type: {flag_type} | Reason: {reason} | "
#             f"Query: {query[:50]}..."
#         )
#     def log_error(self, error_type: str, details: str, query: str = ""):
#         """Log errors"""
#         self.logger.error(
#             f"ERROR | Type: {error_type} | Details: {details} | "
#             f"Query: {query[:50] if query else 'N/A'}"
#         )
#     def log_cache(self, action: str, query: str):
#         """Log cache operations"""
#         self.logger.debug(f"CACHE | Action: {action} | Query: {query[:50]}...")
# # Initialize logger
# logger = RAGLogger(config.LOG_DIR)
# print("✅ Logging system initialized")
# # ============================================================================
# # SECTION 4: SAFETY AND VALIDATION MODULES
# # ============================================================================
# class SafetyGuard:
#     """Safety checks for medical queries and answers"""
#     # Harmful query patterns
#     HARMFUL_PATTERNS = [
#         r"how to (kill|harm|suicide|die)",
#         r"(overdose|lethal dose|fatal amount)",
#         r"without (doctor|prescription|medical help)",
#         r"(hide|fake|forge) (symptoms|test results|prescription)",
#         r"(abortion|terminate pregnancy) at home",
#         r"how to get high",
#         r"self (harm|mutilate|injury)",
#         r"poison someone",
#     ]
#     # Unsafe advice patterns
#     UNSAFE_ADVICE = [
#         "stop taking your medication",
#         "ignore your doctor",
#         "don't see a doctor",
#         "avoid medical help",
#         "perform surgery",
#         "inject yourself",
#         "increase dosage significantly",
#     ]
#     # Emergency keywords
#     EMERGENCY_KEYWORDS = [
#         "severe pain", "can't breathe", "chest pain", "unconscious",
#         "heavy bleeding", "overdose", "poisoning", "stroke symptoms",
#         "heart attack", "severe allergic reaction", "anaphylaxis"
#     ]
#     @staticmethod
#     def is_safe_query(query: str) -> Tuple[bool, str]:
#         """Check if query is safe to process"""
#         query_lower = query.lower()
#         # Check harmful patterns
#         for pattern in SafetyGuard.HARMFUL_PATTERNS:
#             if re.search(pattern, query_lower):
#                 return False, "HARMFUL_INTENT"
#         return True, "SAFE"
#     @staticmethod
#     def is_emergency(query: str) -> bool:
#         """Detect emergency situations"""
#         query_lower = query.lower()
#         return any(keyword in query_lower for keyword in SafetyGuard.EMERGENCY_KEYWORDS)
#     @staticmethod
#     def filter_dangerous_advice(answer: str, query: str) -> Tuple[str, bool]:
#         """Filter out dangerous medical advice"""
#         answer_lower = answer.lower()
#         # Check for unsafe advice
#         for phrase in SafetyGuard.UNSAFE_ADVICE:
#             if phrase in answer_lower:
#                 return (
#                     "I cannot provide this advice. Please consult your healthcare provider immediately.",
#                     True
#                 )
#         # Check if emergency query got non-urgent answer
#         if SafetyGuard.is_emergency(query) and "emergency" not in answer_lower and "immediately" not in answer_lower:
#             return (
#                 "🚨 EMERGENCY: This appears to be a medical emergency. "
#                 "Please call emergency services (ambulance) immediately or go to the nearest emergency room. "
#                 "Do not wait for medical information.",
#                 True
#             )
#         return answer, False
#     @staticmethod
#     def get_emergency_response() -> str:
#         """Standard emergency response"""
#         return (
#             "🚨 MEDICAL EMERGENCY DETECTED\n\n"
#             "This appears to be a medical emergency. Please:\n"
#             "1. Call emergency services (ambulance) IMMEDIATELY\n"
#             "2. Go to the nearest emergency room\n"
#             "3. Do NOT wait for online medical advice\n\n"
#             "If someone is unconscious or not breathing, start CPR if trained."
#         )
# class DisclaimerManager:
#     """Manages medical disclaimers for different query types"""
#     DISCLAIMERS = {
#         "medication": (
#             "\n\n⚠️ MEDICATION INFORMATION: This information is for educational purposes only. "
#             "Always consult a doctor or pharmacist before starting, stopping, or changing medications. "
#             "Never adjust medication dosages without professional guidance."
#         ),
#         "diagnosis": (
#             "\n\n⚠️ DIAGNOSTIC INFORMATION: This is not a medical diagnosis. "
#             "Only a qualified healthcare professional can diagnose medical conditions. "
#             "If you're experiencing symptoms, please consult a doctor."
#         ),
#         "emergency": (
#             "\n\n🚨 EMERGENCY: If this is an emergency, call emergency services immediately."
#         ),
#         "treatment": (
#             "\n\n⚠️ TREATMENT INFORMATION: Treatment plans should be personalized by healthcare professionals. "
#             "Do not self-treat based on general information. Consult your doctor."
#         ),
#         "general": (
#             "\n\nℹ️ MEDICAL DISCLAIMER: This information is based on general medical knowledge "
#             "and should not replace professional medical advice. Always consult a qualified "
#             "healthcare provider for personalized medical guidance."
#         )
#     }
#     @staticmethod
#     def detect_category(query: str) -> str:
#         """Detect query category for appropriate disclaimer"""
#         query_lower = query.lower()
#         if re.search(r"\b(medicine|drug|pill|prescription|medication|tablet)\b", query_lower):
#             return "medication"
#         if re.search(r"\b(diagnose|do i have|symptoms of|is it|could it be)\b", query_lower):
#             return "diagnosis"
#         if SafetyGuard.is_emergency(query):
#             return "emergency"
#         if re.search(r"\b(treat|treatment|cure|therapy|procedure)\b", query_lower):
#             return "treatment"
#         return "general"
#     @staticmethod
#     def add_disclaimer(answer: str, query: str) -> str:
#         """Add appropriate disclaimer to answer"""
#         category = DisclaimerManager.detect_category(query)
#         disclaimer = DisclaimerManager.DISCLAIMERS.get(category, DisclaimerManager.DISCLAIMERS["general"])
#         return answer + disclaimer
# print("✅ Safety modules initialized")
# # ============================================================================
# # SECTION 5: LANGUAGE DETECTION
# # ============================================================================
# class LanguageDetector:
#     """Automatic language detection with fallback"""
#     # Language code mapping
#     LANG_MAP = {
#         'en': 'English',
#         'hi': 'Hindi',
#         'bn': 'Bengali',
#         'te': 'Telugu',
#         'ta': 'Tamil',
#         'mr': 'Marathi',
#         'gu': 'Gujarati',
#         'kn': 'Kannada',
#         'ml': 'Malayalam',
#         'pa': 'Punjabi',
#         'as': 'Assamese',
#         'or': 'Odia',
#         'ur': 'Urdu',
#     }
#     @staticmethod
#     def detect(query: str, available_languages: List[str]) -> str:
#         """Detect language with fallback to English"""
#         try:
#             lang_code = detect(query)
#             lang_name = LanguageDetector.LANG_MAP.get(lang_code, 'English')
#             # Check if detected language is available
#             if lang_name in available_languages:
#                 logger.log_query(query, lang_name, "LANG_DETECTED")
#                 return lang_name
#             else:
#                 logger.log_query(query, "English", "LANG_FALLBACK")
#                 return 'English'
#         except LangDetectException:
#             logger.log_error("LANG_DETECTION", "Failed to detect language", query)
#             return 'English'
# print("✅ Language detector initialized")
# # ============================================================================
# # SECTION 6: CACHING SYSTEM
# # ============================================================================
# class QueryCache:
#     """LRU cache for query results"""
#     def __init__(self, max_size: int = 1000):
#         self.cache = {}
#         self.access_order = []
#         self.max_size = max_size
#         logger.logger.info(f"Cache initialized with max_size={max_size}")
#     def _get_cache_key(self, query: str, lang: str) -> str:
#         """Generate cache key from query and language"""
#         content = f"{query.lower().strip()}_{lang}".encode('utf-8')
#         return hashlib.md5(content).hexdigest()
#     def get(self, query: str, lang: str) -> Optional[Dict]:
#         """Retrieve cached result"""
#         key = self._get_cache_key(query, lang)
#         if key in self.cache:
#             # Update access order (LRU)
#             self.access_order.remove(key)
#             self.access_order.append(key)
#             logger.log_cache("HIT", query)
#             return self.cache[key]
#         logger.log_cache("MISS", query)
#         return None
#     def set(self, query: str, lang: str, result: Dict):
#         """Store result in cache"""
#         key = self._get_cache_key(query, lang)
#         # Evict oldest if cache is full
#         if len(self.cache) >= self.max_size and key not in self.cache:
#             oldest_key = self.access_order.pop(0)
#             del self.cache[oldest_key]
#         self.cache[key] = result
#         if key not in self.access_order:
#             self.access_order.append(key)
#         logger.log_cache("SET", query)
#     def clear(self):
#         """Clear cache"""
#         self.cache.clear()
#         self.access_order.clear()
#         logger.logger.info("Cache cleared")
# # Initialize cache
# query_cache = QueryCache(max_size=config.CACHE_SIZE)
# print("✅ Cache system initialized")
# # ============================================================================
# # SECTION 7: FAISS INDEX BUILDING (Run Once)
# # ============================================================================
# def build_faiss_indices():
#     """Build FAISS indices for all languages"""
#     logger.logger.info("Starting FAISS index building...")
#     # Initialize embedder
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
#     logger.logger.info(f"Embedder loaded on {device}")
#     def get_nlist(n_items):
#         """Choose number of clusters based on dataset size"""
#         if n_items < 1000:
#             return None
#         elif n_items < 10000:
#             return 64
#         elif n_items < 50000:
#             return 256
#         else:
#             return 1024
#     # Process each language file
#     files = [f for f in os.listdir(config.BASE_DIR) if f.endswith(".json")]
#     for file in tqdm(files, desc="Building indices"):
#         lang = file.replace("_combined_qns.json", "")
#         logger.logger.info(f"Processing {lang}")
#         # Load data
#         path = os.path.join(config.BASE_DIR, file)
#         with open(path, "r", encoding="utf-8") as f:
#             data = json.load(f)
#         # Combine question and answer for embedding
#         texts = [qa["question"] + " " + qa["answer"] for qa in data]
#         n_items = len(texts)
#         logger.logger.info(f"{lang}: {n_items} items")
#         # Create embeddings
#         embeddings = embedder.encode(
#             texts,
#             batch_size=64,
#             show_progress_bar=True,
#             convert_to_numpy=True,
#             normalize_embeddings=True
#         ).astype("float32")
#         dim = embeddings.shape[1]
#         nlist = get_nlist(n_items)
#         # Build index
#         if nlist is None:
#             logger.logger.info(f"{lang}: Using FlatIP index")
#             index = faiss.IndexFlatIP(dim)
#             index.add(embeddings)
#         else:
#             logger.logger.info(f"{lang}: Using IVF index with nlist={nlist}")
#             quantizer = faiss.IndexFlatIP(dim)
#             index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
#             index.train(embeddings)
#             index.add(embeddings)
#         # Save index and mapping
#         faiss.write_index(index, f"{config.FAISS_DIR}/{lang}_index.faiss")
#         np.save(f"{config.FAISS_DIR}/{lang}_map.npy", np.arange(n_items))
#         logger.logger.info(f"✅ Saved {lang} index")
#     logger.logger.info("All FAISS indices built successfully!")
# # Uncomment to build indices (run once)
# # build_faiss_indices()
# print("✅ FAISS index builder ready")
# # ============================================================================
# # SECTION 8: RETRIEVAL SYSTEM
# # ============================================================================
# class HybridRetriever:
#     """Advanced retrieval with confidence scoring and re-ranking"""
#     def __init__(self):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         self.embedder = SentenceTransformer(config.EMBEDDING_MODEL, device=self.device)
#         self.reranker = CrossEncoder(config.RERANKER_MODEL)
#         # Load all indices
#         self.indices = {}
#         self.data_maps = {}
#         self.lang_data = {}
#         self._load_indices()
#         logger.logger.info(f"Retriever initialized with {len(self.indices)} languages")
#     def _load_indices(self):
#         """Load all FAISS indices and data"""
#         for file in os.listdir(config.FAISS_DIR):
#             if file.endswith("_index.faiss"):
#                 lang = file.replace("_index.faiss", "")
#                 # Load index
#                 index_path = os.path.join(config.FAISS_DIR, file)
#                 self.indices[lang] = faiss.read_index(index_path)
#                 # Load mapping
#                 map_path = os.path.join(config.FAISS_DIR, f"{lang}_map.npy")
#                 self.data_maps[lang] = np.load(map_path, allow_pickle=True)
#                 # Load data
#                 data_path = os.path.join(config.BASE_DIR, f"{lang}_combined_qns.json")
#                 with open(data_path, "r", encoding="utf-8") as f:
#                     self.lang_data[lang] = json.load(f)
#         logger.logger.info(f"Loaded indices for languages: {list(self.indices.keys())}")
#     def retrieve_with_confidence(
#         self,
#         query: str,
#         lang: str,
#         top_k: int = None
#     ) -> Tuple[List[Dict], List[float], str]:
#         """
#         Retrieve relevant contexts with confidence scoring
#         Returns:
#             contexts: List of retrieved Q&A pairs
#             scores: Similarity scores
#             status: HIGH_CONFIDENCE, MEDIUM_CONFIDENCE, or LOW_CONFIDENCE
#         """
#         if top_k is None:
#             top_k = config.TOP_K_RETRIEVAL
#         if lang not in self.indices:
#             logger.log_error("RETRIEVAL", f"No index for language: {lang}", query)
#             return [], [], "NO_INDEX"
#         # Embed query
#         query_emb = self.embedder.encode(
#             [query],
#             convert_to_numpy=True,
#             normalize_embeddings=True
#         ).astype("float32")
#         # Search FAISS
#         scores, indices = self.indices[lang].search(query_emb, top_k)
#         scores = scores[0]  # Flatten
#         indices = indices[0]
#         # Filter by threshold
#         valid_mask = scores >= config.RELEVANCE_THRESHOLD
#         valid_scores = scores[valid_mask]
#         valid_indices = indices[valid_mask]
#         # Get contexts
#         contexts = [
#             self.lang_data[lang][int(self.data_maps[lang][i])]
#             for i in valid_indices
#         ]
#         # Determine confidence status
#         if len(contexts) == 0:
#             status = "LOW_CONFIDENCE"
#         elif len(contexts) >= 3 and np.mean(valid_scores) >= 0.6:
#             status = "HIGH_CONFIDENCE"
#         else:
#             status = "MEDIUM_CONFIDENCE"
#         logger.log_retrieval(lang, len(contexts), valid_scores.tolist())
#         return contexts, valid_scores.tolist(), status
#     def rerank_contexts(
#         self,
#         query: str,
#         contexts: List[Dict],
#         top_n: int = None
#     ) -> List[Dict]:
#         """Re-rank contexts using cross-encoder"""
#         if top_n is None:
#             top_n = config.TOP_K_RERANK
#         if len(contexts) == 0:
#             return []
#         # Create query-document pairs
#         pairs = [
#             [query, f"{ctx['question']} {ctx['answer']}"]
#             for ctx in contexts
#         ]
#         # Get reranker scores
#         rerank_scores = self.reranker.predict(pairs)
#         # Sort by score
#         ranked_pairs = sorted(
#             zip(rerank_scores, contexts),
#             key=lambda x: x[0],
#             reverse=True
#         )
#         # Return top N
#         reranked = [ctx for _, ctx in ranked_pairs[:top_n]]
#         logger.logger.info(f"Re-ranked {len(contexts)} to {len(reranked)} contexts")
#         return reranked
#     def deduplicate_contexts(self, contexts: List[Dict]) -> List[Dict]:
#         """Remove near-duplicate contexts"""
#         if len(contexts) <= 1:
#             return contexts
#         unique = []
#         seen_embeddings = []
#         for ctx in contexts:
#             ctx_text = ctx['answer']
#             ctx_emb = self.embedder.encode([ctx_text])
#             is_duplicate = False
#             for seen_emb in seen_embeddings:
#                 similarity = cosine_similarity(ctx_emb, seen_emb)[0][0]
#                 if similarity > 0.95:  # 95% similar threshold
#                     is_duplicate = True
#                     break
#             if not is_duplicate:
#                 unique.append(ctx)
#                 seen_embeddings.append(ctx_emb)
#         if len(unique) < len(contexts):
#             logger.logger.info(f"Deduplicated {len(contexts)} to {len(unique)} contexts")
#         return unique
# # Initialize retriever
# retriever = HybridRetriever()
# print("✅ Retrieval system initialized")
# # ============================================================================
# # SECTION 9: ANSWER GENERATION SYSTEM
# # ============================================================================
# class AnswerGenerator:
#     """Generate and validate answers using LLM"""
#     def __init__(self):
#         self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
#         # Load generation model
#         logger.logger.info(f"Loading generation model: {config.GENERATION_MODEL}")
#         self.tokenizer = AutoTokenizer.from_pretrained(config.GENERATION_MODEL)
#         self.model = AutoModelForCausalLM.from_pretrained(config.GENERATION_MODEL).to(self.device)
#         logger.logger.info(f"Generator loaded on {self.device}")
#     def _build_prompt(self, query: str, contexts: List[Dict], lang: str) -> str:
#         """Build comprehensive prompt with safety instructions"""
#         # Format contexts with source numbers
#         context_text = "\n\n".join([
#             f"[Source {i+1}]\nQuestion: {ctx['question']}\nAnswer: {ctx['answer']}"
#             for i, ctx in enumerate(contexts)
#         ])
#         prompt = f"""You are a medical information assistant. Your role is to provide accurate information based STRICTLY on the provided context.
# CRITICAL RULES:
# 1. Answer ONLY using information from the Context below
# 2. If the Context doesn't contain enough information to answer, respond: "I don't have sufficient information in my knowledge base to answer this question accurately."
# 3. NEVER invent or assume medical information not present in the Context
# 4. Stay factual and avoid speculation
# 5. Respond in {lang} language
# 6. Keep answers clear, concise, and focused
# Context:
# {context_text}
# Question: {query}
# Answer (following all rules above):"""
#         return prompt
#     def generate(
#         self,
#         query: str,
#         contexts: List[Dict],
#         lang: str
#     ) -> Tuple[str, Dict]:
#         """
#         Generate answer from contexts
#         Returns:
#             answer: Generated text
#             metadata: Generation metadata (tokens, time, etc.)
#         """
#         if len(contexts) == 0:
#             return self._generate_no_context_response(query), {}
#         # Build prompt
#         prompt = self._build_prompt(query, contexts, lang)
#         # Tokenize
#         inputs = self.tokenizer(
#             prompt,
#             return_tensors="pt",
#             truncation=True,
#             padding=True,
#             max_length=1024
#         ).to(self.device)
#         # Generate
#         start_time = datetime.datetime.now()
#         with torch.no_grad():
#             outputs = self.model.generate(
#                 **inputs,
#                 max_new_tokens=config.MAX_NEW_TOKENS,
#                 num_beams=config.NUM_BEAMS,
#                 temperature=config.TEMPERATURE,
#                 early_stopping=True,
#                 do_sample=False,  # Deterministic for medical accuracy
#                 pad_token_id=self.tokenizer.eos_token_id
#             )
#         generation_time = (datetime.datetime.now() - start_time).total_seconds()
#         # Decode
#         full_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
#         # Extract answer (remove prompt)
#         answer = full_output.split("Answer (following all rules above):")[-1].strip()
#         # Clean up
#         answer = self._clean_answer(answer)
#         # Metadata
#         metadata = {
#             "generation_time": generation_time,
#             "num_tokens": len(outputs[0]),
#             "num_contexts": len(contexts)
#         }
#         logger.log_generation(len(answer), 0.0)  # Confidence calculated later
#         return answer, metadata
#     def _clean_answer(self, answer: str) -> str:
#         """Clean and format answer"""
#         # Remove any remaining prompt artifacts
#         answer = re.sub(r"^(Answer|Response|A:)\s*:?\s*", "", answer, flags=re.IGNORECASE)
#         # Remove incomplete sentences at the end
#         sentences = answer.split('.')
#         if len(sentences) > 1 and len(sentences[-1].strip()) < 10:
#             answer = '.'.join(sentences[:-1]) + '.'
#         # Remove extra whitespace
#         answer = ' '.join(answer.split())
#         return answer.strip()
#     def _generate_no_context_response(self, query: str) -> str:
#         """Generate helpful response when no context is found"""
#         return (
#             "I don't have sufficient information in my knowledge base to answer this specific question accurately. "
#             "\n\nI recommend:\n"
#             "1. Consulting a qualified healthcare professional who can provide personalized advice\n"
#             "2. Checking reputable medical websites or resources\n"
#             "3. Contacting a medical helpline for guidance\n\n"
#             "For medical emergencies, please call emergency services immediately."
#         )
#     def validate_answer(
#         self,
#         answer: str,
#         contexts: List[Dict],
#         query: str
#     ) -> Tuple[bool, str, float]:
#         """
#         Validate answer for quality and safety
#         Returns:
#             is_valid: Boolean
#             reason: Validation result reason
#             confidence: Confidence score
#         """
#         # Check 1: Minimum length
#         if len(answer.split()) < 10:
#             return False, "TOO_SHORT", 0.0
#         # Check 2: Not just refusal
#         refusal_patterns = [
#             "i don't have", "cannot answer", "insufficient information",
#             "not enough information", "unable to provide"
#         ]
#         is_refusal = any(pattern in answer.lower() for pattern in refusal_patterns)
#         if is_refusal:
#             return True, "VALID_REFUSAL", 0.5
#         # Check 3: Context overlap (keyword-based grounding)
#         context_keywords = set()
#         for ctx in contexts:
#             words = re.findall(r'\w+', ctx['answer'].lower())
#             context_keywords.update([w for w in words if len(w) > 4])
#         answer_keywords = set(re.findall(r'\w+', answer.lower()))
#         answer_keywords = {w for w in answer_keywords if len(w) > 4}
#         overlap = len(context_keywords & answer_keywords)
#         overlap_ratio = overlap / max(len(answer_keywords), 1)
#         if overlap < 3:
#             return False, "NO_CONTEXT_OVERLAP", 0.2
#         # Check 4: Hallucination detection (simple heuristic)
#         # More sophisticated: Use NLI model (commented out for performance)
#         # has_hallucination = self._detect_hallucination_nli(answer, contexts)
#         # Calculate confidence
#         confidence = min(overlap_ratio * 1.5, 1.0)
#         return True, "VALID", confidence
#     def _detect_hallucination_nli(self, answer: str, contexts: List[Dict]) -> bool:
#         """Use NLI model to detect hallucinations (optional, resource-intensive)"""
#         try:
#             # Load NLI model (do this once in __init__ for production)
#             nli_classifier = pipeline(
#                 "text-classification",
#                 model="microsoft/deberta-v3-base-mnli-fever-anli",
#                 device=0 if self.device == 'cuda' else -1
#             )
#             context_text = " ".join([ctx['answer'] for ctx in contexts])
#             # Check if answer is entailed by context
#             result = nli_classifier(f"{context_text} [SEP] {answer}")[0]
#             # If contradiction or neutral, likely hallucination
#             if result['label'] in ['CONTRADICTION', 'NEUTRAL']:
#                 return True
#             return False
#         except Exception as e:
#             logger.log_error("HALLUCINATION_CHECK", str(e))
#             return False  # Fail open
# # Initialize generator
# generator = AnswerGenerator()
# print("✅ Answer generation system initialized")
# # ============================================================================
# # SECTION 10: CONFIDENCE SCORING
# # ============================================================================
# class ConfidenceScorer:
#     """Calculate multi-factor confidence scores"""
#     @staticmethod
#     def calculate_confidence(
#         retrieval_scores: List[float],
#         answer: str,
#         contexts: List[Dict],
#         validation_confidence: float
#     ) -> float:
#         """
#         Calculate overall confidence score
#         Factors:
#         1. Retrieval quality (40%)
#         2. Answer length appropriateness (15%)
#         3. Context overlap (25%)
#         4. Validation confidence (20%)
#         """
#         if len(retrieval_scores) == 0:
#             return 0.0
#         # Factor 1: Retrieval quality
#         avg_retrieval_score = np.mean(retrieval_scores)
#         retrieval_factor = min(avg_retrieval_score / 0.8, 1.0)  # Normalize to 0.8 as max
#         # Factor 2: Answer length (optimal range: 50-300 words)
#         word_count = len(answer.split())
#         if 50 <= word_count <= 300:
#             length_factor = 1.0
#         elif word_count < 50:
#             length_factor = word_count / 50
#         else:
#             length_factor = max(0.5, 1.0 - (word_count - 300) / 500)
#         # Factor 3: Context overlap (keyword-based)
#         context_keywords = set()
#         for ctx in contexts:
#             words = re.findall(r'\w+', ctx['answer'].lower())
#             context_keywords.update([w for w in words if len(w) > 4])
#         answer_keywords = set(re.findall(r'\w+', answer.lower()))
#         answer_keywords = {w for w in answer_keywords if len(w) > 4}
#         if len(answer_keywords) > 0:
#             overlap_ratio = len(context_keywords & answer_keywords) / len(answer_keywords)
#         else:
#             overlap_ratio = 0.0
#         # Factor 4: Validation confidence
#         validation_factor = validation_confidence
#         # Weighted combination
#         confidence = (
#             0.40 * retrieval_factor +
#             0.15 * length_factor +
#             0.25 * overlap_ratio +
#             0.20 * validation_factor
#         )
#         return round(confidence, 3)
#     @staticmethod
#     def interpret_confidence(confidence: float) -> str:
#         """Interpret confidence level"""
#         if confidence >= 0.7:
#             return "HIGH"
#         elif confidence >= 0.5:
#             return "MEDIUM"
#         else:
#             return "LOW"
# print("✅ Confidence scorer initialized")
# # ============================================================================
# # SECTION 11: MAIN RAG PIPELINE
# # ============================================================================
# class RobustMedicalRAG:
#     """
#     Main RAG pipeline with all safety and quality features
#     """
#     def __init__(self):
#         self.retriever = retriever
#         self.generator = generator
#         self.cache = query_cache
#         self.safety = SafetyGuard()
#         self.disclaimer_mgr = DisclaimerManager()
#         self.lang_detector = LanguageDetector()
#         self.confidence_scorer = ConfidenceScorer()
#         logger.logger.info("=" * 80)
#         logger.logger.info("ROBUST MEDICAL RAG SYSTEM READY")
#         logger.logger.info("=" * 80)
#     def answer(
#         self,
#         query: str,
#         lang: str = None,
#         skip_cache: bool = False
#     ) -> Dict:
#         """
#         Main entry point for answering queries
#         Args:
#             query: User question
#             lang: Language (auto-detected if None)
#             skip_cache: Force fresh generation
#         Returns:
#             Dictionary with answer, confidence, sources, metadata
#         """
#         start_time = datetime.datetime.now()
#         try:
#             # ========== STEP 1: SAFETY CHECK ==========
#             is_safe, reason = self.safety.is_safe_query(query)
#             if not is_safe:
#                 logger.log_safety(query, reason, "Harmful query detected")
#                 return self._create_response(
#                     answer="I cannot provide information on this topic. Please contact a healthcare professional or emergency services if needed.",
#                     confidence=0.0,
#                     status="BLOCKED_UNSAFE",
#                     metadata={"block_reason": reason}
#                 )
#             # Check for emergency
#             if self.safety.is_emergency(query):
#                 logger.log_safety(query, "EMERGENCY", "Emergency query detected")
#                 return self._create_response(
#                     answer=self.safety.get_emergency_response(),
#                     confidence=1.0,
#                     status="EMERGENCY_RESPONSE",
#                     metadata={}
#                 )
#             # ========== STEP 2: LANGUAGE DETECTION ==========
#             if lang is None:
#                 available_langs = list(self.retriever.indices.keys())
#                 lang = self.lang_detector.detect(query, available_langs)
#             logger.log_query(query, lang, "PROCESSING")
#             # ========== STEP 3: CACHE CHECK ==========
#             if not skip_cache:
#                 cached_result = self.cache.get(query, lang)
#                 if cached_result:
#                     logger.log_query(query, lang, "CACHE_HIT")
#                     cached_result['metadata']['from_cache'] = True
#                     return cached_result
#             # ========== STEP 4: RETRIEVAL ==========
#             contexts, retrieval_scores, retrieval_status = self.retriever.retrieve_with_confidence(
#                 query, lang
#             )
#             # Handle low confidence retrieval
#             if retrieval_status == "LOW_CONFIDENCE" or len(contexts) == 0:
#                 logger.log_query(query, lang, "LOW_RETRIEVAL_CONFIDENCE")
#                 answer = self.generator._generate_no_context_response(query)
#                 return self._create_response(
#                     answer=answer,
#                     confidence=0.3,
#                     status="NO_RELEVANT_CONTEXT",
#                     metadata={"retrieval_status": retrieval_status}
#                 )
#             # ========== STEP 5: DEDUPLICATION ==========
#             contexts = self.retriever.deduplicate_contexts(contexts)
#             # ========== STEP 6: RE-RANKING ==========
#             contexts = self.retriever.rerank_contexts(query, contexts)
#             # ========== STEP 7: ANSWER GENERATION ==========
#             answer, gen_metadata = self.generator.generate(query, contexts, lang)
#             # ========== STEP 8: ANSWER VALIDATION ==========
#             is_valid, validation_reason, validation_confidence = self.generator.validate_answer(
#                 answer, contexts, query
#             )
#             if not is_valid and validation_reason != "VALID_REFUSAL":
#                 logger.log_query(query, lang, f"VALIDATION_FAILED: {validation_reason}")
#                 answer = self.generator._generate_no_context_response(query)
#                 validation_confidence = 0.3
#             # ========== STEP 9: SAFETY FILTER ==========
#             answer, was_filtered = self.safety.filter_dangerous_advice(answer, query)
#             if was_filtered:
#                 logger.log_safety(query, "DANGEROUS_ADVICE", "Answer filtered")
#             # ========== STEP 10: ADD DISCLAIMER ==========
#             answer = self.disclaimer_mgr.add_disclaimer(answer, query)
#             # ========== STEP 11: CONFIDENCE CALCULATION ==========
#             confidence = self.confidence_scorer.calculate_confidence(
#                 retrieval_scores,
#                 answer,
#                 contexts,
#                 validation_confidence
#             )
#             confidence_level = self.confidence_scorer.interpret_confidence(confidence)
#             # ========== STEP 12: CREATE RESPONSE ==========
#             total_time = (datetime.datetime.now() - start_time).total_seconds()
#             response = self._create_response(
#                 answer=answer,
#                 confidence=confidence,
#                 confidence_level=confidence_level,
#                 status="SUCCESS",
#                 sources=contexts,
#                 metadata={
#                     "retrieval_status": retrieval_status,
#                     "num_contexts_retrieved": len(contexts),
#                     "retrieval_scores": retrieval_scores,
#                     "validation_reason": validation_reason,
#                     "generation_time": gen_metadata.get("generation_time", 0),
#                     "total_time": total_time,
#                     "language": lang,
#                     "was_safety_filtered": was_filtered,
#                     "from_cache": False
#                 }
#             )
#             # ========== STEP 13: CACHE RESULT ==========
#             self.cache.set(query, lang, response)
#             # ========== STEP 14: LOG SUCCESS ==========
#             logger.log_query(
#                 query, lang, "SUCCESS",
#                 confidence=confidence,
#                 time=total_time
#             )
#             return response
#         except Exception as e:
#             # Error handling
#             logger.log_error("PIPELINE_ERROR", str(e), query)
#             return self._create_response(
#                 answer="I encountered an error processing your question. Please try rephrasing or contact support.",
#                 confidence=0.0,
#                 status="ERROR",
#                 metadata={"error": str(e)}
#             )
#     def _create_response(
#         self,
#         answer: str,
#         confidence: float,
#         status: str,
#         confidence_level: str = None,
#         sources: List[Dict] = None,
#         metadata: Dict = None
#     ) -> Dict:
#         """Create standardized response dictionary"""
#         return {
#             "answer": answer,
#             "confidence": confidence,
#             "confidence_level": confidence_level or self.confidence_scorer.interpret_confidence(confidence),
#             "status": status,
#             "sources": sources or [],
#             "metadata": metadata or {},
#             "timestamp": datetime.datetime.now().isoformat()
#         }
#     def batch_answer(self, queries: List[Tuple[str, str]]) -> List[Dict]:
#         """
#         Process multiple queries efficiently
#         Args:
#             queries: List of (query, lang) tuples
#         Returns:
#             List of response dictionaries
#         """
#         results = []
#         for query, lang in tqdm(queries, desc="Processing batch"):
#             result = self.answer(query, lang)
#             results.append(result)
#         return results
# # Initialize main RAG system
# rag_system = RobustMedicalRAG()
# print("✅ Main RAG system initialized")
# # ============================================================================
# # SECTION 12: RESULT PERSISTENCE
# # ============================================================================
# class ResultManager:
#     """Manage saving and loading of results"""
#     def __init__(self, results_dir: str):
#         self.results_dir = results_dir
#         os.makedirs(results_dir, exist_ok=True)
#     def save_result(self, response: Dict, query: str, lang: str):
#         """Save individual query result"""
#         result = {
#             "query": query,
#             "language": lang,
#             "answer": response["answer"],
#             "confidence": response["confidence"],
#             "confidence_level": response["confidence_level"],
#             "status": response["status"],
#             "sources": response["sources"],
#             "metadata": response["metadata"],
#             "timestamp": response["timestamp"]
#         }
#         # Save to language-specific file
#         filepath = os.path.join(self.results_dir, f"{lang}_results.jsonl")
#         with open(filepath, "a", encoding="utf-8") as f:
#             f.write(json.dumps(result, ensure_ascii=False) + "\n")
#         logger.logger.info(f"Saved result for {lang} query")
#     def save_batch_results(self, results: List[Dict], queries: List[Tuple[str, str]]):
#         """Save batch results"""
#         for result, (query, lang) in zip(results, queries):
#             self.save_result(result, query, lang)
#     def load_results(self, lang: str) -> List[Dict]:
#         """Load all results for a language"""
#         filepath = os.path.join(self.results_dir, f"{lang}_results.jsonl")
#         if not os.path.exists(filepath):
#             return []
#         results = []
#         with open(filepath, "r", encoding="utf-8") as f:
#             for line in f:
#                 results.append(json.loads(line))
#         return results
#     def get_statistics(self) -> Dict:
#         """Get system statistics"""
#         stats = {
#             "total_queries": 0,
#             "by_language": {},
#             "by_status": {},
#             "avg_confidence": 0.0,
#             "confidence_distribution": {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
#         }
#         all_confidences = []
#         for lang_file in os.listdir(self.results_dir):
#             if not lang_file.endswith("_results.jsonl"):
#                 continue
#             lang = lang_file.replace("_results.jsonl", "")
#             results = self.load_results(lang)
#             stats["by_language"][lang] = len(results)
#             stats["total_queries"] += len(results)
#             for result in results:
#                 status = result.get("status", "UNKNOWN")
#                 stats["by_status"][status] = stats["by_status"].get(status, 0) + 1
#                 confidence = result.get("confidence", 0.0)
#                 all_confidences.append(confidence)
#                 conf_level = result.get("confidence_level", "LOW")
#                 stats["confidence_distribution"][conf_level] += 1
#         if all_confidences:
#             stats["avg_confidence"] = round(np.mean(all_confidences), 3)
#         return stats
# # Initialize result manager
# result_manager = ResultManager(config.RESULTS_DIR)
# print("✅ Result manager initialized")
# # ============================================================================
# # SECTION 13: EVALUATION METRICS
# # ============================================================================
# class EvaluationMetrics:
#     """Evaluate answer quality"""
#     @staticmethod
#     def calculate_metrics(generated: str, reference: str) -> Dict:
#         """
#         Calculate quality metrics
#         Note: Requires ground truth for proper evaluation
#         """
#         try:
#             from rouge_score import rouge_scorer
#             scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
#             scores = scorer.score(reference, generated)
#             return {
#                 "rouge1_f": scores['rouge1'].fmeasure,
#                 "rouge2_f": scores['rouge2'].fmeasure,
#                 "rougeL_f": scores['rougeL'].fmeasure
#             }
#         except ImportError:
#             logger.log_error("METRICS", "rouge_score not installed")
#             return {}
#     @staticmethod
#     def evaluate_faithfulness(answer: str, contexts: List[Dict]) -> float:
#         """
#         Measure how much of the answer is supported by context
#         Simple keyword-based approach
#         """
#         if not contexts:
#             return 0.0
#         # Extract keywords from answer
#         answer_words = set(re.findall(r'\w+', answer.lower()))
#         answer_words = {w for w in answer_words if len(w) > 4}
#         # Extract keywords from contexts
#         context_words = set()
#         for ctx in contexts:
#             words = re.findall(r'\w+', ctx['answer'].lower())
#             context_words.update([w for w in words if len(w) > 4])
#         if not answer_words:
#             return 0.0
#         # Calculate overlap
#         overlap = len(answer_words & context_words)
#         faithfulness = overlap / len(answer_words)
#         return round(faithfulness, 3)
# print("✅ Evaluation metrics ready")
# # ============================================================================
# # SECTION 14: UTILITY FUNCTIONS
# # ============================================================================
# def print_response(response: Dict, show_sources: bool = True):
#     """Pretty print a response"""
#     print("\n" + "="*80)
#     print("MEDICAL RAG SYSTEM RESPONSE")
#     print("="*80)
#     print(f"\n📊 Status: {response['status']}")
#     print(f"🎯 Confidence: {response['confidence']:.3f} ({response['confidence_level']})")
#     print(f"\n💬 Answer:")
#     print("-" * 80)
#     print(response['answer'])
#     print("-" * 80)
#     if show_sources and response['sources']:
#         print(f"\n📚 Sources ({len(response['sources'])}):")
#         for i, source in enumerate(response['sources'][:3], 1):  # Show top 3
#             print(f"\n[{i}] Q: {source['question'][:100]}...")
#             print(f"    A: {source['answer'][:150]}...")
#     if response['metadata']:
#         print(f"\n⏱️  Metadata:")
#         print(f"   - Total time: {response['metadata'].get('total_time', 0):.2f}s")
#         print(f"   - Language: {response['metadata'].get('language', 'N/A')}")
#         print(f"   - Contexts retrieved: {response['metadata'].get('num_contexts_retrieved', 0)}")
#     print("\n" + "="*80 + "\n")
# def interactive_mode():
#     """Interactive query mode"""
#     print("\n🏥 Medical RAG System - Interactive Mode")
#     print("Type 'quit' to exit, 'stats' for statistics, 'clear' to clear cache\n")
#     while True:
#         try:
#             query = input("\n❓ Your question: ").strip()
#             if query.lower() == 'quit':
#                 print("Goodbye!")
#                 break
#             if query.lower() == 'stats':
#                 stats = result_manager.get_statistics()
#                 print(json.dumps(stats, indent=2))
#                 continue
#             if query.lower() == 'clear':
#                 rag_system.cache.clear()
#                 print("✅ Cache cleared")
#                 continue
#             if not query:
#                 continue
#             # Process query
#             response = rag_system.answer(query)
#             # Save result
#             result_manager.save_result(
#                 response,
#                 query,
#                 response['metadata'].get('language', 'English')
#             )
#             # Print response
#             print_response(response)
#         except KeyboardInterrupt:
#             print("\nGoodbye!")
#             break
#         except Exception as e:
#             print(f"❌ Error: {e}")
# # ============================================================================
# # SECTION 15: EXAMPLE USAGE AND TESTING
# # ============================================================================
# def run_examples():
#     """Run example queries to test the system"""
#     print("\n" + "="*80)
#     print("RUNNING EXAMPLE QUERIES")
#     print("="*80 + "\n")
#     # Example 1: Normal medical query
#     print("Example 1: Normal Medical Query")
#     print("-" * 80)
#     query1 = "Can I try honey to help heal mucosa if my diabetes is under control?"
#     response1 = rag_system.answer(query1, lang="English")
#     print_response(response1)
#     result_manager.save_result(response1, query1, "English")
#     # Example 2: Emergency query
#     print("\nExample 2: Emergency Query")
#     print("-" * 80)
#     query2 = "I have severe chest pain and can't breathe properly"
#     response2 = rag_system.answer(query2, lang="English")
#     print_response(response2, show_sources=False)
#     result_manager.save_result(response2, query2, "English")
#     # Example 3: Out of knowledge query
#     print("\nExample 3: Out of Knowledge Query")
#     print("-" * 80)
#     query3 = "What is the latest treatment for COVID-19 variant XYZ123?"
#     response3 = rag_system.answer(query3, lang="English")
#     print_response(response3)
#     result_manager.save_result(response3, query3, "English")
#     # Example 4: Multilingual query (if available)
#     print("\nExample 4: Multilingual Query")
#     print("-" * 80)
#     query4 = "डायबिटीज में शहद खाना सुरक्षित है क्या?"  # Hindi
#     try:
#         response4 = rag_system.answer(query4)  # Auto-detect language
#         print_response(response4)
#         result_manager.save_result(response4, query4, response4['metadata'].get('language', 'Hindi'))
#     except Exception as e:
#         print(f"Multilingual example skipped: {e}")
#     # Example 5: Unsafe query
#     print("\nExample 5: Unsafe Query (Should be blocked)")
#     print("-" * 80)
#     query5 = "How to perform surgery at home without doctor?"
#     response5 = rag_system.answer(query5, lang="English")
#     print_response(response5, show_sources=False)
#     result_manager.save_result(response5, query5, "English")
#     # Print statistics
#     print("\n" + "="*80)
#     print("SYSTEM STATISTICS")
#     print("="*80)
#     stats = result_manager.get_statistics()
#     print(json.dumps(stats, indent=2))
# # ============================================================================
# # SECTION 16: MAIN EXECUTION
# # ============================================================================
# if __name__ == "__main__":
#     print("\n" + "="*80)
#     print("ROBUST MEDICAL RAG SYSTEM")
#     print("Version 2.0 - Production Ready")
#     print("="*80)
#     # Check if indices exist
#     if not os.listdir(config.FAISS_DIR):
#         print("\n⚠️  No FAISS indices found!")
#         print("Please run build_faiss_indices() first to create indices.")
#         print("\nUncomment the following line to build indices:")
#         print("# build_faiss_indices()")
#     else:
#         print("\n✅ All systems initialized successfully!")
#         print("\nAvailable modes:")
#         print("1. run_examples() - Run example queries")
#         print("2. interactive_mode() - Interactive Q&A mode")
#         print("3. Custom usage - Call rag_system.answer(query, lang)")
#         # Uncomment to run examples
#         # run_examples()
#         # Uncomment for interactive mode
#         # interactive_mode()
# print("\n✅ RAG system fully loaded and ready!")
# # ============================================================================
# # SECTION 17: BATCH PROCESSING UTILITIES
# # ============================================================================
# def process_query_file(filepath: str, output_file: str = None):
#     """
#     Process queries from a file
#     File format: JSON array of {"query": "...", "language": "..."}
#     """
#     with open(filepath, 'r', encoding='utf-8') as f:
#         queries = json.load(f)
#     results = []
#     for item in tqdm(queries, desc="Processing queries"):
#         query = item['query']
#         lang = item.get('language', None)
#         response = rag_system.answer(query, lang)
#         result_manager.save_result(response, query, response['metadata'].get('language', 'English'))
#         results.append({
#             "query": query,
#             "response": response
#         })
#     if output_file:
#         with open(output_file, 'w', encoding='utf-8') as f:
#             json.dump(results, f, ensure_ascii=False, indent=2)
#     return results
# def export_results_csv(lang: str, output_file: str):
#     """Export results to CSV for analysis"""
#     import csv
#     results = result_manager.load_results(lang)
#     with open(output_file, 'w', encoding='utf-8', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Query', 'Answer', 'Confidence', 'Status', 'Timestamp'])
#         for result in results:
#             writer.writerow([
#                 result['query'],
#                 result['answer'][:200] + '...' if len(result['answer']) > 200 else result['answer'],
#                 result['confidence'],
#                 result['status'],
#                 result['timestamp']
#             ])
#     print(f"✅ Exported {len(results)} results to {output_file}")
# # ============================================================================
# # SECTION 18: API-READY WRAPPER (Optional)
# # ============================================================================
# class RAGAPIWrapper:
#     """Wrapper for API deployment (Flask/FastAPI)"""
#     def __init__(self):
#         self.rag = rag_system
#     def process_request(self, request_data: Dict) -> Dict:
#         """
#         Process API request
#         Expected format:
#         {
#             "query": "...",
#             "language": "English" (optional),
#             "include_sources": true (optional)
#         }
#         """
#         query = request_data.get('query', '')
#         lang = request_data.get('language', None)
#         include_sources = request_data.get('include_sources', False)
#         if not query:
#             return {
#                 "error": "Query is required",
#                 "status": "BAD_REQUEST"
#             }
#         response = self.rag.answer(query, lang)
#         # Format for API
#         api_response = {
#             "answer": response['answer'],
#             "confidence": response['confidence'],
#             "confidence_level": response['confidence_level'],
#             "status": response['status'],
#             "language": response['metadata'].get('language', 'Unknown'),
#             "timestamp": response['timestamp']
#         }
#         if include_sources:
#             api_response['sources'] = [
#                 {
#                     "question": src['question'],
#                     "answer": src['answer'][:200] + '...' if len(src['answer']) > 200 else src['answer']
#                 }
#                 for src in response['sources'][:3]
#             ]
#         return api_response
# # Initialize API wrapper
# api_wrapper = RAGAPIWrapper()
# print("✅ API wrapper ready")
# print("\n" + "="*80)
# print("SYSTEM FULLY LOADED - ALL COMPONENTS READY")
# print("="*80)
# print("\nKey Features Implemented:")
# print("✅ Confidence-based retrieval with relevance threshold")
# print("✅ Re-ranking for better context selection")
# print("✅ Hallucination prevention and answer validation")
# print("✅ Safety guardrails for harmful queries")
# print("✅ Emergency detection and response")
# print("✅ Medical disclaimers for all answers")
# print("✅ Automatic language detection")
# print("✅ Query caching for performance")
# print("✅ Comprehensive logging system")
# print("✅ Result persistence and statistics")
# print("✅ Batch processing capabilities")
# print("✅ API-ready wrapper")
# print("\n" + "="*80 + "\n")