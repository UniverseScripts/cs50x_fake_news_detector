import math
from typing import Tuple, List, Dict
from transformers import pipeline

# --------------------Utilities--------------------
# Constrain the output
def _to_int_percent(p: float) -> int:
    # Convert a probability [0,1] to an integer percent [0..100].
    if p is None or math.isnan(p):
        return 0
    return max(0, min(100, int(round(p * 100))))


def postprocess_predictions(all_scores: List[Dict[str, float]]) -> Tuple[str, int]:
    """
    Accepts a list like: [{'label': 'FAKE', 'score': 0.82}, ...]
    Ensures code returns (verdict, confidence_int) with verdict ∈ {'REAL','FAKE'} and confidence ∈ [0..100].
    """
    if not all_scores:
        return "REAL", 0
    # Guards scores must be a list[dict]
    # not isinstance(all_scores, list) or not all(isinstance(d, dict) for d in all_scores):
        return "REAL", 0

    # Choose the label with the highest score
    top = max(all_scores, key=lambda d: d.get("score", 0.0))
    raw_label = str(top.get("label", "")).strip().lower()
    conf_int = _to_int_percent(top.get("score", 0.0))


    # Direct mappings
    if "fake" in raw_label or "false" in raw_label or "pants" in raw_label:
        return "FAKE", conf_int
    if "real" in raw_label or "true" in raw_label or "authentic" in raw_label:
        return "REAL", conf_int

    # Handle common generic label sets
    labels_lower = [str(d.get("label", "")).strip().lower() for d in all_scores]
    if set(labels_lower) == {"label_0", "label_1"}:
        # Heuristic: label_1 often used as positive class; we map by the top label name
        return ("FAKE" if raw_label == "label_1" else "REAL"), conf_int

    if set(labels_lower) == {"negative", "positive"}:
        # Fallback mapping if sentiment labels were used
        return ("FAKE" if raw_label == "negative" else "REAL"), conf_int

    # Last-resort: treat suspicious tokens as FAKE
    suspicious = {"misinfo", "misinformation", "decept", "hoax", "rumor", "fabricated", "clickbait"}
    if any(tok in raw_label for tok in suspicious):
        return "FAKE", conf_int

    # Default
    return "REAL", conf_int


# --------------------Chunking based on tokens--------------------
# Chunks by overlapping tokens
def chunks(text, tokenizer, max_len=512, stride=256) -> List[str]:
    # Checks for str format
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    if not text.strip():
        return []
    
    # Converts text to tokens, indices tokens
    enc = tokenizer(text, add_special_tokens=True, truncation=False)
    ids = enc["input_ids"]
    
    # Declare a list of chunks and a pointer
    chunks: List[str] = []
    pointer = 0
    
    
    while(pointer < len(ids)):
        # Computes the end of a window (Ex: In a list with 1000 tokens [1..1000], window#1 ranges from [0..512], end returns 512)
        end = min(pointer + max_len, len(ids))
        # Ex: Chumk#1 is window#1 [0..511] 
        chunks_ids = ids[pointer:end]
        if not chunks_ids:
            break
        chunks.append(tokenizer.decode(chunks_ids, skip_special_tokens=True))
        if end == len(ids):
            break
        # Computes the start of the subsequent window (Ex: window#2 starts on [256..256+512=768], overlaps window#1 on [256..512])
        pointer += stride
    return chunks   
            
# --------------------Model--------------------
#Load a pre-trained model
class FakeNewsDetector:
    def __init__(self):
        self.classifier = pipeline("text-classification",
                         model="Pulk17/Fake-News-Detection",
                         top_k=None,
                         device=-1) # Input "0" if you wish the AI to run on your GPU with NVIDIA's CUDA installed, "-1" for CPU.
        
    def predict(self, text):
        if not str(text) or not text.strip():
            return "REAL", 0
        
        try:
            tok = self.classifier.tokenizer
            
            tok.model_max_length = int(1e9) # Overwrites the max token length (512 for BERT models), emits the warning but does not affect the code
            
            windows = chunks(text, tok, max_len=512, stride=256)
            
            if not windows:
                return "REAL", 0
            
            # Case 1: Single call for tokens <= 512
            #if len(windows) == 1:       
                # single_call = self.classifier(windows[0], truncation=True, max_length=512, padding=False)
                
                #output = single_call if isinstance(single_call, dict) else [single_call]
                    
                #verdict, confidence = postprocess_predictions(output)
                #return verdict, max(0, min(100, int(confidence)))
            
            # Case 2: Chunking for tokens >= 512
            # Utilizes chunking to return list of component results, each component contains a dict.
            # [
                # [{"label": REAL, "score": <float>[0,1]}, {"label": FAKE, "score": 1 - <float>[0,1]}] for chunk#1, ...
            # ] 
            results = self.classifier(windows, truncation=True, max_length=512, padding=False)
            
            agg: Dict[str, float] = {}
            
            # Loops to aggregate the scores across multiple chunks (label & score/probability)
            for chunk_scores in results:
                if not isinstance(chunk_scores, list):
                    continue
                for d in chunk_scores:
                    if not isinstance(d, dict):
                        continue
                    label = str(d.get("label", "")).strip()
                    score = float(d.get("score", 0.0))
                    agg[label] = agg.get(label, 0.0) + score

            total = sum(agg.values()) or 1.0
            scores = [{"label": k, "score": v/total} for k, v in agg.items()]
            
            # Get results as float from 0-1
            verdict, confidence = postprocess_predictions(scores)
            
            # Probability classfication
            verdict = "FAKE" if verdict == "FAKE" else "REAL"
            confidence = max(0, min(100, int(confidence)))
            return verdict, confidence
        
        except Exception as e:
            return f"Error: {str(e)}", 0
        
# Call for Global instance
detector = FakeNewsDetector()