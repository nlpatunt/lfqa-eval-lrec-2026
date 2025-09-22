import os
from dotenv import load_dotenv 
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from pathlib import Path
class Specificity_score:

    def __init__(self,
                 hf_token: str = None,
                 model_id: str = "gtfintechlab/SubjECTiveQA-SPECIFIC",
                 device: str = None):

        # Resolve token
        env_path = Path(r"C:\Users\rafid\Source\Repos\lfqa-eval\config\.env")
        load_dotenv(dotenv_path=env_path)

        self.hf_token = os.getenv("HF_TOKEN")
        if not self.hf_token:
            raise ValueError(
                "Hugging Face token not provided. Pass hf_token=... or set HUGGING_FACE_HUB_TOKEN / HF_TOKEN."
            )

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)

        self.model_id = model_id
        # Loads a gated/private model using a token.
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, token=self.hf_token)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, token=self.hf_token)
        except TypeError:
            # Older transformers fallback
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_auth_token=self.hf_token)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_id, use_auth_token=self.hf_token)

        self.model.eval()
        self.model.to(self.device)

        # Label mapping (0/1/2) -> strings
        self.id2label = {0: "LOW_SPECIFICITY", 1: "NEUTRAL", 2: "HIGH_SPECIFICITY"}

    def _build_payload(self, text=None, question=None, answer=None):
        #You can score either a single paragraph (text) or a QA pair (question,answer).
        #if text:
            #return f"Answer: {text}"
        if question is not None and answer is not None:
            return f"Question: {question} Answer: {answer}"
        raise ValueError("Provide either `text` OR both `question` and `answer`.")

    @staticmethod
    def _to_scores(probs):
        p_low, p_neutral, p_high = probs
        score_high_prob = p_high
        #print(p_low , p_neutral,p_high)
        score_0_1 = p_low * 0.0 + p_neutral * 0.5 + p_high * 1.0
        score_1_5 = 1.0 + score_0_1 * 4.0
        return score_high_prob, score_0_1, score_1_5


    """
    def score(self, text: str = None, question: str = None, answer: str = None, max_length: int = 512):
        payload = self._build_payload(text=text, question=question, answer=answer)
        enc = self.tokenizer(
            payload,
            return_tensors="pt",
            #truncation=True, #if text too long, truncate from the end 512 max size generally
            padding=True,
            max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**enc).logits[0]
            probs_t = torch.softmax(logits, dim=-1).detach().cpu()
        probs = probs_t.tolist()

        pred_id = int(torch.argmax(probs_t).item())
        pred_label = self.id2label[pred_id]
        score_high_prob, score_0_1, score_1_5 = self._to_scores(probs)

        return {
            "label": pred_label,
            "probs": {"low": probs[0], "neutral": probs[1], "high": probs[2]},
            "score_high_prob": float(score_high_prob),
            "score_0_1": float(score_0_1),
            "score_1_5": float(score_1_5),
        }
    """
    def score(self, text: str = None, question: str = None, answer: str = None, max_length: int = 512):
        """
        Scores a paragraph OR a (question, answer) pair for specificity.
        If the input exceeds the model's context window (e.g., 512 tokens),
        it is split into overlapping chunks and aggregated into one prediction.
        """
        payload = self._build_payload(text=text, question=question, answer=answer)

        # ---- Safe context limits ----
        try:
            context_limit = int(getattr(self.model.config, "max_position_embeddings", 512)) or 512
        except Exception:
            context_limit = 512

        try:
            window = int(max_length) if max_length else context_limit
            if window <= 0 or window > context_limit:
                window = context_limit
        except Exception:
            window = context_limit

        stride = max(0, min(window // 4, 128))  # <= 25% overlap

        # ---- Tokenize with overflow-aware chunking ----
        enc = self.tokenizer(
            payload,
            return_tensors="pt",
            truncation=True,
            max_length=window,
            stride=stride,
            return_overflowing_tokens=True,#splits long text into multiple overlapping windows.
            padding="longest",#pad each chunk to the length of the longest one in the batch.
            #So if your text is 1200 tokens and window = 512, you’ll get ~3 overlapping chunks.
        )

        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # ---- Forward pass ----
        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
            probs_t = torch.softmax(logits, dim=-1)

        # ---- Aggregate ----
        if probs_t.shape[0] == 1:#Case 1: Only one chunk
            agg_probs = probs_t[0]
        else:#Case 2: Multiple chunks
            lengths = attention_mask.sum(dim=1, keepdim=True).clamp(min=1)
            #Do a weighted average, where each chunk’s probabilities are weighted by how many real tokens it covers.
            agg_probs = (probs_t * lengths).sum(dim=0) / lengths.sum(dim=0)

        # ---- Final scores ----
        pred_id = int(torch.argmax(agg_probs).item())
        pred_label = self.id2label[pred_id]
        probs = agg_probs.detach().cpu().tolist()
        score_high_prob, score_0_1, score_1_5 = self._to_scores(probs)

        return {
            "label": pred_label,
            "probs": {"low": probs[0], "neutral": probs[1], "high": probs[2]},
            "score_high_prob": float(score_high_prob),
            "score_0_1": float(score_0_1),
            "score_1_5": float(score_1_5),
        }


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # Option A: pass explicitly
    scorer = Specificity_score(hf_token="hf_your_token_here")

    # Option B: set env var:
    #   export HUGGING_FACE_HUB_TOKEN=hf_your_token_here   (Linux/Mac)
    #   setx HUGGING_FACE_HUB_TOKEN hf_your_token_here     (Windows, new shell)
    #scorer = Specificity_score()

    # Paragraph-only
    #para = "We expect a 10% YoY revenue increase due to a Q4 mid-market SaaS launch adding about $12M ARR."
    #print(scorer.score(text=para))

    # Question–Answer
    q = "How do I prevent ray from running tasks on the head node? "
    a = "As an AI language model, I cannot see your entire system setup and state. However, here are some general tips that can help you prevent your ray from running tasks on the head node:\n\n1. Set up a dedicated head node: Having a separate head node where you can receive and process your tasks can prevent them from running on the main node.\n\n2. Use a task queue: If you want to run tasks on multiple nodes simultaneously, you can use a task queue. This will allow you to specify the task you want to run and have it executed on one node, while other nodes continue to run other tasks.\n\n3. Use a reprojection matrix: If you are using a library or framework that has a reprojection matrix, you can use it to map your tasks to different nodes. This can help you avoid running tasks on the head node.\n\n4. Use a dedicated scheduler: If you have a dedicated scheduler, you can use it to run your tasks on different nodes at the same time. This can help prevent Ray from running tasks on the head node.\n\n5. Monitor your system: Keep an eye on your system and make sure everything is running smoothly. If there are any issues, you may need to address them.\n"
    print(scorer.score(question=q, answer=a))

    # Question–Answer
    #q = "Why did engagement drop last month?"
    #a = (
    #"Daily active users decreased by 12.4% from 1.2M in July to 1.05M in August. "
    #"The main driver was a 16-day outage in the iOS push notification service "
    #"between Aug 2 and Aug 18, which cut click-through rates by 34%. "
    #"After deploying the patch on Aug 19, DAU rebounded to 1.18M by Aug 25."
    #)
    #print(scorer.score(question=q, answer=a))