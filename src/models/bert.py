"""BERT-Large model for MLPerf inference on ROCm."""

import torch
from transformers import BertForQuestionAnswering, BertTokenizer


def load_bert(device: torch.device, precision: str = "fp32") -> torch.nn.Module:
    """Load BERT-Large for question answering."""
    model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    model = BertForQuestionAnswering.from_pretrained(model_name)
    model = model.to(device)

    if precision == "fp16":
        model = model.half()

    model.eval()
    return model


class BertModel:
    """BERT wrapper with pre/post processing."""

    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased-whole-word-masking-finetuned-squad"
        )
        self.max_seq_length = 384

    def preprocess(self, question: str, context: str):
        """Tokenize question and context."""
        encoding = self.tokenizer(
            question, context,
            max_length=self.max_seq_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].to(self.device),
            "attention_mask": encoding["attention_mask"].to(self.device),
            "token_type_ids": encoding["token_type_ids"].to(self.device),
        }

    def postprocess(self, output, input_ids):
        """Extract answer span from model output."""
        start_logits = output.start_logits[0]
        end_logits = output.end_logits[0]

        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()

        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0][start_idx:end_idx + 1])
        answer = self.tokenizer.convert_tokens_to_string(tokens)

        confidence = (
            torch.softmax(start_logits, dim=0)[start_idx].item() +
            torch.softmax(end_logits, dim=0)[end_idx].item()
        ) / 2

        return {
            "answer": answer,
            "start": start_idx,
            "end": end_idx,
            "confidence": confidence,
        }

    @torch.inference_mode()
    def infer(self, input_ids, attention_mask, token_type_ids=None):
        """Run inference."""
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
