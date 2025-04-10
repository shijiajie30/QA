import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from model import SimpleTransformerQA

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size
model = SimpleTransformerQA(vocab_size).to(device)
model.load_state_dict(torch.load('SampleTransformerQA_model.pth'))


def answer_question(question, context, threshold=0.1):
    model.eval()
    inputs = tokenizer(question, context, return_tensors="pt", max_length=384, truncation=True).to(device)
    with torch.no_grad():
        start_logits, end_logits = model(inputs["input_ids"])

    start_probs = F.softmax(start_logits, dim=1)
    end_probs = F.softmax(end_logits, dim=1)

    start_idx = torch.argmax(start_probs, dim=1).item()
    end_idx = torch.argmax(end_probs, dim=1).item()
    cls_score = (start_probs[0, 0] + end_probs[0, 0]) / 2

    # 如果模型倾向于预测 CLS 位置，说明可能是无答案
    if cls_score > threshold:
        return "无法回答这个问题。"

    tokens = inputs["input_ids"][0][start_idx:end_idx + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True)


# 测试问答
ctx = "The Transformers library by Hugging Face provides thousands of pretrained models..."
q = "What does the Transformers library provide?"
print("Answer:", answer_question(q, ctx))