import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import SimpleTransformerQA
from data_processing import get_train_data_loader, tokenizer

# 训练准备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vocab_size = tokenizer.vocab_size
model = SimpleTransformerQA(vocab_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
train_loader = get_train_data_loader()
epochs = 5

# 训练循环
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader)
    total_loss = 0
    for batch in loop:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_pos = batch["start_positions"].to(device)
        end_pos = batch["end_positions"].to(device)

        start_logits, end_logits = model(input_ids, attention_mask)

        loss_start = F.cross_entropy(start_logits, start_pos)
        loss_end = F.cross_entropy(end_logits, end_pos)
        loss = (loss_start + loss_end) / 2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

print("训练结束！")

# 保存模型
torch.save(model.state_dict(), 'SampleTransformerQA_model.pth')