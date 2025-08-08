import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm

# 设置随机种子，保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 1. 数据准备
class TextDataset(Dataset):
    def __init__(self, text, seq_length=50):
        self.seq_length = seq_length
        # 创建字符到索引的映射
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

        # 转换文本为索引序列
        self.indices = [self.char_to_idx[ch] for ch in text]

    def __len__(self):
        return len(self.indices) - self.seq_length

    def __getitem__(self, idx):
        # 输入序列和目标序列（输入序列的下一个字符）
        input_seq = self.indices[idx:idx+self.seq_length]
        target_seq = self.indices[idx+1:idx+self.seq_length+1]
        return torch.tensor(input_seq), torch.tensor(target_seq)

# 2. 模型定义
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)
        logits = self.fc(output)
        return logits, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())
        return hidden

# 3. 训练函数
def train_model(model, train_loader, criterion, optimizer, device, epochs=10):
    model.to(device)
    model.train()

    for epoch in range(epochs):
        total_loss = 0

        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)

            # 重置梯度
            optimizer.zero_grad()

            # 根据当前批次大小初始化隐藏状态
            hidden = model.init_hidden(batch_size)
            hidden = tuple([h.to(device) for h in hidden])

            # 前向传播
            logits, hidden = model(inputs, hidden)

            # 计算损失
            loss = criterion(logits.view(-1, model.fc.out_features), targets.view(-1))

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')

# 4. 生成文本函数
def generate_text(model, start_text, length, device, dataset):
    model.eval()
    with torch.no_grad():
        # 初始化输入和隐藏状态
        input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in start_text], device=device).unsqueeze(0)
        hidden = model.init_hidden(1)
        hidden = tuple([h.to(device) for h in hidden])

        generated_text = start_text

        # 生成文本
        for _ in range(length):
            logits, hidden = model(input_seq, hidden)
            # 取最后一个时间步的输出
            logits = logits[:, -1, :]
            # 使用softmax获取概率分布
            probs = torch.nn.functional.softmax(logits, dim=1)
            # 采样下一个字符
            idx = torch.multinomial(probs, num_samples=1).item()
            # 添加到生成的文本中
            generated_text += dataset.idx_to_char[idx]
            # 更新输入序列
            input_seq = torch.tensor([[idx]], device=device)

    return generated_text

# 5. 主函数
def main():
    # 配置参数
    embedding_dim = 256
    hidden_dim = 512
    num_layers = 2
    batch_size = 64
    seq_length = 50
    epochs = 10
    learning_rate = 0.001

    # 检查是否有GPU可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 准备数据（这里使用一个简单的文本示例，实际应用中应该使用更大的语料库）
    # 注意：在实际训练中，你需要替换为自己的文本数据
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 创建数据集和数据加载器
    dataset = TextDataset(text, seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型、损失函数和优化器
    model = LSTMModel(dataset.vocab_size, embedding_dim, hidden_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    train_model(model, train_loader, criterion, optimizer, device, epochs)

    # 保存模型
    torch.save(model.state_dict(), 'model.pth')
    print('Model saved to model.pth')

    # 生成文本示例
    start_text = 'Once upon a time'
    generated_text = generate_text(model, start_text, 200, device, dataset)
    print('Generated text:')
    print(generated_text)

if __name__ == '__main__':
    main()