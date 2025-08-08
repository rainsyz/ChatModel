import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from train_model import LSTMModel
import os
import json
import argparse
import random
from datetime import datetime

class DialogueDataset(Dataset):
    """对话数据集，用于模型微调"""
    def __init__(self, dialogue_file, char_to_idx, max_length=100):
        self.char_to_idx = char_to_idx
        self.max_length = max_length
        self.data = self._load_dialogues(dialogue_file)

    def _load_dialogues(self, dialogue_file):
        """加载对话数据"""
        try:
            with open(dialogue_file, 'r', encoding='utf-8') as f:
                dialogues = json.load(f)
        except Exception as e:
            print(f"加载对话文件出错: {e}")
            return []

        data = []
        # 将对话转换为文本序列
        for i in range(len(dialogues) - 1):
            if dialogues[i]['role'] == 'user' and dialogues[i+1]['role'] == 'model':
                user_turn = dialogues[i]['content']
                model_turn = dialogues[i+1]['content']
                combined = f"user: {user_turn}\nmodel: {model_turn}\n"
                # 转换为索引序列
                seq = [self.char_to_idx.get(ch, 0) for ch in combined[:self.max_length]]
                # 填充或截断到固定长度
                if len(seq) < self.max_length:
                    seq += [0] * (self.max_length - len(seq))
                else:
                    seq = seq[:self.max_length]
                data.append(torch.tensor(seq))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def load_char_mapping():
    """加载字符映射表"""
    # 尝试从data.txt创建字符映射
    try:
        with open('data.txt', 'r', encoding='utf-8') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        return char_to_idx
    except Exception as e:
        print(f"创建字符映射时出错: {e}")
        # 使用默认映射
        return {chr(i): i for i in range(128)}

def fine_tune_model(model_path, dialogue_file, epochs=5, batch_size=32, learning_rate=0.001):
    """微调模型"""
    # 加载字符映射
    char_to_idx = load_char_mapping()
    vocab_size = len(char_to_idx)

    # 创建数据集和数据加载器
    dataset = DialogueDataset(dialogue_file, char_to_idx)
    if len(dataset) == 0:
        print("没有可用的对话数据，无法进行微调。")
        return

    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 创建模型并加载权重
    model = LSTMModel(
        vocab_size=vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2
    )

    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print('基础模型加载成功！')
        except Exception as e:
            print(f"加载模型时出错: {e}")
            return
    else:
        print('未找到模型文件，请先运行train_model.py训练基础模型。')
        return

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 设置为训练模式
    model.train()

    # 开始微调
    print(f"开始微调模型，共{epochs}个周期...")
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(data_loader):
            # 准备输入和目标
            inputs = batch[:, :-1]
            targets = batch[:, 1:]

            # 初始化隐藏状态
            hidden = model.init_hidden(batch_size=inputs.size(0))

            # 前向传播
            logits, hidden = model(inputs, hidden)

            # 计算损失
            loss = criterion(logits.reshape(-1, vocab_size), targets.reshape(-1))

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"周期 {epoch+1}/{epochs}, 批次 {batch_idx+1}/{len(data_loader)}, 损失: {loss.item():.4f}")

        avg_loss = total_loss / len(data_loader)
        print(f"周期 {epoch+1}/{epochs} 完成，平均损失: {avg_loss:.4f}")

    # 保存微调后的模型
    fine_tuned_model_path = f"model_finetuned_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
    torch.save(model.state_dict(), fine_tuned_model_path)
    print(f"微调后的模型已保存到: {fine_tuned_model_path}")

    return fine_tuned_model_path

def generate_dialogue_example(model, char_to_idx, idx_to_char, prompt, max_length=100, temperature=0.7):
    """生成对话示例"""
    model.eval()
    device = torch.device('cpu')
    model.to(device)

    # 初始化输入和隐藏状态
    input_seq = torch.tensor([char_to_idx.get(ch, 0) for ch in prompt], device=device).unsqueeze(0)
    hidden = model.init_hidden(1)
    hidden = tuple([h.to(device) for h in hidden])

    generated_text = prompt

    # 生成文本
    for _ in range(max_length):
        logits, hidden = model(input_seq, hidden)
        logits = logits[:, -1, :] / temperature
        probs = torch.nn.functional.softmax(logits, dim=1)
        idx = torch.multinomial(probs, num_samples=1).item()
        next_char = idx_to_char.get(idx, '?')
        generated_text += next_char
        input_seq = torch.tensor([[idx]], device=device)

    return generated_text

def main():
    parser = argparse.ArgumentParser(description='微调对话模型')
    parser.add_argument('--model', type=str, default='model.pth', help='基础模型路径')
    parser.add_argument('--dialogue', type=str, required=True, help='对话数据文件路径')
    parser.add_argument('--epochs', type=int, default=5, help='训练周期数')
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001, help='学习率')
    args = parser.parse_args()

    # 加载字符映射
    char_to_idx = load_char_mapping()
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}

    # 微调模型
    fine_tuned_model_path = fine_tune_model(
        args.model,
        args.dialogue,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )

    # 加载微调后的模型并生成示例
    if fine_tuned_model_path and os.path.exists(fine_tuned_model_path):
        print("\n生成微调后模型的对话示例...")
        model = LSTMModel(
            vocab_size=len(char_to_idx),
            embedding_dim=256,
            hidden_dim=512,
            num_layers=2
        )
        model.load_state_dict(torch.load(fine_tuned_model_path))

        # 使用一些测试提示
        test_prompts = [
            "user: 你好\nmodel: ",
            "user: 今天天气怎么样\nmodel: ",
            "user: 你能做什么\nmodel: "
        ]

        for prompt in test_prompts:
            response = generate_dialogue_example(model, char_to_idx, idx_to_char, prompt)
            print(f"\n{response}")

if __name__ == '__main__':
    main()