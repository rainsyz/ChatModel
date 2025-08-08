import torch
import torch.nn as nn
from train_model import LSTMModel, TextDataset
import os

def load_model_and_dataset():
    # 加载文本数据
    with open('data.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # 创建数据集
    dataset = TextDataset(text)

    # 创建模型并加载权重
    model = LSTMModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2
    )

    # 加载训练好的模型权重
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
        print('模型加载成功！')
    else:
        print('未找到模型文件，请先运行train_model.py训练模型。')
        exit()

    # 设置为评估模式
    model.eval()

    return model, dataset

def generate_response(model, prompt, dataset, max_length=100, temperature=0.7):
    """
    基于输入提示生成响应
    temperature: 控制生成文本的随机性，值越小越确定，值越大越随机
    """
    device = torch.device('cpu')  # 我们使用CPU进行推理
    model.to(device)

    # 初始化输入和隐藏状态
    try:
        input_seq = torch.tensor([dataset.char_to_idx[ch] for ch in prompt], device=device).unsqueeze(0)
    except KeyError as e:
        # 如果提示中包含不在词汇表中的字符
        unknown_char = str(e).strip("'")
        print(f"警告: 字符 '{unknown_char}' 不在模型词汇表中，已替换为空格")
        prompt = prompt.replace(unknown_char, ' ')
        input_seq = torch.tensor([dataset.char_to_idx.get(ch, 0) for ch in prompt], device=device).unsqueeze(0)

    hidden = model.init_hidden(1)
    hidden = tuple([h.to(device) for h in hidden])

    generated_text = prompt

    # 生成文本
    for _ in range(max_length):
        logits, hidden = model(input_seq, hidden)
        # 取最后一个时间步的输出
        logits = logits[:, -1, :] / temperature
        # 使用softmax获取概率分布
        probs = torch.nn.functional.softmax(logits, dim=1)
        # 采样下一个字符
        idx = torch.multinomial(probs, num_samples=1).item()
        # 添加到生成的文本中
        next_char = dataset.idx_to_char[idx]
        generated_text += next_char
        # 如果遇到结束符或换行符，可以提前结束
        if next_char in ['.', '!', '?', '\n'] and len(generated_text) > len(prompt) + 5:
            break
        # 更新输入序列
        input_seq = torch.tensor([[idx]], device=device)

    return generated_text

def chat_with_model(model, dataset, max_history=3):
    """
    与模型进行对话交互
    max_history: 保留的最大对话历史轮数
    """
    print("=== 开始与模型对话 ===")
    print("提示: 输入 'exit' 或 'quit' 结束对话")
    print("      输入 'clear' 清空对话历史")
    print("\n")

    conversation_history = []

    while True:
        # 获取用户输入
        user_input = input("你: ")

        if user_input.lower() in ['exit', 'quit']:
            print("模型: 再见！")
            break
        elif user_input.lower() == 'clear':
            conversation_history = []
            print("模型: 对话历史已清空")
            continue

        # 添加用户输入到对话历史
        conversation_history.append(f"你: {user_input}")

        # 如果历史太长，删除最早的记录
        if len(conversation_history) > max_history * 2:
            conversation_history = conversation_history[-max_history*2:]

        # 构建提示
        prompt = "\n".join(conversation_history) + "\n模型: "

        # 生成响应
        response = generate_response(model, prompt, dataset)

        # 提取模型响应部分
        model_response = response[len(prompt):].strip()

        # 显示模型响应
        print(f"模型: {model_response}")

        # 添加模型响应到对话历史
        conversation_history.append(f"模型: {model_response}")

if __name__ == '__main__':
    # 加载模型和数据集
    model, dataset = load_model_and_dataset()

    # 开始对话
    chat_with_model(model, dataset)