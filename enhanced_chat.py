import torch
import torch.nn as nn
from train_model import LSTMModel, TextDataset
import os
import json
import time
import argparse
from datetime import datetime

class ConversationManager:
    """对话管理器，负责处理对话历史和上下文"""
    def __init__(self, max_history=5):
        self.max_history = max_history
        self.conversation_history = []
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_directory = "conversations"
        os.makedirs(self.save_directory, exist_ok=True)

    def add_user_input(self, user_input):
        """添加用户输入到对话历史"""
        self.conversation_history.append({
            "role": "user",
            "content": user_input,
            "timestamp": time.time()
        })
        # 保持历史记录不超过最大长度
        self._trim_history()

    def add_model_response(self, response):
        """添加模型响应到对话历史"""
        self.conversation_history.append({
            "role": "model",
            "content": response,
            "timestamp": time.time()
        })
        # 保持历史记录不超过最大长度
        self._trim_history()

    def _trim_history(self):
        """修剪对话历史，保持在最大长度以内"""
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history*2:]

    def get_context(self, max_tokens=500):
        """获取对话上下文，限制最大token数量"""
        context = []
        total_tokens = 0

        # 从最近的对话开始，向前添加
        for turn in reversed(self.conversation_history):
            turn_text = f"{turn['role']}: {turn['content']}"
            turn_tokens = len(turn_text)

            if total_tokens + turn_tokens > max_tokens:
                # 如果添加当前回合会超过限制，就只添加部分
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 0:
                    # 只添加能容纳的部分
                    context.append(turn_text[:remaining_tokens] + "...")
                break

            context.append(turn_text)
            total_tokens += turn_tokens

        # 反转以保持正确的时间顺序
        return "\n".join(reversed(context)) + "\nmodel: "

    def save_conversation(self):
        """保存对话历史到文件"""
        save_path = os.path.join(self.save_directory, f"conversation_{self.session_id}.json")
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
        return save_path

    def clear_history(self):
        """清空对话历史"""
        self.conversation_history = []
        return "对话历史已清空"

def load_model_and_dataset(model_path='model.pth'):
    # 加载文本数据
    try:
        with open('data.txt', 'r', encoding='utf-8') as f:
            text = f.read()
    except FileNotFoundError:
        print("错误: 未找到data.txt文件，请确保该文件存在。")
        exit()
    except Exception as e:
        print(f"读取数据文件时出错: {e}")
        exit()

    # 创建数据集
    try:
        dataset = TextDataset(text)
    except Exception as e:
        print(f"创建数据集时出错: {e}")
        exit()

    # 创建模型并加载权重
    model = LSTMModel(
        vocab_size=dataset.vocab_size,
        embedding_dim=256,
        hidden_dim=512,
        num_layers=2
    )

    # 加载训练好的模型权重
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print(f'模型加载成功: {model_path}')
        except Exception as e:
            print(f"加载模型时出错: {e}")
            exit()
    else:
        print(f'未找到模型文件: {model_path}')
        exit()

    # 设置为评估模式
    model.eval()

    return model, dataset

def generate_response(model, prompt, dataset, max_length=150, temperature=0.7, top_k=50, top_p=0.95):
    """
    基于输入提示生成响应
    temperature: 控制生成文本的随机性，值越小越确定，值越大越随机
    top_k: 只考虑概率最高的k个字符
    top_p: 只考虑累积概率达到p的字符
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
    except Exception as e:
        print(f"处理输入时出错: {e}")
        return prompt + "[生成出错]"

    hidden = model.init_hidden(1)
    hidden = tuple([h.to(device) for h in hidden])

    generated_text = prompt

    # 生成文本
    for _ in range(max_length):
        try:
            logits, hidden = model(input_seq, hidden)
            # 取最后一个时间步的输出
            logits = logits[:, -1, :] / temperature

            # 应用top-k采样
            if top_k > 0:
                values, indices = torch.topk(logits, top_k)
                logits[logits < values[:, -1:]] = -float('Inf')

            # 应用top-p采样
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.nn.functional.softmax(sorted_logits, dim=-1), dim=-1)
                # 删除累积概率超过p的标记
                sorted_indices_to_remove = cumulative_probs > top_p
                # 确保至少保留一个标记
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                logits[:, indices_to_remove] = -float('Inf')

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
        except Exception as e:
            print(f"生成文本时出错: {e}")
            break

    return generated_text

def chat_with_model(model, dataset):
    """与模型进行对话交互"""
    print("=== 开始与模型对话 ===")
    print("提示: 输入 'exit' 或 'quit' 结束对话")
    print("      输入 'clear' 清空对话历史")
    print("      输入 'save' 保存当前对话")
    print("      输入 'history [num]' 设置最大历史记录轮数(默认5)")
    print("      输入 'temp [value]' 设置温度参数(0.1-2.0，默认0.7)")
    print("      输入 'help' 查看帮助信息")
    print("\n")

    # 创建对话管理器
    conversation_manager = ConversationManager(max_history=5)
    temperature = 0.7

    while True:
        try:
            # 获取用户输入
            user_input = input("你: ")

            # 处理特殊命令
            if user_input.lower() in ['exit', 'quit']:
                save_path = conversation_manager.save_conversation()
                print(f"模型: 对话已保存到 {save_path}")
                print("模型: 再见！")
                break
            elif user_input.lower() == 'clear':
                response = conversation_manager.clear_history()
                print(f"模型: {response}")
                continue
            elif user_input.lower() == 'save':
                save_path = conversation_manager.save_conversation()
                print(f"模型: 对话已保存到 {save_path}")
                continue
            elif user_input.lower().startswith('history '):
                try:
                    new_max_history = int(user_input.split()[1])
                    if new_max_history > 0:
                        conversation_manager.max_history = new_max_history
                        print(f"模型: 最大历史记录已设置为 {new_max_history} 轮")
                    else:
                        print("模型: 请输入大于0的整数")
                except ValueError:
                    print("模型: 请输入有效的数字")
                continue
            elif user_input.lower().startswith('temp '):
                try:
                    new_temp = float(user_input.split()[1])
                    if 0.1 <= new_temp <= 2.0:
                        temperature = new_temp
                        print(f"模型: 温度参数已设置为 {temperature}")
                    else:
                        print("模型: 温度参数应在0.1到2.0之间")
                except ValueError:
                    print("模型: 请输入有效的数字")
                continue
            elif user_input.lower() == 'help':
                print("=== 帮助信息 ===")
                print("exit/quit: 结束对话")
                print("clear: 清空对话历史")
                print("save: 保存当前对话")
                print("history [num]: 设置最大历史记录轮数")
                print("temp [value]: 设置温度参数(0.1-2.0)")
                print("help: 查看帮助信息")
                continue

            # 添加用户输入到对话历史
            conversation_manager.add_user_input(user_input)

            # 构建提示
            prompt = conversation_manager.get_context()

            # 生成响应
            print("模型: 正在思考...")
            response = generate_response(model, prompt, dataset, temperature=temperature)

            # 提取模型响应部分
            model_response = response[len(prompt):].strip()

            # 显示模型响应
            print(f"模型: {model_response}")

            # 添加模型响应到对话历史
            conversation_manager.add_model_response(model_response)
        except Exception as e:
            print(f"对话过程中出错: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description='增强版对话系统')
    parser.add_argument('--model', type=str, default='model.pth', help='模型文件路径')
    args = parser.parse_args()

    # 加载模型和数据集
    model, dataset = load_model_and_dataset(args.model)

    # 开始对话
    chat_with_model(model, dataset)

if __name__ == '__main__':
    main()