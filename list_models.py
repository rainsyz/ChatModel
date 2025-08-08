import os
import glob
import datetime

def list_model_files():
    # 获取当前目录下所有.pth文件
    model_files = glob.glob('*.pth')

    if not model_files:
        print("未找到任何模型文件(.pth)")
        print("请先运行train_model.py训练模型，或运行improve_model.py微调模型")
        return

    model_info = []
    for file in model_files:
        try:
            # 获取文件修改时间
            mtime = os.path.getmtime(file)
            modified_time = datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
            # 获取文件大小
            file_size = os.path.getsize(file) / 1024  # KB
            model_info.append((file, modified_time, file_size))
        except Exception as e:
            print(f"获取文件信息失败 '{file}': {e}")
            continue

    # 按修改时间降序排序
    model_info.sort(key=lambda x: x[1], reverse=True)

    print("可用的模型文件:\n")
    print(f"{'文件名':<40} {'修改时间':<20} {'文件大小':<10}")
    print("-" * 70)
    for file, modified_time, file_size in model_info:
        print(f"{file:<40} {modified_time:<20} {file_size:.2f} KB")

    # 提供使用建议
    print("\n使用示例:")
    print("1. 基本对话: python chat_with_model.py")
    print("2. 增强版对话: python enhanced_chat.py --model 模型文件名")
    print("   例如: python enhanced_chat.py --model model_finetuned_20231107_153045.pth")

def main():
    print("====== 模型文件列表 ======")
    list_model_files()
    print("=========================")

if __name__ == '__main__':
    main()