import sys
import os
import random
import time
from utils.db_handler import db_manager

# # --- 核心路径修复 (确保能找到 utils 包) ---
# current_path = os.path.abspath(__file__)
# project_root = os.path.dirname(current_path)
# if project_root not in sys.path:
#     sys.path.insert(0, project_root)

# try:
#     from utils.db_handler import db_manager
#     print("✅ 成功导入 db_manager")
# except ImportError as e:
#     print(f"❌ 导入失败: {e}")
#     print("请检查你的 utils 文件夹下是否有 db_handler.py，并且里面定义了 db_manager")
#     sys.exit(1)

def run_test():
    print(f"\n{'='*10} 开始 MySQL 全链路测试 {'='*10}")
    
    # 生成随机测试数据，避免冲突
    test_suffix = str(random.randint(1000, 9999))
    test_session_id = f"test_session_{test_suffix}"
    test_file_md5 = f"test_md5_{test_suffix}"
    test_file_name = f"test_doc_{test_suffix}.txt"

    # --- 1. 测试知识库文件记录 (模拟 vector_store.py) ---
    print(f"\n[1/3] 测试知识库表 (knowledge_files)...")
    try:
        # A. 检查是否存在 (应该是 False)
        exists_before = db_manager.is_file_exists(test_file_md5)
        print(f"   - 插入前检查: {'未存在 (正常)' if not exists_before else '❌ 异常: 数据已存在'}")

        # B. 插入数据
        # 注意：这里调用的是你刚才修改的 add_file_record
        db_manager.add_file_record(test_file_name, test_file_md5)
        print(f"   - 尝试插入文件记录: {test_file_name}")

        # C. 再次检查 (应该是 True)
        exists_after = db_manager.is_file_exists(test_file_md5)
        if exists_after:
            print("   - ✅ 验证插入成功 (MySQL 已记录该文件)")
        else:
            print("   - ❌ 验证失败: 插入后未查到数据")

    except Exception as e:
        print(f"   - ❌ 发生错误: {e}")

    # --- 2. 测试聊天记录 (模拟 react_agent.py) ---
    print(f"\n[2/3] 测试聊天记录表 (chat_history)...")
    try:
        # A. 插入用户提问
        db_manager.save_message(test_session_id, "user", "你好，这是一个测试")
        print("   - 已写入 User 消息")
        
        # B. 插入 AI 回复
        db_manager.save_message(test_session_id, "assistant", "收到，系统运行正常")
        print("   - 已写入 Assistant 消息")

        # C. 读取历史记录
        history = db_manager.load_messages(test_session_id)
        print(f"   - 读取历史记录条数: {len(history)}")
        
        if len(history) >= 2:
            print(f"   - 第一条内容: {history[0]['content']}")
            print("   - ✅ 聊天记录读写测试通过")
        else:
            print("   - ❌ 聊天记录读取数量不足")

    except Exception as e:
        print(f"   - ❌ 发生错误: {e}")

    print(f"\n{'='*10} 测试结束 {'='*10}")
    print("提示：你可以去 Navicat 查看这些以 'test_' 开头的数据")

if __name__ == '__main__':
    run_test()