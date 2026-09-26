import os
from dotenv import load_dotenv
from dashscope import Generation

# 显式指定 .env 文件路径，解决找不到文件问题
env_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=env_path)

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    print("错误: 没有读到 API Key, 请检查 .env 文件")
    print(f"尝试读取路径：{env_path}")
    exit(1)

print(f"✅ 成功读取密钥：{api_key[:15]}******")

resp = Generation.call(
    model="qwen-plus",
    messages=[{"role": "user", "content": "你好，请回答“收到”两个字"}],
    api_key=api_key,
    result_format="message"
)

if resp.status_code == 200:
    print("✅调用成功!")
    print("回复内容:", resp.output.choices[0].message.content)
else:
    print("❌调用失败")
    print("状态码:", resp.status_code)
    print("错误信息:", resp.message)
