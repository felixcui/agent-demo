
import os
from dotenv import load_dotenv, find_dotenv

# ### 加载环境变量
# 你必须自己准备好OpenAI的的Key，在本文中需要使用GPT4的模型
load_dotenv(find_dotenv(), override=True)
os.getenv('OPENAI_API_KEY')
from agent_demo import MyAgent

def launch_agent():
    human_icon = "\U0001F468"
    ai_icon = "\U0001F916"

    while True:
        task = input(f"{ai_icon}：有什么可以帮您？\n{human_icon}：")
        if task.strip().lower() == "quit":
            break
        agent = MyAgent()
        reply = agent.run(task)
        print(f"{ai_icon}：{reply}\n")

if __name__ == "__main__":
    launch_agent()

#提问示例：
# - 9月份的销售额是多少
# - 销售总额最大的产品是什么
# - 帮我找出销售额不达标的供应商
# - 给这两家供应商发一封邮件通知此事
# - 对比8月和9月销售情况，写一份报告