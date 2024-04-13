
# ## 实现思路
# ### 构建智能体的一般性思路
# 如果要将大模型融入到你的应用中，最佳的方案是构建一个智能体。
# 常见的智能体结构都会包含三个核心环节：
# - 思考
# - 行动
# - 观察
# 
# 其中，图中Planning部份是**思考环节**，这是最重要的环节，也是典型的 LLM+Prompt+OutputParser 的结构。
# 智能体的LLM必须是推理能力足够强的大模型，例如GPT4，否则无法完成任务。<br>
# 智能体的Prompt一般依照相关论文的思路拟定思考逻辑，例如 ReAct 模式，下面的实践中还会包括反思、思维链等技巧。<br>
# 智能体需要调度哪个Action、需要什么输入参数，都由OutputParser负责解析。
# 
# 图中Action部份就是**行动环节**，就是根据思考结果（或称为计划）去调用工具，这个过程在AgentExcuter中有自己的实现逻辑，如果不使用他就要自己实现。
# 图中的Tools是部份**行动环节**可调用的外部工具，langchain 内置了一部份开箱即用的工具，但更多时候需要自己实现。
# # **观察环节**就是将工具调用的结果装入**思考环节**的提示语中，以便大模型在推理时使用。
# 图中的Memory是记忆部份，是智能体每个步骤或每次任务的执行结果，对应步骤的结果称为短期记忆，对应任务的执行结果称为长期记忆。
# 经过这些环节，智能体的运行就会形成一个完整闭环：思考、行动、观察，再思考、行动、观察 ... 经过N次循环后得出结论，这就是智能体的核心实现逻辑。
 
# ### 本文中智能体的实现逻辑
# 本文的实现实现逻辑是典型的ReAct策略，但其中增加了一些思维链技巧：
# 1. 用户输入
# 2. 思考：调用LLM思考
# 3. 判断：根据思考结果，判断是已经获得答案，还是需要调用工具
# 4. 行动：如果调用工具，就在工具箱中执行
# 5. 观察：将执行结果装入短时记忆
# 6. 回到第2步骤继续，直到思考结果中包括了FINISH
# 7. 生成最终答案

 
import os
import json
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
#from langchain_chinese import ChatZhipuAI
from langchain_community.chat_models import ChatZhipuAI
from langchain_community.llms import Ollama 
from langchain.schema.output_parser import StrOutputParser
from langchain.tools.render import render_text_description
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from pydantic import ValidationError

from tools  import *
from common import *
from rag import *
from prompt import *
 

# 自定义工具集
tools = [
    directory_inspection_tool,
    document_qa_tool, #知识库工具
    document_generation_tool,
    email_tool,
    excel_inspection_tool,
    excel_analysis_tool,
    finish_placeholder,
]
print(render_text_description(tools))

# ### 定义Action输出解析
# #### 定义Action
class Action(BaseModel):
    name: str = Field(description="工具或指令名称")
    args: Optional[Dict[str,Any]] = Field(default=None, description="工具或指令参数，由参数名称和参数值组成")

# 解析Action
action_output_parser = PydanticOutputParser(pydantic_object=Action)
# 实现自动纠错
robust_parser = OutputFixingParser.from_llm(parser=action_output_parser, llm=llm)
# #### 提示语中的action_parser
action_parser = chinese_friendly(action_output_parser.get_format_instructions())
print(action_output_parser.get_format_instructions())
 

from os.path import dirname
# ### main_prompt
main_prompt = prompt.partial(
    constraints=constraints,
    work_dir= dirname(__file__) + "/data",
    resources=resources,
    performance_evaluation=performance_evaluation,
    thought_instructions=thought_instructions,
    format_instructions=action_parser
)
# 此时，只剩下了short_term_memory、long_term_memory和tools需要填充：<br>
# 其中，short_term_memory用于保存中间步骤的运行结果，等待智能体运行时填充。<br>
# 另外两个参数则有助于应用的灵活配置。

 
class AgentChain:
    action_parser = chinese_friendly(action_output_parser.get_format_instructions())

    def __init__(self):
        self.llm = ChatOpenAI(
             model_name="gpt-4-1106-preview",
             temperature=0,
             model_kwargs={"seed": 42},
         )
         #self.llm = Ollama(base_url="http://10.139.226.233:11434", model="llama2")
        
    def get_reason_chain(self, task):
        prompt = main_prompt.partial(task_description=task, format_instructions=action_parser)
        print(prompt)
        llm = self.llm
        print("----- 推理过程启用了ChatOpenAI -----")
        reason_chain = prompt | llm | StrOutputParser()
        return reason_chain

    def get_final_chain(self, task):
        prompt = final_prompt.partial(task_description=task)
        final_chain = prompt |  self.llm | StrOutputParser()
        return final_chain
 
class AgentMemory:
    # AgentMemory所需要的 LLM 仅用于计算token，并不用于生成
    def __init__(self, llm = ChatOpenAI()):        
        # 构造一个基于Token缓存的记忆体
        self.memory = ConversationTokenBufferMemory(
            llm=llm,
            max_token_limit=4000,
        )
        
        # 初始化短时记忆
        self.memory.save_context(
            {"input": "\n初始化"},
            {"output": "\n开始"}
        )

    # 保存记忆
    def save(self, input, output):
        self.memory.save_context(input, output)

    # 提取记忆
    def load(self) -> str:
        messages = self.memory.chat_memory.messages
        string_messages = [messages[i].content for i in range(1,len(messages))]
        return "\n".join(string_messages)

# 智能体装配，run方法是主入口
class MyAgent():
    def __init__(self):
        # 最大思考步骤数
        self.max_thought_steps = 10
        # 链
        self.chain = AgentChain()
        # 记忆体
        self.short_memory = AgentMemory()

    def _step(self, task):
        # 输出LLM结果
        response = ""
        chain = self.chain.get_reason_chain(task)
        for s in chain.stream({
            "short_term_memory": self.short_memory.load(),
            "long_term_memory": "",
            "tools": render_text_description(tools)
        }):
            color_print(s, THOUGHT_COLOR, end="")
            response += s
    
        # 输出Action
        # 如果文字中包含多个JSON对象，就只保留最后一个
        action = action_output_parser.parse(response)
        color_print(action, THOUGHT_COLOR, end="")
        
        return action, response

    def _keep_last_json_paragraph(self, text):
        # 使用正则表达式找到所有的 JSON 对象
        matches = re.findall(r'(\{.*?\})(?=\s|$)', text, re.DOTALL)
    
        # 如果没有找到任何 JSON 对象，返回原始文本
        if not matches:
            return text
    
        # 获取最后一个 JSON 对象
        return matches[-1]

    def _final_step(self, task) -> str:
        """最后一步, 生成最终的输出"""
        chain = self.chain.get_final_chain(task)
        response = chain.invoke({
            "short_term_memory": self.short_memory.load()
        })
        return response
    

    from langchain.tools.base import BaseTool
    def _find_tool(self, tool_name: str) -> Optional[BaseTool]:
        for tool in tools:
            if tool.name == tool_name:
                return tool
        return None
    
    def _exec_action(self, action: Action) -> str:
        # 查找工具
        tool = self._find_tool(action.name)
        # action_expr = format_action(action)
        if tool is None:
            observation = (
                f"Error: 找不到工具或指令 '{action.name}'. "
                f"请从提供的工具/指令列表中选择，请确保按对顶格式输出。"
            )

        else:
            try:
                # 执行工具
                observation = tool.run(action.args)
            except ValidationError as e:
                # 工具的入参异常
                observation = (
                    f"Validation Error in args: {str(e)}, args: {action.args}"
                )
            except Exception as e:
                # 工具执行异常
                observation = f"Error: {str(e)}, {type(e).__name__}, args: {action.args}"

        return observation


    def run(self, task_description):
        # 实际思考步骤数
        thought_step_count = 0

        # 结论
        reply = ""

        # 思考循环
        while thought_step_count < self.max_thought_steps:
            # 思考
            action, response = self._step(task_description)

            # 结束
            if action.name == "FINISH":
                # 如果返回的动作是 FINISH 就中断循环
                color_print(f"\n----\nFINISH", OBSERVATION_COLOR)
                reply = self._final_step(task_description)
                break
            
            else:
                # 否则执行动作中指定的工具
                # 提取工具执行结果，以供思考时观察
                observation = self._exec_action(action)
                color_print(f"\n----\n结果:\n{observation}", OBSERVATION_COLOR)
                
                # 记录结果，提供观察
                self.short_memory.save(
                    {"input": response},
                    {"output": "返回结果:\n" + observation}
                )

                # 累加思考步骤数，继续思考
                thought_step_count += 1

        # 处理无法得出结论的情况
        if not reply:
            reply = "抱歉，我没能完成您的任务。"
            
        # 返回结论
        return reply