from colorama import init, Fore, Back, Style
import sys
from langchain.agents import Tool
from langchain.tools import StructuredTool
import os
from common import *


# ### 着色打印工具
THOUGHT_COLOR = Fore.GREEN
OBSERVATION_COLOR = Fore.YELLOW
ROUND_COLOR = Fore.RED
CODE_COLOR = Fore.BLUE
def color_print(text, color=None, end="\n"):
    if color is not None:
        content = color + text + Style.RESET_ALL + end
    else:
        content = text + end
    sys.stdout.write(content)
    sys.stdout.flush()
# 使用示例：
# color_print("我现在开始思考...", color = THOUGHT_COLOR)
# color_print("第一轮开始了", color = ROUND_COLOR)
# color_print("def hello(): \n    print('hi')", color = CODE_COLOR)

def list_files_in_directory(path: str) -> str:
    """List all file names in the directory"""
    file_names = os.listdir(path.strip())

    return "\n".join(file_names)

# 参数中的文件目录需要修改为你自己的目录
#print(list_files_in_directory("../../"))
#print(list_files_in_directory("./data/autogpt-demo"))
# #### 定义 tool 函数
# 输出工具
directory_inspection_tool = StructuredTool.from_function(
    func=make_safe_tool(list_files_in_directory),
    name="ListDirectory",
    description="探查文件夹的内容和结构，展示它的文件名和文件夹名",
)

#print(make_safe_tool(list_files_in_directory)("./data/autogpt-demo"))
#print(directory_inspection_tool.invoke("./data/autogpt-demo"))


from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
# #### 使用 LLM 生成文档
def write(query: str):
    """按用户要求生成文章"""
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("你是专业的文档写手。你根据客户的要求，写一份文档。输出中文。"),
            HumanMessagePromptTemplate.from_template("{query}"),
        ]
    )

    chain = {"query": RunnablePassthrough()} | template | ChatOpenAI() | StrOutputParser()

    chunks = []
    for chunk in chain.stream(query):
        chunks.extend(chunk)
        # yield chunk
        print(chunk, end="", flush=True)
        
    return chunks
 
# #### 定义 tool 函数
# 输出工具
document_generation_tool = StructuredTool.from_function(
    func=make_safe_tool(write),
    name="GenerateDocument",
    description="根据需求描述生成一篇正式文档",
)
# #### 使用示例
# 示例
#for chunk in document_generation_tool.stream("写一封邮件给张三，内容是：你好，我是李四。"):
#    print(chunk)


# ### 发送Email
# #### 准备
import webbrowser
import urllib.parse
import re
# #### 检查email格式是否合法
def _is_valid_email(email: str) -> bool:
    receivers = email.split(';')
    # 正则表达式匹配电子邮件
    pattern = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    for receiver in receivers:
        if not bool(re.match(pattern, receiver.strip())):
            return False
    return True

 
# #### 触发系统调用发送邮件
def send_email(
        to: str,
        subject: str,
        body: str,
        cc: str = None,
        bcc: str = None,
) -> str:
    """给指定的邮箱发送邮件"""

    if not _is_valid_email(to):
        return f"电子邮件地址 {to} 不合法"

    # 对邮件的主题和正文进行URL编码
    subject_code = urllib.parse.quote(subject)
    body_code = urllib.parse.quote(body)

    # 构造mailto链接
    mailto_url = f'mailto:{to}?subject={subject_code}&body={body_code}'
    if cc is not None:
        cc = urllib.parse.quote(cc)
        mailto_url += f'&cc={cc}'
    if bcc is not None:
        bcc = urllib.parse.quote(bcc)
        mailto_url += f'&bcc={bcc}'

    webbrowser.open(mailto_url)

    return f"状态: 成功\n备注: 已发送邮件给 {to}, 标题: {subject}"

# #### 定义 tool 函数
# 发送邮件
email_tool = StructuredTool.from_function(
    func=make_safe_tool(send_email),
    name="SendEmail",
    description="给指定的邮箱发送邮件。确保邮箱地址是xxx@xxx.xxx的格式。多个邮箱地址以';'分割。",
)

#send_email("43801@qq.com", "hello", "happy new year!")


import re
from langchain.tools import StructuredTool
from langchain_core.output_parsers import BaseOutputParser
# from Utils.PythonExecUtil import execute_python_code
from langchain_openai import ChatOpenAI
from langchain_experimental.utilities import PythonREPL
# #### 自定义一个OutputParse
class PythonCodeParser(BaseOutputParser):
    """从OpenAI返回的文本中提取Python代码。"""

    def _remove_marked_lines(self, input_str: str) -> str:
        lines = input_str.strip().split('\n')
        if lines and lines[0].strip().startswith('```'):
            del lines[0]
        if lines and lines[-1].strip().startswith('```'):
            del lines[-1]

        ans = '\n'.join(lines)
        return ans

    def parse(self, text: str) -> str:
        # 使用正则表达式找到所有的Python代码块
        python_code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        # 从re返回结果提取出Python代码文本
        python_code = None
        if len(python_code_blocks) > 0:
            python_code = python_code_blocks[0]
            python_code = self._remove_marked_lines(python_code)
        return python_code

 
# #### 定义提示语模板
from langchain.prompts import PromptTemplate
excel_analyser_prompt = PromptTemplate.from_template("""
你的任务是先分析，再生成代码。

请根据用户的输入，一步步分析：
（1）用户的输入是否依赖某个条件，而这个条件没有明确赋值？
（2）我是否需要对某个变量的值做假设？
（3）已经从用户的输入中拆解概念，将其中包含的数字或实体名称，映射为所生成的函数入参，并在代码中使用？
（4）不能生成用户输入中没有包含的函数入参，将导致严重后果，这一点是否已经确认？

如果我需要对某个变量的值做假设，请直接输出：
```python
print("我需要知道____的值，才能生成代码。请完善你的查询。") # 请将____替换为需要假设的的条件
```
否则，创建Python代码，分析指定文件的内容。

MUST 确保你生成的代码按照固定格式：先定义一个带有参数的函数，再执行该函数完成分析任务。

MUST 请不要将问题中的数字、名称写死在代码中，而是提取问题中对应的数字、名称等作为你定义的函数的入参变量，
并在代码中做 ==、>、< 等逻辑判断时使用这些入参变量；
MUST 确保问题中的所有数字或实体名称都做了上述的入参映射，不要遗漏。
MUST 确保所有入参都使用相应默认值。

MUST 请不要使用filename作为入参变量，直接写死在代码里即可。

MUST 请在函数定义时增加文档字符串，针对问题总结函数的用途，并同时说明各参数的用途

MUST 你生成代码中所有的常量都必须来自我给你的信息或来自文件本身。不要编造任何常量。
如果常量缺失，你的代码将无法运行。你可以拒绝生成代码，但是不要生成编造的代码。
确保你生成的代码最终以print的方式输出结果(回答用户的问题)。

MUST 你可以使用的库只包括：pandas, re, math, datetime, openpyxl
确保你的代码只使用上述库，否则你的代码将无法运行。
当你引用这些库时，请在函数内部执行import。

MUST 确保你的代码可以通过运行的。

给定文件为：
{{filename}}

文件内容样例：
{{inspections}}

你输出的Python代码前后必须有markdown标识符，如下所示：
```python
# example code
def hello(product):
    print(product)
```

用户输入：
{{query}}
""", template_format="jinja2")

# #### 定义执行链
llm = ChatOpenAI(
        model="gpt-4-0125-preview",
        temperature=0,
        # model_kwargs={"seed": 42},
    )
analysis_chain = excel_analyser_prompt | llm | PythonCodeParser()

# #### 生成 python 代码并执行
import ast
import types
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_experimental.utilities import PythonREPL

def excel_analyse(query, filename):
    """分析一个结构化文件（例如excel文件）的内容。"""

    filename = filename.strip()
    # columns = get_column_names(filename)
    inspections = get_first_n_rows(filename, 3)

    code = ""

    # 打印详细信息
    color_print("\n#!/usr/bin/env python", CODE_COLOR, end="\n")

    # 生成代码
    for c in analysis_chain.stream({
        "query": query,
        "filename": filename,
        "inspections": inspections
    }):
        ## 打印详细信息
        color_print(c, CODE_COLOR, end="")
        ## 收集代码成果
        code += c

    if code:
        # 动态创建工具函数   
        tree = ast.parse(code)
        func_def = tree.body[0]
        func_str = ast.unparse(func_def)
        exec(func_str)
        
        #func = globals()[func_def.name]
        #tool = convert_to_openai_tool(func)
        #print("\n New Tool:")
        #print(tool)
        
        # 执行代码
        return PythonREPL().run(code)
        # return newTool.func()
    else:
        return "没有找到可执行的Python代码"
# 输出工具
excel_analysis_tool = StructuredTool.from_function(
    func=make_safe_tool(excel_analyse),
    name="AnalyseExcel",
    description="""
        通过pandas数据处理脚本分析一个结构化文件（例如excel文件）的内容。
        输人中必须包含文件的完整路径和具体分析方式和分析依据，阈值常量等。
        如果输入信息不完整，你可以拒绝回答。
    """,
)
# #### 使用示例
#ret = excel_analysis_tool.invoke({"query": "8月份手机销售额最大的公司是？", "filename":"./data/autogpt-demo/2023年8月-9月销售记录.xlsx"})




# ### Excel结构探查
# #### 探查Excel的sheet、列名和前N行数据
import pandas as pd
 
def get_sheet_names(
        filename : str
) -> str :
    """获取 Excel 文件的工作表名称"""
    excel_file = pd.ExcelFile(filename.strip())
    sheet_names = excel_file.sheet_names
    return f"这是 '{filename}' 文件的工作表名称：\n\n{sheet_names}"

 
def get_column_names(
        filename : str
) -> str:
    """获取 Excel 文件的列名"""

    # 读取 Excel 文件的第一个工作表
    df = pd.read_excel(filename.strip(), sheet_name=0)  # sheet_name=0 表示第一个工作表
    column_names = '\n'.join(
        df.columns.to_list()
    )

    result = f"这是 '{filename.strip()}' 文件第一个工作表的列名：\n\n{column_names}"
    return result

 
def get_first_n_rows(
        filename : str,
        n : int = 3
) -> str :
    """获取 Excel 文件的前 n 行"""

    filename = filename.strip()
    result = get_sheet_names(filename)+"\n\n"
    result += get_column_names(filename)+"\n\n"

    # 读取 Excel 文件的第一个工作表
    df = pd.read_excel(filename, sheet_name=0)  # sheet_name=0 表示第一个工作表
    n_lines = '\n'.join(
        df.head(n).to_string(index=False, header=True).split('\n')
    )

    result += f"这是 '{filename}' 文件第一个工作表的前{n}行样例：\n\n{n_lines}"
    return result

 
# #### 定义 tool 函数
 
# <div class="alert alert-info">
# <b>增加Tool的参数说明：</b><br/>
#     实践中，我发现大模型有时会弄错参数，虽然会迭代纠正，但还是习惯性做了优化，在工具说明中补充了参数说明。
# </div>

# 输出工具
excel_inspection_tool = StructuredTool.from_function(
    func=make_safe_tool(get_first_n_rows),
    name="InspectExcel",
    description="""
    探查表格文件的内容和结构，展示它的列名和前n行，n默认为3。
    
    使用该函数时应当准备提供filename和n两个参数，其中：
    
    - filename：要探查的Excel文件名
    - n: 默认的行数
    
    """,
)

#print(get_first_n_rows("./data/autogpt-demo/供应商名录.xlsx"))
#print(excel_inspection_tool.invoke("./data/autogpt-demo/供应商名录.xlsx"))


finish_placeholder = StructuredTool.from_function(
    func=lambda: None,
    name="FINISH",
    description="用于表示任务完成的占位符工具"
)
