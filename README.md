# agent-demo
## Agent整体架构
![](./images/agent-overview.png)
## Agent实现流程
![](./images/agent-flowchart.png)
*【代码中长期记忆部分未实现】*

## 基本功能
代码可以自动推理规划使用下面的工具完成用户的任务:
- ListDirectory: ListDirectory(path: str) -> str - 探查文件夹的内容和结构，展示它的文件名和文件夹名
- AskDocument: AskDocument(filename: str, query: str) -> str - 根据一个Word或PDF文档的内容，回答一个问题。考虑上下文信息，确保问题对相关概念的定义表述完整。
- GenerateDocument: GenerateDocument(query: str) - 根据需求描述生成一篇正式文档
- SendEmail: SendEmail(to: str, subject: str, body: str, cc: str = None, bcc: str = None) -> str - 给指定的邮箱发送邮件。确保邮箱地址是xxx@xxx.xxx的格式。多个邮箱地址以';'分割。
- InspectExcel: InspectExcel(filename: str, n: int = 3) -> str - 探查表格文件的内容和结构，展示它的列名和前n行，n默认为3。 使用该函数时应当准备提供filename和n两个参数，其中： - filename：要探查的Excel文件名 - n: 默认的行数
- AnalyseExcel: AnalyseExcel(query, filename) - 通过pandas数据处理脚本分析一个结构化文件（例如excel文件）的内容。 输人中必须包含文件的完整路径和具体分析方式和分析依据，阈值常量等。 如果输入信息不完整，你可以拒绝回答。

基于data目录下的文件，可以提问的任务示例：
 - 9月份的销售额是多少
 - 销售总额最大的产品是什么
 - 帮我找出销售额不达标的供应商
 - 对比8月和9月销售情况，写一份报告