
# ### Tool的包装函数
# - 将异常转换为文字
# - 将输出结果转换为文本
def _safe_func_call(func, *args, **kwargs):
    """Call a function with any arguments, return error message if an exception is raised"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        return str(e)

from functools import wraps

def make_safe_tool(func):
    """Create a new function that wraps the given function with error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return str(func(*args, **kwargs))
        except Exception as e:
            return str(e)
    return wrapper

# ### 中文转换工具
# 后面在将Pandtic类型转换为提示语的一部份时会用到，没有这部份会输出Unicode编码，很不友好。
def chinese_friendly(string) -> str:
    lines = string.split('\n')
    for i, line in enumerate(lines):
        if line.startswith('{') and line.endswith('}'):
            try:
                lines[i] = json.dumps(json.loads(line), ensure_ascii=False)
            except:
                pass
    return '\n'.join(lines)
