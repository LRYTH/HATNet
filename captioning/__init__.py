import os

def get_project_root():
    """返回项目根目录 (HATNet/) 的绝对路径"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
