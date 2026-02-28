import os
# os.path 是 Python 标准库 os 模块中的一个子模块，专门用于处理文件路径相关的操作。
"""
注意,get_project_root用来获取当前文件的根目录,也就是根文件夹
其通过当前文件,用两次dirname函数获取父文件夹,但要注意只适用于当前文件结构
get_abs_path用来获取绝对路径,用path.join()来拼接
"""


def get_project_root() ->str:
    # 当前文件的绝对路径 __file__是特殊变量，表示当前py文件的路径 ，abspath()返回当前文件的绝对路径
    current_file_path = os.path.abspath(__file__)

    # 获取工程的根目录，先获取文件所在的文件夹的绝对路径，然后获取父目录的绝对路径
    # dirname()返回文件的父目录
    project_root = os.path.dirname(os.path.dirname(current_file_path))
    # 返回根目录
    return project_root

def get_abs_path(relative_path:str) -> str:
    # 获取工程根目录
    project_root = get_project_root()
    # 获取绝对路径，将根目录和相对路径拼接
    abs_path = os.path.join(project_root, relative_path)
    return abs_path

if __name__ == '__main__':
    print(get_abs_path("config.txt"))
    print(get_abs_path("config/config.txt"))
    