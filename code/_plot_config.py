"""公共 Matplotlib 中文字体配置，所有绘图模块统一导入此文件。"""

import platform
from matplotlib import pyplot as plt

_system = platform.system()

if _system == "Windows":
    plt.rcParams["font.sans-serif"] = ["SimHei"]
elif _system == "Darwin":  # macOS
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC"]
else:  # Linux
    plt.rcParams["font.sans-serif"] = [
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "Noto Sans CJK SC",
        "Droid Sans Fallback",
        "DejaVu Sans",
    ]

plt.rcParams["axes.unicode_minus"] = False
