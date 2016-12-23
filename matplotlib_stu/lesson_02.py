

import numpy as np
import matplotlib.pyplot as plt


n_groups = 5

means_men = (20, 19, 18, 17, 16)
means_women = (25, 32, 34, 20, 25)

fig, ax = plt.subplots()
index = np.arange(n_groups)
# 宽度
bar_width = 0.1

opacity = 0.4
# 参数分别是:
# 坐标, 坐标对应的数组, 宽度, 透明度, 颜色, 标题
rects1 = plt.bar(index, means_men, bar_width, alpha=opacity, color='b', label='Men')
rects2 = plt.bar(index + bar_width, means_women, bar_width, alpha=opacity, color='r', label='Women')

plt.xlabel('Group')
plt.ylabel('Scores')
plt.title('Scores by group and gender')

# x轴标题位置
plt.xticks(index + bar_width, ('A', 'B', 'C', 'D', 'E'))

plt.ylim(0, 40)
plt.legend()

plt.tight_layout()
plt.show()