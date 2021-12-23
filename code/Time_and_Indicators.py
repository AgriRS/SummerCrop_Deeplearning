# encoding=utf-8
import matplotlib.pyplot as plt
from pylab import *                                 #支持中文
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
font_size = 16
legend_font_size = 16
# plt.figure(figsize=(12, 9), dpi= 300)

names = ['2019-08-31 Site A', '2019-09-05 Site B', '2020-09-04 Site A', '2020-09-04 Site B', '2021-09-09 Site A',
         '2021-09-11 Site B', '2021-09-09 Site C', '2021-09-09 Site D']
x = range(len(names))



y1_256_OA = [94.1, 96.09, 95.66, 91.13, 93.54, 95.08, 96.25, 93.33]
y1_256_Kappa = [85.16, 95.31, 91.75, 89.07, 88.45, 93.06, 93.57, 91.45]
y1_256_F1score = [51.86, 85.6, 75.45, 79.59, 66.36, 75.82, 73.02, 65.42]



# plt.ylim(ymin = 300)
# plt.ylim(ymax = 3700)

plt.ylim(ymin = 40)
plt.ylim(ymax = 100)

plt.plot(x, y1_256_OA, marker='o', ms=4, color='red', label=u'OA')#, mec='b', mfc='w',label=u'Mean') #控制符号大小的 ms=10,
plt.plot(x, y1_256_Kappa, marker='o', ms=4, color='dodgerblue', label=u'Kappa')#, mec='b', mfc='w',label=u'Mean')
plt.plot(x, y1_256_F1score, marker='o', ms=4, color='green', label=u'macro F1')#, mec='b', mfc='w',label=u'Mean')

plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.xlabel('x',fontsize=font_size)
plt.ylabel('y',fontsize=font_size)


plt.legend()  # 让图例生效
plt.legend(fontsize=legend_font_size, loc = 4)
plt.xticks(x, names, rotation=20)
# plt.margins(0)
# plt.subplots_adjust(bottom=0.15)
plt.xlabel(u"") #X轴标签Bands
plt.ylabel("Accuracy / %") #Y轴标签
plt.title("") #标题
plt.grid(axis = 'y')
# plt.margins(0.1)
# plt.subplots_adjust()
# plt.savefig(r'D:\image_name.jpg', bbox_inches='tight')
plt.show()