import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
plt.figure(figsize=(12,9))
plt.rcParams['font.sans-serif'] = ['Times New Roman']


input_txt = r'D:\SummerSentinelData\20211120\Model_results_Paper\UNET++_20Bands_with_no_sampling.txt'
x1 = []
x2 = []
yloss_1 = []
yloss_2 = []
ymiou_1 = []
ymiou_2 = []
f = open(input_txt)

for line in f:
    line = line.strip('\n')
    line = line.split(',')
    print(line)
    if line[1] != '':
        if line[3] != '':
            x1.append(int(line[0]))
            x2.append(int(line[0]))
            yloss_1.append(float(line[1]))
            ymiou_1.append(float(line[2]))
            yloss_2.append(float(line[3]))
            ymiou_2.append(float(line[4]))
        else:
            x1.append(int(line[0]))
            yloss_1.append(float(line[1]))
            ymiou_1.append(float(line[2]))
    else:
        x2.append(int(line[0]))
        yloss_2.append(float(line[3]))
        ymiou_2.append(float(line[4]))

f.close

#Sentinel  0.35-0.8 Timm-RegNetY-320 0.3
plt.ylim(ymin = 0.3)
plt.ylim(ymax = 1.2)


plt.plot(x1, yloss_1, marker='o', ms=0, label= u'Loss with up-sampling', color='r', linestyle='-', linewidth=1.2)
plt.plot(x1, ymiou_1, marker='o', ms=0, label= u'mIoU without up-sampling', color='#1677bb', linestyle='-', linewidth=1.2)
plt.plot(x2, yloss_2, marker='o', ms=0, label= u'Loss without up-sampling', color='darkred', linewidth=1.2)
plt.plot(x2, ymiou_2, marker='o', ms=0, label= u'mIoU with up-sampling', color='b',linewidth=1.2)

plt.xticks(x1[0:len(x1):1], x1[0:len(x1):1], rotation=0)

xMax = max(x1)
plt.xticks(range(0, xMax, 10), fontsize=22)
plt.yticks(fontsize=22)
plt.xlabel('x', fontsize=22)
plt.ylabel('y', fontsize=22)

plt.margins(0)
plt.legend()  # 让图例生效
plt.legend(fontsize=10)

plt.xlabel("Epoch", fontsize=22)
# plt.ylabel("Epoch average time / minute", fontsize=22)
plt.ylabel("Loss and mIoU", fontsize=10)

# plt.title("Patch size", fontsize=28, y = 0.92)
plt.tick_params(axis="both")
plt.rcParams['savefig.dpi'] = 300  # 图片像素

plt.show()
