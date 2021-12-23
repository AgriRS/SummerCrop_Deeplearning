import matplotlib.pyplot as plt
# from matplotlib.pyplot import MultipleLocator
# plt.figure(figsize=(10,8))
plt.rcParams['font.sans-serif'] = ['Times New Roman']

font_size_user = 20
#GF_without_upsapling_256_9Bands
input_txt = r'D:\2020SentinelData\CropJ_add_results\S_2A_256__nosampling.txt'
x1 = []
# x2 = []
yloss_1 = []
# yloss_2 = []
ymiou_1 = []
# ymiou_2 = []
f = open(input_txt)

for line in f:
    line = line.strip('\n')
    line = line.split(',')
    print(line)

    if line[1] != '':
        x1.append(int(line[0]))
        yloss_1.append(float(line[1]))
        ymiou_1.append(float(line[2]))
    else:
        x1.append(int(line[0]))
        yloss_1.append(float(line[1]))
        ymiou_1.append(float(line[2]))

f.close


plt.plot(x1, yloss_1, marker='o', ms=0, label= u'Loss', color='darkred', linestyle='-', linewidth=1) #tomato
plt.plot(x1, ymiou_1, marker='o', ms=0, label= u'mIoU', color='b', linestyle='-', linewidth=1)  ##1677bb

plt.xticks(fontsize=font_size_user)
plt.yticks(fontsize=font_size_user)
plt.xlabel('x', fontsize=font_size_user)
plt.ylabel('y', fontsize=font_size_user)

plt.margins(0)
plt.legend()  # 让图例生效
plt.legend(fontsize=font_size_user)

plt.xlabel("Epochs", fontsize=font_size_user)
plt.ylabel("Loss and mIoU", fontsize=font_size_user)


# plt.title("Patch size", fontsize=font_size_user, y = 0.92)
plt.tick_params(axis="both")
plt.rcParams['savefig.dpi'] = 300  # 图片像素

# plt.rcParams['figure.figsize'] = (10, 10)  # 尺寸
plt.show()
