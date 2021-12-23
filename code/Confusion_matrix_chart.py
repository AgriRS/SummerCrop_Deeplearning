from pylab import *
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
font_size = 16
#labels表示你不同类别的代号，比如这里的demo中有13个类别
labels = ['Corn', 'Peanut', 'Soybean', 'Rice', 'NCL', 'OTH', 'FL', 'Urban', 'Water', 'GH', ' PA UA']
tick_marks = np.array(range(len(labels))) + 0.5

def plot_confusion_matrix(cm, title='Confusion Matrix', cmap=plt.cm.binary):
    # plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.imshow(cm, interpolation='nearest', cmap='Greys', vmin=999999999, vmax=1000000000)
    plt.title(title)
    # plt.colorbar()
    xlocations = np.array(range(len(labels)))
    plt.xticks(xlocations, labels, rotation=0, fontsize=font_size)
    plt.yticks(xlocations, labels, fontsize=font_size)

    plt.xlabel('Deep segmentation and classification', fontsize=20)
    plt.ylabel('Ground samples', fontsize=20)

txt_Path = r"D:\SummerSentinelData\Results\ConfusionMatrix\DeepLearning\Texture\Texture_Merge_result.txt"
sub_list_data = []
f_open = open(txt_Path)
for i in range(10):
    dataList = f_open.readline()
    print(dataList)
    sub_list_data.append(dataList.split(','))

cm = np.array(sub_list_data)
# print(cm)
ind_array = np.arange(len(labels))
x, y = np.meshgrid(ind_array, ind_array)
fig, ax = plt.subplots()

for x_val, y_val in zip(x.flatten(), y.flatten()):
    c = cm[y_val][x_val]
    print(x_val)
    print(y_val)
    print('-'*50)
    if x_val == 10 or y_val == 10:
        c = ('%.2f'%c)#round(c, 2)
    else:
        c = int32(c)
    # print('----------------------------')
    # print(x_val)
    # print(y_val)
    # print(c)
    # print('----------------------------')
    if x_val == y_val:
        plt.text(x_val, y_val, c, color='Red', fontsize=17, va='center', ha='center') #"%0.2f" % (c,)
    else:
        plt.text(x_val, y_val, c, color='Black', fontsize=17, va='center', ha='center')  # "%0.2f" % (c,)

# offset the tick
plt.gca().set_xticks(tick_marks, minor=True)
plt.gca().set_yticks(tick_marks, minor=True)
plt.gca().xaxis.set_ticks_position('none')
plt.gca().yaxis.set_ticks_position('none')
plt.grid(True, which='minor', linestyle='-') #, linewidth=1
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)
plt.tick_params(bottom=False, top=False, left=False, right=False)
plt.gcf().subplots_adjust(bottom=0.15)

plot_confusion_matrix(cm, title='')
fig.tight_layout()
plt.savefig(r'D:\Gaofen_2021_2_CM.jpg',dpi = 500, bbox_inches='tight')
plt.show()

