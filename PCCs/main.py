import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import process_data
import get_pre_data
def get_R(x1, x2, dim=1, func=pearsonr):
    # x1,x2为np对象 counts
    r1, p1 = [], []
    for g in range(x1.shape[dim]):  # 遍历列
        if dim == 1:
            r, pv = func(x1[:, g], x2[:, g])
        elif dim == 0:
            r, pv = func(x1[g, :], x2[g, :])
        r1.append(r)
        p1.append(pv)
    r1 = np.array(r1)
    p1 = np.array(p1)
    return r1, p1

def boxes(**dic):
    # 键是名字 值是一个列表，是一组数据，一个值（list）做出一个箱子
    labels = []
    datas = []

    box_colors = ['lightcoral', 'lightblue', 'lightgreen', 'lightpink', 'lightyellow', 'lightgray']

    for label, data in dic.items():
        labels.append(label)
        datas.append(data)

    fig, ax = plt.subplots()
    ax.set_facecolor('#f0f0f0')
    ax.yaxis.grid(True, linestyle='-', linewidth=1.5, alpha=0.7, color='white')
    ax.xaxis.grid(True, linestyle='-', linewidth=1.5, alpha=0.7, color='white')


    bp = ax.boxplot(datas,
                patch_artist=True,widths=0.8,
                medianprops={'color': 'black', 'linewidth': '1.5'},
                flierprops=dict(marker='', color='red', markersize=5, linestyle='none'),
                labels=labels)

    for box, color in zip(bp['boxes'], box_colors):
        box.set_facecolor(color)

    plt.yticks(np.arange(0.2, 0.8, 0.1))  # y坐标
    plt.ylim(0., 1.)
    plt.title('PCCs between imputed and true expression')
    plt.show()

if __name__ == '__main__':
    truth_counts, stEnTrans_data, interpolation_data = process_data.get_mel1_rep1()

    stEnTrans = get_pre_data.get_data_stEnTrans(stEnTrans_data, load='C:\\Users\DELL\PycharmProjects\stEnTrans\PCCs\stEnTrans_mel_p3h4.params', patch_size=3, truth_counts=truth_counts, num_heads=4)

    DIST_self = get_pre_data.get_data_DIST(stEnTrans_data, load='C:\\Users\DELL\PycharmProjects\stEnTrans\PCCs\DIST_mel.params', truth_counts=truth_counts)

    NEDI = get_pre_data.get_data_NEDI(stEnTrans_data, truth_counts)

    Linear = get_pre_data.get_data_interpolation(interpolation_data, method='linear', truth_counts=truth_counts)

    Cubic = get_pre_data.get_data_interpolation(interpolation_data, method='cubic', truth_counts=truth_counts)

    NN = get_pre_data.get_data_interpolation(interpolation_data, method='nearest', truth_counts=truth_counts)

    boxes(stEnTrans=stEnTrans, DIST=DIST_self, Linear=Linear, Cubic=Cubic, NN=NN, NEDI=NEDI)

