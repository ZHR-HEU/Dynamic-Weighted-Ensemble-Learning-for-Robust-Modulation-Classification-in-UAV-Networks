# -*- coding: utf-8 -*-
"""Modified CNN model training for 20.mat"""
import scipy.io as scio
import numpy as np
from keras.metrics import accuracy
from keras.utils import to_categorical
from numpy import array
from numpy.f2py.crackfortran import endifs
from scipy.cluster.hierarchy import single
from scipy.integrate import quad
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras import models
from keras.layers import Flatten, Dense, BatchNormalization, Dropout, Conv1D
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import random
import seaborn as sns
from werkzeug.datastructures import Range

snrs = range(-20, 21,1)  # 信噪比范围
test = []
N=128
#

# 设置支持中文的字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False


def g(x, L, k, x0, y_min):
    return y_min + (L - y_min) / (1 + np.exp(-k * (x - x0)))
# x = np.array(snrs)
# y = np.array(test) * 100  # 转为百分比
#
# # 初始参数猜测（L, k, x0, y_min）
# p0 = [max(y), 1, np.median(x), min(y)]  # 显式设置y_min=25%
#
# # 拟合
# params, _ = curve_fit(g, x, y, p0, maxfev=10000)
# L_fit, k_fit, x0_fit, y_min_fit = params
y_min_fit=23.62
L_fit=101.10
k_fit=0.33
x0_fit=5.42
# 输出公式
print(f"拟合函数: g(x) = {y_min_fit:.2f} + ({L_fit:.2f} - {y_min_fit:.2f}) / (1 + exp(-{k_fit:.2f}(x - {x0_fit:.2f}))")


def generate_k_integers_uniform(mean_value, k):
    """
    生成 k 个整数，均值为 mean_value，且每个数在 [-20, 20] 范围内。
    """
    # 初始化结果列表
    numbers = []

    # 生成前 k-1 个数，确保最后一个数在范围内
    for i in range(k - 1):
        # 计算剩余数的总和范围
        remaining_sum = mean_value * k - sum(numbers)
        # 计算当前数的范围
        lower_bound = max(-20, remaining_sum - 20 * (k - i - 1))
        upper_bound = min(20, remaining_sum + 20 * (k - i - 1))
        # 生成当前数
        num = random.randint(lower_bound, upper_bound)
        numbers.append(num)

    # 计算最后一个数
    last_num = int(mean_value * k - sum(numbers))
    numbers.append(last_num)

    # 检查最后一个数是否在范围内
    if -20 <= last_num <= 20:
        return numbers
    else:
        raise ValueError(f"无法生成满足条件的数: 最后一个数 {last_num} 超出范围")

# 定义函数：加载数据并返回归一化后的数据和标签
def load_data(snr,flag):
    data_path = f"Dataset/test/{snr}.mat"
    data = scio.loadmat(data_path)
    x = data.get('IQ')[:, :, np.newaxis]
    x_real = x.real
    x_imag = x.imag
    x = np.concatenate((x_real, x_imag), axis=2)
    if flag==1:
        x = (x - np.mean(x)) / np.std(x)  # 归一化
        return x
    else:
        return x


# 定义函数：生成标签
def generate_labels(num):
    test_num_per_modulation = num
    y = np.vstack((
        np.zeros((test_num_per_modulation, 1)),
        np.ones((test_num_per_modulation, 1)),
        np.ones((test_num_per_modulation, 1)) * 2,
        np.ones((test_num_per_modulation, 1)) * 3
    ))
    return to_categorical(y)

# 定义函数：计算节点权重
def calculate_weights(inrs, L_fit, k_fit, x0_fit, y_min_fit):
    ratios = []
    for inr in inrs:
        avg_inr_linear = 10 ** (inr / 10)
        lambda_param = 1 / avg_inr_linear

        def f(avg_inr_linear, lambda_param):
            x_db = 10 * np.log10(avg_inr_linear)
            return g(x_db, L_fit, k_fit, x0_fit, y_min_fit) * lambda_param * np.exp(-lambda_param * avg_inr_linear)

        lower, upper = 10 ** (-20 / 10), 10 ** (18 / 10)
        integral, _ = quad(f, lower, upper, args=(lambda_param,))
        ratios.append(integral)
    inrs = np.array(inrs)
    ratios = np.array(ratios)
    # 显式增加高干噪比节点的权重
    high_inr_threshold = np.median(inrs)  # 设置高干噪比阈值
    ratios[inrs > high_inr_threshold] *= 2  # 高干噪比节点权重加倍
    # 改进后的权重过滤机制
    mean_acc = np.mean(ratios)
    std_acc = np.std(ratios)  # 计算标准差
    threshold = mean_acc - 0.5 * std_acc  # 设置动态阈值
    ratios = np.where(ratios < threshold, 0, ratios)  # 仅过滤明显低于平均值的节点

    # # 保留高干噪比节点的权重
    # min_weight = 0.1  # 设置最低权重阈值
    # ratios = np.where(ratios < min_weight, min_weight, ratios)

    # # 动态调整权重
    # ratios = ratios * (1 + inrs / np.max(inrs))  # 根据干噪比调整权重

    # 改进后的归一化
    if np.sum(ratios) > 0:
        ratios /= np.sum(ratios)
    else:
        ratios = np.ones_like(ratios) / len(ratios)  # 如果权重全为0，则均分权重
    print(f"节点权重：{ratios}")
    return ratios
import pickle
    # 定义保存字典的函数
def save_results(results, filename="results.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        print(f"结果已保存到 {filename}")
import os

# 定义加载字典的函数
def load_results(filename="results.pkl"):
    if os.path.exists(filename):
        with open(filename, "rb") as f:
            results = pickle.load(f)
        print(f"结果已从 {filename} 加载")
        return results
    else:
        print(f"文件 {filename} 不存在")
        return None
# 定义函数：计算集成准确率
def ensemble_accuracy(node_predictions, ratios, true_labels):
    proba_ensemble_weighted = np.sum([ratios[node] * node_predictions[node] for node in range(Rr)], axis=0)
    pred_labels = [np.argmax(p, axis=1) for p in node_predictions]
    vote_counts = np.zeros_like(proba_ensemble_weighted)
    for node in range(Rr):
        for i in range(len(pred_labels[node])):
            vote_counts[i, pred_labels[node][i]] += ratios[node]

    # 搜索最佳 alpha
    alpha_values = np.linspace(0, 1, 11)
    best_alpha, best_acc = 0, 0
    for alpha in alpha_values:
        proba_ensemble = alpha * proba_ensemble_weighted + (1 - alpha) * vote_counts
        predict_labels = np.argmax(proba_ensemble, axis=1)
        acc_ensemble = accuracy_score(true_labels, predict_labels)
        if acc_ensemble > best_acc:
            best_acc = acc_ensemble
            best_alpha = alpha
    print(f"best_alpha：{best_alpha}")
    # 使用最佳 alpha 计算最终集成准确率
    proba_ensemble = best_alpha * proba_ensemble_weighted + (1 - best_alpha) * vote_counts
    predict_labels = np.argmax(proba_ensemble, axis=1)
    return accuracy_score(true_labels, predict_labels)
# 主程序修改部分
if __name__ == "__main__":
    # 定义要测试的节点数列表
    Rr_list = [5, 10, 15, 20]
    num = 1000  # 每个节点数的实验次数
    results = load_results("results.pkl")
    y = generate_labels(1000)
    mean_value_range = range(-5, 16, 2)  # 测试所有INR均值

    # 存储不同节点数的结果
    all_new_avg = {Rr: [] for Rr in Rr_list}
    single=[results[mean_value]["accuracy"]for mean_value in mean_value_range]
    for Rr in Rr_list:
        print(f"\n当前测试节点数: {Rr}")
        new_avg = []

        for mean_value in mean_value_range:
            new_sum = 0

            for _ in range(num):
                # 生成INR值（增加最大尝试次数限制）
                try:
                    inrs = generate_k_integers_uniform(mean_value, Rr)
                except ValueError as e:
                    print(e)
                    continue

                # 计算权重
                ratios = calculate_weights(inrs, L_fit, k_fit, x0_fit, y_min_fit)

                # 加载节点预测结果
                node_predictions = [results[inr]["predictions"] for inr in inrs]
                true_labels = np.argmax(y, axis=1)

                # 计算集成准确率
                new_acc = ensemble_accuracy(node_predictions, ratios, true_labels)
                new_sum += new_acc

            # 保存平均准确率
            avg_acc = new_sum / num if num > 0 else 0
            new_avg.append(avg_acc)
            print(f"INR均值 {mean_value}: 平均准确率 {avg_acc:.4f}")

        all_new_avg[Rr] = new_avg
    print(f"单节点结果{single}")
    print(f"5节点结果{all_new_avg[5]}")
    print(f"10节点结果{all_new_avg[10]}")
    print(f"15节点结果{all_new_avg[15]}")
    print(f"20节点结果{all_new_avg[20]}")
    # 结果可视化
    plt.figure(figsize=(12, 7))
    markers = ['o', 's', 'D', '^', 'v']  # 不同标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # 对比色

    for idx, Rr in enumerate(Rr_list):
        plt.plot(mean_value_range,
                 all_new_avg[Rr],
                 label=f'Rr={Rr}',
                 marker=markers[idx % len(markers)],
                 color=colors[idx % len(colors)],
                 linestyle='--' if idx > 2 else '-',
                 linewidth=1.5,
                 markersize=8)
    plt.plot(mean_value_range,single,
                 label='Rr=1',
                 linestyle= '-',
                 linewidth=1.5,
                 markersize=8)

    plt.xlabel("INR均值", fontsize=12, fontweight='bold')
    plt.ylabel("平均准确率", fontsize=12, fontweight='bold')
    plt.title("多节点动态权重集成性能对比", fontsize=14, fontweight='bold')
    plt.xticks(mean_value_range[::2])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='lower right', fontsize=10)
    plt.tight_layout()

    # 保存高清图
    plt.savefig('multi_node_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()