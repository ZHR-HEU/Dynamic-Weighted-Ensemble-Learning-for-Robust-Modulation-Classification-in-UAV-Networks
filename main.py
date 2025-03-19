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

# 设置支持中文的字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False
N=128

# 定义CNN模型
def build_model():
    model = models.Sequential()
    model.add(Conv1D(256, 3, activation='relu', padding='same', input_shape=[N, 2]))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Conv1D(128, 3, activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))

    model.add(Dense(4, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    return model


# 加载模型
model = build_model()
model.load_weights("Model/CNN/20.hdf5")

snrs = range(-20, 20,2)  # 信噪比范围
test = []
for snr in snrs:
	hdf5_file_path = "Model/CNN/20.hdf5"
	hdf5_file_path.encode('utf-8')
	model.load_weights(hdf5_file_path)
	data_path="ideal/Dataset/test/"+str(snr)+".mat"
	data = scio.loadmat(data_path)
	x = data.get('IQ')
	x = x[:,:,np.newaxis]
	x_real = x.real
	x_imag = x.imag
	x = np.concatenate((x_real, x_imag), axis = 2)
	test_num_per_modulation = 1000
	y1 = np.zeros([test_num_per_modulation,1])
	y2 = np.ones([test_num_per_modulation,1])
	y3 = np.ones([test_num_per_modulation,1])*2
	y4 = np.ones([test_num_per_modulation,1])*3
	y = np.vstack((y1,y2,y3,y4))
	y = array(y)
	y = to_categorical(y)
	X_test=x
	Y_test=y
	[loss, acc] = model.evaluate(X_test,Y_test, batch_size=1000, verbose=0)
	test.append(acc)
print('TEST:/n{}'.format(test))
# # 将列表转为numpy数组
# x = np.array(snrs)
# y = np.array(test) * 100  # 转为百分比形式
# # 定义逻辑斯蒂函数
# def g(x, L, k, x0):
#     return L / (1 + np.exp(-k * (x - x0)))
#
# # 使用curve_fit进行Sigmoid拟合
# p0 = [max(y), 1, np.median(x)]  # 初始参数猜测
# params, _ = curve_fit(g, x, y, p0, maxfev=10000)
#
# # 生成拟合曲线数据点
# x_fit = np.linspace(min(x)-2, max(x)+2, 100)
# y_fit = g(x_fit, *params)
#
# # 打印拟合参数
# L, k, x0 = params
# print("拟合函数公式:", f"g(x) = {L:.4f} / (1 + exp(-{k:.4f} * (x - {x0:.4f})))")
# 定义带偏移的Sigmoid函数
def g(x, L, k, x0, y_min):
    return y_min + (L - y_min) / (1 + np.exp(-k * (x - x0)))
x = np.array(snrs)
y = np.array(test) * 100  # 转为百分比

# 初始参数猜测（L, k, x0, y_min）
p0 = [max(y), 1, np.median(x), min(y)]  # 显式设置y_min=25%

# 拟合
params, _ = curve_fit(g, x, y, p0, maxfev=10000)
L_fit, k_fit, x0_fit, y_min_fit = params

# 输出公式
print(f"拟合函数: g(x) = {y_min_fit:.2f} + ({L_fit:.2f} - {y_min_fit:.2f}) / (1 + exp(-{k_fit:.2f}(x - {x0_fit:.2f}))")

# 生成拟合曲线
x_fit = np.linspace(min(x), max(x), 100)
y_fit = g(x_fit, *params)
# 定义函数：生成五个整数，均值为 mean_value
def generate_five_integers_uniform(mean_value):
    while True:
        numbers = [random.randint(-20, 20) for _ in range(4)]
        fifth_number = int(mean_value * 5 - sum(numbers))
        if -20 <= fifth_number <= 20:
            numbers.append(fifth_number)
            return numbers


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
def generate_labels():
    test_num_per_modulation = 1000
    y = np.vstack((
        np.zeros((test_num_per_modulation, 1)),
        np.ones((test_num_per_modulation, 1)),
        np.ones((test_num_per_modulation, 1)) * 2,
        np.ones((test_num_per_modulation, 1)) * 3
    ))
    return to_categorical(y)

Rr=5
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

    mean_acc = np.mean(ratios)
    ratios = np.where(ratios < mean_acc, 0, ratios)
    if np.sum(ratios) > 0:
        ratios /= np.sum(ratios)
    else:
        ratios = np.ones_like(ratios) / len(ratios)
    print(f"节点权重：{ratios}")
    return ratios
import pickle

# 定义函数：计算 Weighted Voting (WV) 准确率
def weighted_voting_accuracy(node_predictions, ratios, true_labels):
    """
    计算 Weighted Voting 的准确率。
    :param node_predictions: 所有节点的预测结果列表，每个元素是一个节点的预测概率数组。
    :param ratios: 节点权重列表。
    :param true_labels: 真实标签。
    :return: Weighted Voting 的准确率。
    """
    # 获取每个节点的预测类别
    pred_labels = [np.argmax(p, axis=1) for p in node_predictions]

    # 加权投票
    final_predictions = []
    for i in range(len(true_labels)):
        # 统计每个类别的加权票数
        votes = np.zeros(node_predictions[0].shape[1])  # 初始化票数数组
        for node in range(len(node_predictions)):
            votes[pred_labels[node][i]] += ratios[node]
        # 选择加权票数最多的类别
        final_predictions.append(np.argmax(votes))

    # 计算准确率
    return accuracy_score(true_labels, final_predictions)

# 定义函数：计算 Direct Voting (DV) 准确率
def direct_voting_accuracy(node_predictions, true_labels):
    """
    计算 Direct Voting 的准确率。
    :param node_predictions: 所有节点的预测结果列表，每个元素是一个节点的预测概率数组。
    :param true_labels: 真实标签。
    :return: Direct Voting 的准确率。
    """
    # 获取每个节点的预测类别
    pred_labels = [np.argmax(p, axis=1) for p in node_predictions]

    # 多数投票
    final_predictions = []
    for i in range(len(true_labels)):
        # 统计每个类别的票数
        votes = np.zeros(node_predictions[0].shape[1])  # 初始化票数数组
        for node in range(len(node_predictions)):
            votes[pred_labels[node][i]] += 1
        # 选择票数最多的类别
        final_predictions.append(np.argmax(votes))

    # 计算准确率
    return accuracy_score(true_labels, final_predictions)

# 定义函数：计算 Weighted Average (WA) 准确率
def weighted_average_accuracy(node_predictions, ratios, true_labels):
    """
    计算 Weighted Average 的准确率。
    :param node_predictions: 所有节点的预测结果列表，每个元素是一个节点的预测概率数组。
    :param ratios: 节点权重列表。
    :param true_labels: 真实标签。
    :return: Weighted Average 的准确率。
    """
    # 加权平均
    weighted_proba = np.zeros_like(node_predictions[0])
    for node in range(len(node_predictions)):
        weighted_proba += ratios[node] * node_predictions[node]

    # 获取最终预测类别
    final_predictions = np.argmax(weighted_proba, axis=1)

    # 计算准确率
    return accuracy_score(true_labels, final_predictions)

    # 定义保存字典的函数
def save_results(results, filename="results.pkl"):
        with open(filename, "wb") as f:
            pickle.dump(results, f)
        print(f"结果已保存到 {filename}")

# 主程序
if __name__ == "__main__":
    # 初始化结果存储
    snrs = range(-20, 21,1)
    results= {snr: {"accuracy": 0, "predictions": []} for snr in snrs}
    # results_1= {snr: {"accuracy": 0, "predictions": []} for snr in snrs}
    single=[]
    # 预计算每个信噪比下的准确率和预测结果
    for snr in snrs:
        x = load_data(snr,0)
        y = generate_labels()
        loss, acc = model.evaluate(x, y, batch_size=1000, verbose=0)
        results[snr]["accuracy"] = acc
        results[snr]["predictions"] = model.predict(x, batch_size=1000)
        single.append(results[snr]["accuracy"])
    # save_results(results, "results.pkl")
    #     x = load_data(snr, 1)
    #     y = generate_labels()
    #     loss, acc = model.evaluate(x, y, batch_size=1000, verbose=0)
    #     results_1[snr]["accuracy"] = acc
    #     results_1[snr]["predictions"] = model.predict(x, batch_size=1000)



    # 定义均值范围和结果存储
    mean_value_range = range(-20, 21, 1)
    new_avg, DA_avg, DV_avg, WA_avg, WV_avg = [], [], [], [], []
    num=1000
    for mean_value in mean_value_range:
        new_sum, DA_sum, DV_sum, WA_sum, WV_sum = 0, 0, 0, 0, 0
        count=1;
        for _ in range(num):
            print(f"这是第{mean_value}的第 {count} 次循环")
            count = count + 1
            inrs = generate_five_integers_uniform(mean_value)
            ratios = calculate_weights(inrs, L_fit, k_fit, x0_fit, y_min_fit)
            node_predictions = [results[inr]["predictions"] for inr in inrs]
            true_labels = np.argmax(y, axis=1)

            # 计算集成准确率
            new_acc = ensemble_accuracy(node_predictions, ratios, true_labels)
            new_sum += new_acc
            # node_predictions = [results_1[inr]["predictions"] for inr in inrs]
            # 计算 Direct Voting (DV) 准确率
            dv_acc = direct_voting_accuracy(node_predictions, true_labels)
            DV_sum += dv_acc

            # 计算 Weighted Voting (WV) 准确率
            wv_acc = weighted_voting_accuracy(node_predictions, ratios, true_labels)
            WV_sum += wv_acc

            # 计算 Weighted Average (WA) 准确率
            wa_acc = weighted_average_accuracy(node_predictions, ratios, true_labels)
            WA_sum += wa_acc

            # 计算 Direct Average (DA) 准确率
            DA_acc = np.mean([results[inr]["accuracy"] for inr in inrs])
            DA_sum += DA_acc

        # 计算 10 次的平均准确率
        new_avg.append(new_sum / num)
        DA_avg.append(DA_sum / num)
        DV_avg.append(DV_sum / num)
        WA_avg.append(WA_sum / num)
        WV_avg.append(WV_sum / num)
    # 输出结果
    print(f"最终集成平均准确率: {new_avg}")
    print(f"DA 平均准确率: {DA_avg}")
    print(f"DV 平均准确率: {DV_avg}")
    print(f"WA 平均准确率: {WA_avg}")
    print(f"WV 平均准确率: {WV_avg}")
    print(f"单个节点平均准确率: {single}")
    x = list(range(-20, 21, 1))
    # 结果可视化
    plt.figure(figsize=(10, 6))
    plt.plot(x,single,label='Single Node', marker='o')
    plt.plot(x,new_avg, label='New', marker='*')
    plt.plot(x,DA_avg, label='DA', marker='o')
    plt.plot(x,DV_avg, label='DV', marker='o')
    plt.plot(x,WA_avg, label='WA', marker='o')
    plt.plot(x,WV_avg, label='WV', marker='o')
    plt.xlabel("INR均值")
    plt.ylabel("准确率")
    plt.legend()
    plt.show()