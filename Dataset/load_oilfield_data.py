import os
import numpy as np
import logging
import random
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler, LabelEncoder

logger = logging.getLogger(__name__)


def load_from_ts_file(file_path):
    """
    自定义函数，从.ts文件加载数据和标签
    """
    data_list = []
    labels = []

    # 读取文件
    with open(file_path, 'r') as f:
        lines = f.readlines()

    # 找到数据开始的位置
    data_start_idx = -1
    for i, line in enumerate(lines):
        if line.strip() == '@data':
            data_start_idx = i + 1
            break

    if data_start_idx == -1:
        raise ValueError(f"无法在文件中找到数据部分: {file_path}")

    # 解析数据和标签
    i = data_start_idx
    while i < len(lines):
        # 数据行
        if i < len(lines) and "," in lines[i]:
            values_str = lines[i].strip()
            values = [float(x) for x in values_str.split(',')]
            data_list.append(values)

            # 标签行
            if i + 1 < len(lines):
                label = lines[i + 1].strip()
                labels.append(label)
                i += 2
            else:
                break
        else:
            i += 1

    return data_list, np.array(labels)


def fill_missing(x: np.array, max_len: int, vary_len: str = "suffix-noise", normalise: bool = True):
    """
    填充缺失值，与UEA数据加载器类似
    """
    if vary_len == "zero":
        if normalise:
            x = StandardScaler().fit_transform(x)
        x = np.nan_to_num(x)
    elif vary_len == 'prefix-suffix-noise':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)
            diff_len = int(0.5 * (max_len - seq_len))

            for j in range(diff_len):
                x[i, j] = random.random() / 1000

            for j in range(diff_len, diff_len + seq_len):
                x[i, j] = series[j - diff_len]

            for j in range(diff_len + seq_len, max_len):
                x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    elif vary_len == 'uniform-scaling':
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)
            if seq_len == 0:
                continue

            for j in range(max_len):
                scaling_factor = int(j * seq_len / max_len)
                if scaling_factor < seq_len:
                    x[i, j] = series[scaling_factor]
            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]
    else:
        # suffix-noise: 原始序列在前，后面填充小随机噪声
        for i in range(len(x)):
            series = list()
            for a in x[i, :]:
                if np.isnan(a):
                    break
                series.append(a)
            series = np.array(series)
            seq_len = len(series)

            # 保留原始序列，后面填充小随机噪声
            x[i, :seq_len] = series
            for j in range(seq_len, max_len):
                x[i, j] = random.random() / 1000

            if normalise:
                tmp = StandardScaler().fit_transform(x[i].reshape(-1, 1))
                x[i] = tmp[:, 0]

    return x


def process_ts_data(data_list, max_len, vary_len: str = "suffix-noise", normalise: bool = False):
    """
    处理时间序列数据为统一格式
    """
    num_instances = len(data_list)
    # 对于油田数据，只有一个维度
    num_dim = 1

    # 创建输出数组 [样本数, 维度=1, 序列长度]
    output = np.zeros((num_instances, num_dim, max_len), dtype=np.float64)

    # 填充数据
    for i in range(num_instances):
        series = data_list[i]
        length = min(len(series), max_len)
        output[i, 0, :length] = series[:length]

    # 处理缺失值和标准化
    for i in range(num_dim):
        output[:, i, :] = fill_missing(output[:, i, :], max_len, vary_len, normalise)

    return output


def mean_std(train_data):
    """
    计算均值和标准差，用于标准化
    """
    m_len = np.mean(train_data, axis=2)
    mean = np.mean(m_len, axis=0)

    s_len = np.std(train_data, axis=2)
    std = np.max(s_len, axis=0)

    return mean, std


def mean_std_transform(train_data, mean, std):
    """
    使用均值和标准差进行标准化
    """
    return (train_data - mean) / std


def load(config):
    """
    加载油田数据
    """
    # 构建数据字典
    Data = {}
    problem = config['data_dir'].split('/')[-1]

    # 检查是否存在预处理数据
    if os.path.exists(config['data_dir'] + '/' + problem + '.npy'):
        logger.info("Loading preprocessed data ...")
        Data_npy = np.load(config['data_dir'] + '/' + problem + '.npy', allow_pickle=True)

        Data['max_len'] = Data_npy.item().get('max_len')
        Data['All_train_data'] = Data_npy.item().get('All_train_data')
        Data['All_train_label'] = Data_npy.item().get('All_train_label')
        Data['train_data'] = Data_npy.item().get('train_data')
        Data['train_label'] = Data_npy.item().get('train_label')
        Data['val_data'] = Data_npy.item().get('val_data')
        Data['val_label'] = Data_npy.item().get('val_label')
        Data['test_data'] = Data_npy.item().get('test_data')
        Data['test_label'] = Data_npy.item().get('test_label')

        logger.info("{} samples will be used for training".format(len(Data['train_label'])))
        logger.info("{} samples will be used for validation".format(len(Data['val_label'])))
        logger.info("{} samples will be used for testing".format(len(Data['test_label'])))

    else:
        logger.info("Loading and preprocessing data ...")
        train_file = config['data_dir'] + "/" + problem + "_TRAIN.ts"
        test_file = config['data_dir'] + "/" + problem + "_TEST.ts"

        # 使用自定义函数加载数据
        train_data_list, y_train_raw = load_from_ts_file(train_file)
        test_data_list, y_test_raw = load_from_ts_file(test_file)

        logger.info("Train samples: {}, Test samples: {}".format(len(train_data_list), len(test_data_list)))

        # 编码标签
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(y_train_raw)
        y_test = label_encoder.transform(y_test_raw)

        # 找出最大序列长度
        train_lengths = [len(x) for x in train_data_list]
        test_lengths = [len(x) for x in test_data_list]
        max_seq_len = max(max(train_lengths), max(test_lengths))
        logger.info(f"Maximum sequence length: {max_seq_len}")

        # 处理数据 - 将其转换为 [样本数, 维度=1, 序列长度] 的格式
        X_train = process_ts_data(train_data_list, max_seq_len, vary_len="suffix-noise", normalise=False)
        X_test = process_ts_data(test_data_list, max_seq_len, vary_len="suffix-noise", normalise=False)

        # 标准化处理（如果需要）
        if config['Norm']:
            mean, std = mean_std(X_train)
            mean = np.repeat(mean, max_seq_len).reshape(X_train.shape[1], max_seq_len)
            std = np.repeat(std, max_seq_len).reshape(X_train.shape[1], max_seq_len)
            X_train = mean_std_transform(X_train, mean, std)
            X_test = mean_std_transform(X_test, mean, std)

        Data['max_len'] = max_seq_len
        Data['All_train_data'] = X_train
        Data['All_train_label'] = y_train

        if config['val_ratio'] > 0:
            train_data, train_label, val_data, val_label = split_dataset(X_train, y_train, config['val_ratio'])
        else:
            val_data, val_label = [None, None]

        logger.info("{} samples will be used for training".format(len(train_label)))
        logger.info("{} samples will be used for validation".format(len(val_label)))
        logger.info("{} samples will be used for testing".format(len(y_test)))

        Data['train_data'] = train_data
        Data['train_label'] = train_label
        Data['val_data'] = val_data
        Data['val_label'] = val_label
        Data['test_data'] = X_test
        Data['test_label'] = y_test

        np.save(config['data_dir'] + "/" + problem, Data, allow_pickle=True)

    return Data


def split_dataset(data, label, validation_ratio):
    """
    分割数据集为训练集和验证集
    """
    splitter = model_selection.StratifiedShuffleSplit(n_splits=1, test_size=validation_ratio, random_state=1234)
    train_indices, val_indices = zip(*splitter.split(X=np.zeros(len(label)), y=label))
    train_data = data[train_indices]
    train_label = label[train_indices]
    val_data = data[val_indices]
    val_label = label[val_indices]
    return train_data, train_label, val_data, val_label
