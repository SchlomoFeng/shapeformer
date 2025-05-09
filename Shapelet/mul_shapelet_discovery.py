import numpy as np
import Shapelet.auto_pisd as auto_pisd
import Shapelet.pst_support_method as pstsm
import Shapelet.shapelet_support_method as ssm
import time
import multiprocessing
from functools import partial
import pickle
import gc


class ShapeletDiscover():
    def __init__(self, window_size=50, num_pip=0.1, processes=4, len_of_ts=None, dim=2, batch_size_cpu=5):
        self.window_size = window_size
        self.num_pip = num_pip
        self.list_group_ppi = []
        self.len_of_ts = len_of_ts
        self.list_labels = None
        self.dim = dim
        self.processes = processes
        self.batch_size_cpu = batch_size_cpu

    # save list_group_ppi with pickle
    def save_shapelet_candidates(self, path="store/s1.pkl"):
        file = open(path, 'wb')
        pickle.dump(self.list_group_ppi, file)
        file.close()

    # load shapelet information from disk
    def load_shapelet_candidates(self, path="store/s1.pkl"):
        file = open(path, 'rb')
        ppi = pickle.load(file)
        if ppi is not None:
            self.list_group_ppi = ppi
        file.close()

    def set_window_size(self, window_size):
        self.window_size = window_size

    def get_shapelet_info(self, number_of_shapelet, p=0.0, pi=0.0):
        if number_of_shapelet == 0:
            number_of_shapelet = 1

        list_shapelet = None
        for i in range(len(self.list_group_ppi)):
            list_ppi = np.concatenate(self.list_group_ppi[i])
            list_group_shapelet = pstsm.find_c_shapelet_non_overlab(list_ppi, number_of_shapelet, p=p, p_inner=pi,
                                                                    len_ts=self.len_of_ts)
            list_group_shapelet = np.asarray(list_group_shapelet)
            list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
            if list_shapelet is None:
                list_shapelet = list_group_shapelet
            else:
                list_shapelet = np.concatenate((list_shapelet, list_group_shapelet), axis=0)

        return list_shapelet

    def get_shapelet_info_v1(self, number_of_shapelet):
        if number_of_shapelet == 0:
            number_of_shapelet = 1

        list_shapelet = None
        for i in range(len(self.list_group_ppi)):
            for d in range(self.dim):
                list_ppi = self.list_group_ppi[i][d]
                list_group_shapelet = pstsm.find_c_shapelet_non_overlab(list_ppi, number_of_shapelet)
                list_group_shapelet = np.asarray(list_group_shapelet)
                list_group_shapelet = list_group_shapelet[list_group_shapelet[:, 1].argsort()]
                if list_shapelet is None:
                    list_shapelet = list_group_shapelet
                else:
                    list_shapelet = np.concatenate((list_shapelet, list_group_shapelet), axis=0)

        return list_shapelet

    def find_ppi(self, batch_info):
        """批处理并行优化版的find_ppi函数

        参数:
            batch_info: 包含(i_start, i_end, l, d)的元组，表示要处理的批次范围、类别和维度
        """
        i_start, i_end, l, d = batch_info
        batch_results = []

        for i in range(i_start, i_end):
            print(f"discovery {i} - {l} - {d}")
            ts_pos = self.group_train_data_pos[l][i]
            t1 = self.group_train_data[l][i][d]
            pdm = {}

            # 只计算必要的矩阵
            for p in range(len(self.train_data)):
                t2 = self.train_data[p][d]
                matrix_1, matrix_2 = auto_pisd.calculate_matrix(t1, t2, self.window_size)
                pdm[p] = matrix_1

            # 处理每个PIS
            for j in range(len(self.group_train_data_piss[l][i][d])):
                list_dist = []  # 在这里初始化list_dist
                ts_pis = self.group_train_data_piss[l][i][d][j]
                ts_ci_pis = self.group_train_data_ci_piss[l][i][d][j]

                # 计算到所有时间序列的距离
                for p in range(len(self.train_data)):
                    if p == ts_pos:
                        list_dist.append(0)
                    else:
                        matrix = pdm[p]
                        ts_pcs = auto_pisd.pcs_extractor(ts_pis, self.window_size, self.len_of_ts)
                        ts_2_ci = self.train_data_ci[p][d]
                        pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                        dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix, self.list_start_pos,
                                                       self.list_end_pos, ts_ci_pis, pcs_ci_list)
                        list_dist.append(dist)

                # 计算信息增益
                ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, self.list_labels[l])
                ppi = [ts_pos, ts_pis[0], ts_pis[1], ig, self.list_labels[l], d]
                batch_results.append(ppi)

            # 释放矩阵内存
            del pdm

        return batch_results

    def _prepare_position_lists(self):
        """预计算start和end位置列表"""
        self.list_start_pos = np.ones(self.len_of_ts, dtype=int)
        self.list_end_pos = np.ones(self.len_of_ts, dtype=int) * (self.window_size * 2 + 1)

        for i in range(self.window_size):
            self.list_end_pos[-(i + 1)] -= self.window_size - i
        for i in range(self.window_size - 1):
            self.list_start_pos[i] += self.window_size - i - 1

    def _group_data_by_labels(self):
        """按标签分组数据"""
        self.group_train_data = [[] for i in self.list_labels]
        self.group_train_data_pos = [[] for i in self.list_labels]
        self.group_train_data_piss = [[] for i in self.list_labels]
        self.group_train_data_ci_piss = [[] for i in self.list_labels]

        for l in range(len(self.list_labels)):
            for i in range(len(self.train_data)):
                if self.train_labels[i] == self.list_labels[l]:
                    self.group_train_data[l].append(self.train_data[i])
                    self.group_train_data_pos[l].append(i)
                    self.group_train_data_piss[l].append(self.train_data_piss[i])
                    self.group_train_data_ci_piss[l].append(self.train_data_ci_piss[i])

    def extract_candidate(self, train_data):
        # Extract shapelet candidate
        time1 = time.time()
        self.train_data_piss = [[] for i in range(len(train_data))]
        p = multiprocessing.Pool(processes=self.processes)
        for i in range(len(train_data)):
            time_series = train_data[i]
            temp_ppi = p.map(partial(auto_pisd.auto_piss_extractor, time_series=time_series, num_pip=self.num_pip, j=i),
                             range(self.dim))
            self.train_data_piss[i] = temp_ppi

        ci_return = [auto_pisd.auto_ci_extractor(train_data[i], self.train_data_piss[i]) for i in
                     range(len(train_data))]
        self.train_data_ci = [ci_return[i][0] for i in range(len(ci_return))]
        self.train_data_ci_piss = [ci_return[i][1] for i in range(len(ci_return))]

        time1 = time.time() - time1
        print("extracting time: %s" % time1)
        p.close()
        p.join()

    def discovery(self, train_data, train_labels, flag=1, batch_size_cpu=50):
        """内存优化版的discovery方法，带并行批处理"""
        time2 = time.time()

        self.train_data = train_data
        self.train_labels = train_labels
        self.len_of_ts = len(train_data[0][0])
        self.list_labels = np.unique(train_labels)

        # 预计算位置列表
        self._prepare_position_lists()

        # 按标签分组训练数据
        self._group_data_by_labels()

        # 初始化结果列表
        self.list_group_ppi = [[] for i in range(len(self.list_labels))]

        # 处理每个标签组
        for l in range(len(self.list_labels)):
            print(f"Processing label {self.list_labels[l]} ({l + 1}/{len(self.list_labels)})")

            # 处理每个维度
            for d in range(self.dim):
                print(f"  Dimension {d + 1}/{self.dim}")
                list_ppi = []

                # 创建批次任务列表
                batch_tasks = []
                for i_start in range(0, len(self.group_train_data[l]), batch_size_cpu):
                    i_end = min(i_start + batch_size_cpu, len(self.group_train_data[l]))
                    batch_tasks.append((i_start, i_end, l, d))

                # 创建进程池并行处理批次
                with multiprocessing.Pool(processes=self.processes) as pool:
                    batch_results = pool.map(self.find_ppi, batch_tasks)

                    # 收集所有批次的结果
                    for batch_result in batch_results:
                        for ppi in batch_result:
                            list_ppi.append(ppi)

                # 转换为numpy数组并保存
                if list_ppi:
                    list_ppi = np.asarray(list_ppi)
                    self.list_group_ppi[l].append(list_ppi)

                # 强制进行垃圾回收
                gc.collect()

        time2 = time.time() - time2
        print(f"window_size: {self.window_size} - evaluating_time: {time2}")
