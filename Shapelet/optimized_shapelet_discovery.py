import numpy as np
import Shapelet.auto_pisd as auto_pisd
import Shapelet.pst_support_method as pstsm
import Shapelet.shapelet_support_method as ssm
import time
import multiprocessing
from functools import partial
import pickle
import gc
import os
import tempfile
from tqdm import tqdm


# 定义顶级辅助函数（在类外部，使其可以被pickle）
def extract_pip_task(args):
    """用于多进程提取PIP的顶级函数"""
    d, time_series, num_pip, j = args
    return d, j, auto_pisd.auto_piss_extractor(d, time_series=time_series, num_pip=num_pip, j=j)


def extract_ci_task(args):
    """用于多进程计算复杂度不变特征的顶级函数"""
    i, ts, piss = args
    return i, auto_pisd.auto_ci_extractor(ts, piss)


def process_find_ppi_batch(args):
    """用于多进程处理find_ppi的顶级函数"""
    sd_instance, batch_indices, l, d = args
    results = []
    for i in batch_indices:
        result = sd_instance.find_ppi(i, l, d)
        results.append(result)
    return results


class ShapeletDiscover():
    def __init__(self, window_size=50, num_pip=0.1, processes=32, len_of_ts=None, dim=1,
                 use_memory_mapping=False, max_batch_size=1000):
        self.window_size = window_size
        self.num_pip = num_pip
        self.list_group_ppi = []
        self.len_of_ts = len_of_ts
        self.list_labels = None
        self.dim = dim
        self.processes = processes
        self.use_memory_mapping = use_memory_mapping
        self.max_batch_size = max_batch_size
        self.temp_files = []  # 跟踪临时文件

        # 初始化空属性，避免属性缺失错误
        self.train_data = None
        self.train_labels = None
        self.train_data_piss = []
        self.train_data_ci = []
        self.train_data_ci_piss = []
        self.group_train_data = []
        self.group_train_data_pos = []
        self.group_train_data_piss = []
        self.group_train_data_ci_piss = []
        self.list_start_pos = None
        self.list_end_pos = None

    def __del__(self):
        # 清理所有临时文件
        for file_path in self.temp_files:
            if os.path.exists(file_path):
                os.unlink(file_path)

    # 保存shapelet候选者
    def save_shapelet_candidates(self, path="store/s1.pkl"):
        file = open(path, 'wb')
        pickle.dump(self.list_group_ppi, file)
        file.close()

    # 从磁盘加载shapelet信息
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

    def create_memmap_matrix(self, shape, dtype='float32'):
        """创建一个内存映射矩阵用于大型计算"""
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_filename = temp_file.name
        temp_file.close()
        self.temp_files.append(temp_filename)  # 跟踪文件以便清理
        return np.memmap(temp_filename, dtype=dtype, mode='w+', shape=shape)

    def find_ppi(self, i, l, d):
        """优化版本的find_ppi方法，避免存储所有距离矩阵"""
        print(f"Discovery {i} - {l} - {d}")
        list_result = []
        ts_pos = self.group_train_data_pos[l][i]
        t1 = self.group_train_data[l][i][d]

        # 处理每个PIS
        for j in range(len(self.group_train_data_piss[l][i][d])):
            ts_pis = self.group_train_data_piss[l][i][d][j]
            ts_ci_pis = self.group_train_data_ci_piss[l][i][d][j]
            list_dist = []

            # 分批处理距离计算
            batch_size = min(self.max_batch_size, len(self.train_data))
            for batch_start in range(0, len(self.train_data), batch_size):
                batch_end = min(batch_start + batch_size, len(self.train_data))

                for p in range(batch_start, batch_end):
                    if p == ts_pos:
                        list_dist.append(0)
                    else:
                        t2 = self.train_data[p][d]

                        # 计算距离矩阵
                        if self.use_memory_mapping and len(t1) > 1000:
                            matrix_shape = (len(t1), 2 * self.window_size + 1)
                            matrix_1 = self.create_memmap_matrix(matrix_shape)
                            _, _ = calculate_matrix_to_memmap(t1, t2, self.window_size, matrix_1, None)
                        else:
                            matrix_1, _ = auto_pisd.calculate_matrix(t1, t2, self.window_size)

                        # 计算最小距离
                        ts_pcs = auto_pisd.pcs_extractor(ts_pis, self.window_size, self.len_of_ts)
                        ts_2_ci = self.train_data_ci[p][d]
                        pcs_ci_list = ts_2_ci[ts_pcs[0]:ts_pcs[1] - 1]
                        dist = auto_pisd.find_min_dist(ts_pis, ts_pcs, matrix_1, self.list_start_pos,
                                                       self.list_end_pos, ts_ci_pis, pcs_ci_list)
                        list_dist.append(dist)

                        # 清理矩阵以释放内存
                        del matrix_1

                # 批处理后清理内存
                gc.collect()

            # 计算信息增益
            ig = ssm.find_best_split_point_and_info_gain(list_dist, self.train_labels, self.list_labels[l])
            ppi = [ts_pos, ts_pis[0], ts_pis[1], ig, self.list_labels[l], d]
            list_result.append(ppi)

        return list_result

    def extract_candidate(self, train_data):
        """提取shapelet候选 - 优化版本，确保多进程高效运行"""
        time1 = time.time()
        self.train_data_piss = [[] for i in range(len(train_data))]

        # 创建一个单一的进程池用于所有计算
        max_workers = min(self.processes, os.cpu_count(), 16)
        print(f"Creating process pool with {max_workers} workers")

        # 准备批量任务
        total_tasks = []
        for i in range(len(train_data)):
            for d in range(self.dim):
                total_tasks.append((d, train_data[i], self.num_pip, i))

        # 使用进程池批量处理所有任务
        results = []
        with multiprocessing.Pool(processes=max_workers) as pool:
            print(f"Starting parallel extraction with {len(total_tasks)} tasks...")
            # 使用顶级函数extract_pip_task和imap_unordered提高效率
            for result in tqdm(pool.imap_unordered(extract_pip_task, total_tasks),
                               total=len(total_tasks),
                               desc="Extracting PIPs"):
                results.append(result)

        # 处理结果，填充self.train_data_piss
        for d, j, pips in results:
            # 确保train_data_piss[j]有足够空间
            while len(self.train_data_piss[j]) <= d:
                self.train_data_piss[j].append(None)
            self.train_data_piss[j][d] = pips

        # 计算复杂度不变特征
        print("Calculating complexity invariant features...")

        # 准备CI提取任务
        ci_tasks = [(i, train_data[i], self.train_data_piss[i]) for i in range(len(train_data))]

        # 并行计算CI特征
        ci_results = {}
        with multiprocessing.Pool(processes=max_workers) as pool:
            for result in tqdm(pool.imap_unordered(extract_ci_task, ci_tasks),
                               total=len(ci_tasks),
                               desc="Computing CI features"):
                i, ci_data = result
                ci_results[i] = ci_data

        # 整理CI结果
        self.train_data_ci = [None] * len(train_data)
        self.train_data_ci_piss = [None] * len(train_data)
        for i in range(len(train_data)):
            self.train_data_ci[i] = ci_results[i][0]
            self.train_data_ci_piss[i] = ci_results[i][1]

        time1 = time.time() - time1
        print(f"Extraction completed in {time1:.2f} seconds")
        gc.collect()

    def find_ppi_batch(self, batch_indices, l, d):
        """处理一批时间序列的find_ppi"""
        results = []
        for i in batch_indices:
            result = self.find_ppi(i, l, d)
            results.append(result)
        return results

    def discovery(self, train_data, train_labels, flag=1):
        """发现shapelets的主要方法，使用分批处理来减少内存使用"""
        time2 = time.time()
        self.train_data = train_data
        self.train_labels = train_labels

        # 确保必需的属性存在
        if not hasattr(self, 'train_data_ci_piss') or not self.train_data_ci_piss:
            print("警告: train_data_ci_piss 属性不存在或为空。请先运行 extract_candidate 方法。")
            # 如果需要，可以在这里添加应急代码或提前返回
            if not self.train_data_ci_piss:
                print("正在尝试重新提取候选...")
                self.extract_candidate(train_data)

        self.len_of_ts = len(train_data[0][0])
        self.list_labels = np.unique(train_labels)

        # 计算起始位置和结束位置列表
        self.list_start_pos = np.ones(self.len_of_ts, dtype=int)
        self.list_end_pos = np.ones(self.len_of_ts, dtype=int) * (self.window_size * 2 + 1)
        for i in range(self.window_size):
            self.list_end_pos[-(i + 1)] -= self.window_size - i
        for i in range(self.window_size - 1):
            self.list_start_pos[i] += self.window_size - i - 1

        # 按标签分组时间序列
        print("Grouping time series by label")
        self.group_train_data = [[] for _ in self.list_labels]
        self.group_train_data_pos = [[] for _ in self.list_labels]
        self.group_train_data_piss = [[] for _ in self.list_labels]
        self.group_train_data_ci_piss = [[] for _ in self.list_labels]

        for l in range(len(self.list_labels)):
            for i in range(len(train_data)):
                if train_labels[i] == self.list_labels[l]:
                    self.group_train_data[l].append(train_data[i])
                    self.group_train_data_pos[l].append(i)
                    self.group_train_data_piss[l].append(self.train_data_piss[i])
                    self.group_train_data_ci_piss[l].append(self.train_data_ci_piss[i])

        # 为每组标签选择shapelet
        self.list_group_ppi = [[] for _ in range(len(self.list_labels))]
        print(f"CPU Count: {multiprocessing.cpu_count()}")

        if flag == 1:
            for l in range(len(self.list_labels)):
                for d in range(self.dim):
                    print(f"Processing label {l} - dimension {d}")

                    # 分批处理
                    total_items = len(self.group_train_data[l])
                    max_workers = min(self.processes, os.cpu_count(), 16)
                    batch_size = max(1, min(self.max_batch_size, total_items // max_workers))

                    list_ppi = []
                    for batch_start in range(0, total_items, batch_size):
                        batch_end = min(batch_start + batch_size, total_items)
                        batch_indices = list(range(batch_start, batch_end))

                        # 处理一批时间序列
                        temp_ppi = []
                        if len(batch_indices) > 1 and max_workers > 1:
                            # 直接在主循环中处理，避免使用进程池
                            for i in batch_indices:
                                batch_result = self.find_ppi(i, l, d)
                                temp_ppi.extend(batch_result)
                        else:
                            # 小批次直接处理
                            for i in batch_indices:
                                result = self.find_ppi(i, l, d)
                                temp_ppi.extend(result)

                        # 添加这批次的结果
                        list_ppi.extend(temp_ppi)

                        # 清理内存
                        gc.collect()

                    list_ppi = np.asarray(list_ppi)
                    self.list_group_ppi[l].append(list_ppi)

                    # 批次处理完成后，再次清理内存
                    gc.collect()
        else:
            # 处理标志=2的情况
            for l in range(len(self.list_labels)):
                for i in range(len(self.group_train_data[l])):
                    temp_ppi = [self.find_ppi(i, l, d) for d in range(self.dim)]
                    self.list_group_ppi[l].append(temp_ppi)
                    gc.collect()

        time2 = time.time() - time2
        print(f"Window size: {self.window_size} - Evaluation time: {time2}")
        return self.list_group_ppi


# 定义calculate_matrix_to_memmap函数（原先在auto_pisd.py中）
def calculate_matrix_to_memmap(ts_1, ts_2, w, matrix_1_out, matrix_2_out=None):
    """
    计算两个时间序列之间的距离矩阵，并将结果写入提供的内存映射数组
    """
    # 填充matrix_1_out
    matrix_1_out.fill(np.inf)

    # 计算中心对角线
    list_dist = (ts_1 - ts_2) ** 2
    matrix_1_out[:, w] = list_dist
    if matrix_2_out is not None:
        matrix_2_out[:, w] = list_dist

    # 计算其他对角线
    for i in range(w):
        list_dist = (ts_1[(i + 1):] - ts_2[:-(i + 1)]) ** 2
        matrix_1_out[(i + 1):, w - (i + 1)] = list_dist
        if matrix_2_out is not None:
            matrix_2_out[:-(i + 1), w + (i + 1)] = list_dist

        list_dist = (ts_2[(i + 1):] - ts_1[:-(i + 1)]) ** 2
        if matrix_2_out is not None:
            matrix_2_out[(i + 1):, w - (i + 1)] = list_dist
        matrix_1_out[:-(i + 1), w + (i + 1)] = list_dist

    return matrix_1_out, matrix_2_out if matrix_2_out is not None else None
