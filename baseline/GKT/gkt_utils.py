import torch
import numpy as np


def get_gkt_graph(num_c, dataset, fold):
    if dataset in ["assist09", "ednet"]:
        data1 = process_file(f'../../pre_process_data/{dataset}/{fold}/train_test/train_question.txt', 7, 2)
        data2 = process_file(f'../../pre_process_data/{dataset}/{fold}/train_test/test_question.txt', 7, 2)
    elif dataset in ["assist12", "junyi"]:
        data1 = process_file(f'../../pre_process_data/{dataset}/{fold}/train_test/train_question.txt', 6, 1)
        data2 = process_file(f'../../pre_process_data/{dataset}/{fold}/train_test/test_question.txt', 6, 1)
    elif dataset in ["assist17"]:
        data1 = process_file(f'../../pre_process_data/{dataset}/{fold}/train_test/train_question.txt', 7, 1)
        data2 = process_file(f'../../pre_process_data/{dataset}/{fold}/train_test/test_question.txt', 7, 1)
    elif dataset in ["xes3g5m", "eedi"]:
        data1 = process_file(f'../../pre_process_data/{dataset}/{fold}/train_test/train_question.txt', 6, 2)
        data2 = process_file(f'../../pre_process_data/{dataset}/{fold}/train_test/test_question.txt', 6, 2)
    all_data = data1 + data2
    graph = build_transition_graph(all_data, num_c)
    np.savez(f'./{dataset}/graph{fold}.npz', matrix=graph)
    return graph


def build_transition_graph(data, concept_num):
    """generate transition graph

    Args:
        df (da): _description_
        concept_num (int): number of concepts

    Returns:
        numpy: graph
    """
    graph = np.zeros((concept_num, concept_num))

    for sequence in data:
        pre = sequence[:-1]
        next = sequence[1:]
        for p, n in zip(pre, next):
            graph[p, n] += 1

    np.fill_diagonal(graph, 0)

    rowsum = np.array(graph.sum(1))

    def inv(x):
        return 1. / x if x != 0 else 0

    inv_func = np.vectorize(inv)
    r_inv = inv_func(rowsum).flatten()
    r_mat_inv = np.diag(r_inv)
    graph = r_mat_inv.dot(graph)
    graph = torch.from_numpy(graph).float()

    return graph


def process_file(file_path, lines_per_group=7, target_line_index=2):
    result = []  # 存储当前文件的二元列表

    with open(file_path, 'r') as file:
        lines = file.readlines()  # 读取所有行

        # 按行分组并提取目标行
        for i in range(target_line_index, len(lines), lines_per_group):
            line = lines[i].strip()  # 去除换行符
            values = line.split(',')  # 按逗号分隔
            values = [int(val) for val in values]  # 转换为整数
            result.append(values)  # 添加到结果列表

    return result
