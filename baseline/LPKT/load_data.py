import numpy as np
import torch
import torch.utils.data as data
import glo

from config import BASICConfig


class getReader():
    def __init__(self, path):
        self.path = path

    def readData(self):

        problem_list = []
        skill_list = []
        ans_list = []
        at_list = []
        start_time_list = []
        split_char = ','

        with open(self.path, 'r') as read:
            for index, line in enumerate(read):
                if BASICConfig.DATASET in ["assist12_1", "assist17"]:
                    if index % 7 == 0:
                        pass
                    elif index % 7 == 1:
                        skills = line.strip().split(split_char)
                        skills = list(map(int, skills))
                        skill_list.append(skills)
                    elif index % 7 == 2:
                        problems = line.strip().split(split_char)
                        problems = list(map(int, problems))
                        problem_list.append(problems)
                    elif index % 7 == 3:
                        ans = line.strip().split(split_char)
                        ans = list(map(int, ans))
                        ans_list.append(ans)
                    elif index % 7 == 4:
                        at = line.strip().split(split_char)
                        at_list.append(at)
                    elif index % 7 == 5:
                        start_time = line.strip().split(split_char)
                        start_time_list.append(start_time)
                    elif index % 7 == 6:
                        pass
                elif BASICConfig.DATASET in ["ednet"]:
                    if index % 7 == 0:
                        pass
                    elif index % 7 == 1:
                        pass
                    elif index % 7 == 2:
                        skills = line.strip().split(split_char)
                        skills = list(map(int, skills))
                        skill_list.append(skills)
                    elif index % 7 == 3:
                        problems = line.strip().split(split_char)
                        problems = list(map(int, problems))
                        problem_list.append(problems)
                    elif index % 7 == 4:
                        ans = line.strip().split(split_char)
                        ans = list(map(int, ans))
                        ans_list.append(ans)
                    elif index % 7 == 5:
                        at = line.strip().split(split_char)
                        at_list.append(at)
                    elif index % 7 == 6:
                        start_time = line.strip().split(split_char)
                        start_time_list.append(start_time)
                elif BASICConfig.DATASET in ["junyi"]:
                    if index % 6 == 0:
                        pass
                    elif index % 6 == 1:
                        skills = line.strip().split(split_char)
                        skills = list(map(int, skills))
                        skill_list.append(skills)
                    elif index % 6 == 2:
                        problems = line.strip().split(split_char)
                        problems = list(map(int, problems))
                        problem_list.append(problems)
                    elif index % 6 == 3:
                        ans = line.strip().split(split_char)
                        ans = list(map(int, ans))
                        ans_list.append(ans)
                    elif index % 6 == 4:
                        at = line.strip().split(split_char)
                        at_list.append(at)
                    elif index % 6 == 5:
                        start_time = line.strip().split(split_char)
                        start_time_list.append(start_time)

        return problem_list, skill_list, ans_list, at_list, start_time_list


class KT_Dataset(data.Dataset):

    def __init__(self, problem_list, skill_list, ans_list, mapped_at, mapped_it, min_problem_num, max_problem_num):

        self.problem_list, self.skill_list, self.ans_list, self.at_list, self.it_list = [], [], [], [], []
        self.min_problem_num = min_problem_num
        self.max_problem_num = max_problem_num

        # 少于min_problem_num丢弃, 多于max_problem_num的分成多个max_problem_num
        for (problem, skill, ans, at, it) in zip(problem_list, skill_list, ans_list, mapped_at, mapped_it):
            num = len(problem)
            if num < min_problem_num:
                continue
            elif num > max_problem_num:
                segment = num // max_problem_num
                now_problem = problem[num - segment * max_problem_num:]
                now_skill = skill[num - segment * max_problem_num:]
                now_ans = ans[num - segment * max_problem_num:]
                now_at = at[num - segment * max_problem_num:]
                now_it = it[num - segment * max_problem_num:]

                # 不能整除 除了分割成segment个max_problem_num长度的列表外还有剩余
                if num > segment * max_problem_num:
                    if num - segment * max_problem_num >= min_problem_num:
                        self.problem_list.append(problem[:num - segment * max_problem_num])
                        self.skill_list.append(skill[:num - segment * max_problem_num])
                        self.ans_list.append(ans[:num - segment * max_problem_num])
                        self.at_list.append(at[:num - segment * max_problem_num])
                        self.it_list.append(it[:num - segment * max_problem_num])

                for i in range(segment):
                    item_problem = now_problem[i * max_problem_num:(i + 1) * max_problem_num]
                    item_skill = now_skill[i * max_problem_num:(i + 1) * max_problem_num]
                    item_ans = now_ans[i * max_problem_num:(i + 1) * max_problem_num]
                    item_at = now_at[i * max_problem_num:(i + 1) * max_problem_num]
                    item_it = now_it[i * max_problem_num:(i + 1) * max_problem_num]

                    self.problem_list.append(item_problem)
                    self.skill_list.append(item_skill)
                    self.ans_list.append(item_ans)
                    self.at_list.append(item_at)
                    self.it_list.append(item_it)

            else:
                item_problem = problem
                item_skill = skill
                item_ans = ans
                item_at = at
                item_it = it
                self.problem_list.append(item_problem)
                self.skill_list.append(item_skill)
                self.ans_list.append(item_ans)
                self.at_list.append(item_at)
                self.it_list.append(item_it)

    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):

        now_problem = self.problem_list[index]
        now_skill = self.skill_list[index]
        now_ans = self.ans_list[index]
        now_at = self.at_list[index]
        now_it = self.it_list[index]

        # 由于需要统一格式（统一长度）
        use_problem = np.zeros(self.max_problem_num, dtype=int)
        use_skill = np.zeros(self.max_problem_num, dtype=int)
        use_ans = np.zeros(self.max_problem_num, dtype=int)
        use_mask = np.zeros(self.max_problem_num, dtype=int)
        use_at = np.zeros(self.max_problem_num, dtype=int)
        use_it = np.zeros(self.max_problem_num, dtype=int)

        num = len(now_problem)
        use_problem[:num] = now_problem
        use_skill[:num] = now_skill
        use_ans[:num] = now_ans
        use_at[:num] = now_at
        use_it[:num] = now_it
        use_mask[1:num] = 1

        use_problem = torch.from_numpy(use_problem).to(BASICConfig.DEVICE).long()
        use_skill = torch.from_numpy(use_skill).to(BASICConfig.DEVICE).long()
        use_ans = torch.from_numpy(use_ans).to(BASICConfig.DEVICE).long()
        use_at = torch.from_numpy(use_at).to(BASICConfig.DEVICE).long()
        use_it = torch.from_numpy(use_it).to(BASICConfig.DEVICE).long()
        res_mask = torch.from_numpy(use_mask != 0).to(BASICConfig.DEVICE)

        return use_problem, use_skill, use_ans, use_at, use_it, res_mask


def getLoader(is_train, path, batch_size, min_problem_num, max_problem_num):
    read_data = getReader(path)
    problem_list, skill_list, ans_list, at_list, start_time_list = read_data.readData()

    # 回答时间映射
    at_set = set(value for row in at_list for value in row)  # 展平列表
    at2id = {value: idx for idx, value in enumerate(sorted(at_set))}  # 映射成连续整数ID
    mapped_at = [[at2id[val] for val in row] for row in at_list] # 映射数组

    # 间隔时间计算
    it_list = []
    for row in start_time_list:
        differences = [0]  # 第一个值的位置为0
        differences.extend([
            min((int(row[i]) - int(row[i - 1]))  // 60, 42300)
            for i in range(1, len(row))
        ])
        it_list.append(differences)

    # 间隔时间映射
    it_set = set(value for row in it_list for value in row)  # 展平列表
    it2id = {value: idx for idx, value in enumerate(sorted(it_set))}  # 映射成连续整数ID
    mapped_it = [[it2id[val] for val in row] for row in it_list] # 映射数组
    glo._init()
    glo.set_value('n_at',len(at_set))
    glo.set_value('n_it', len(it_set))
    dataset = KT_Dataset(problem_list, skill_list, ans_list, mapped_at, mapped_it, min_problem_num,
                         max_problem_num)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    return loader
