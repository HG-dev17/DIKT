import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

class getReader():
    def __init__(self, path, dataset):
        self.path = path
        self.dataset = dataset

    def readData(self):

        problem_list = []
        skill_list = []
        ans_list = []
        split_char = ','

        with open(self.path, 'r') as read:
            for index, line in enumerate(read):
                if self.dataset in ["assist09","ednet"]:
                    if index % 7 == 0:
                        pass
                    if index % 7 == 1:
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
                        pass
                    elif index % 7 == 6:
                        pass

                elif self.dataset in ["assist12", "junyi"]:
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
                        pass
                    elif index % 6 == 5:
                        pass

                elif self.dataset == "assist17":
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
                        pass
                    elif index % 7 == 5:
                        pass
                    elif index % 7 == 6:
                        pass

                elif self.dataset in ["xes3g5m", "eedi"]:
                    if index % 6 == 0:
                        pass
                    elif index % 6 == 1:
                        pass
                    elif index % 6 == 2:
                        skills = line.strip().split(split_char)
                        skills = list(map(int, skills))
                        skill_list.append(skills)
                    elif index % 6 == 3:
                        problems = line.strip().split(split_char)
                        problems = list(map(int, problems))
                        problem_list.append(problems)
                    elif index % 6 == 4:
                        ans = line.strip().split(split_char)
                        ans = list(map(int, ans))
                        ans_list.append(ans)
                    elif index % 6 == 5:
                        pass

        return problem_list, skill_list, ans_list

class KT_Dataset(data.Dataset):

    def __init__(self, problem_max, problem_list, ans_list, skill_list, min_problem_num, max_problem_num,device):
        self.problem_max = problem_max
        self.device = device
        self.min_problem_num = min_problem_num
        self.max_problem_num = max_problem_num
        self.problem_list, self.ans_list, self.skill_list = [], [], []
        # 个人定义，少于 min_problem_num 丢弃
        # 根据论文 多于 max_problem_num  的分成多个 max_problem_num
        for (problem, ans, skill) in zip(problem_list, ans_list, skill_list):
            num = len(problem)
            if num < min_problem_num:
                continue
            elif num > max_problem_num:
                segment = num // max_problem_num
                now_problem = problem[num - segment * max_problem_num:]
                now_ans = ans[num - segment * max_problem_num:]
                now_skill = skill[num - segment * max_problem_num:]

                if num > segment * max_problem_num:
                    self.problem_list.append(problem[:num - segment * max_problem_num])
                    self.ans_list.append(ans[:num - segment * max_problem_num])
                    self.skill_list.append(skill[:num - segment * max_problem_num])

                for i in range(segment):
                    item_problem = now_problem[i * max_problem_num:(i + 1) * max_problem_num]
                    item_ans = now_ans[i * max_problem_num:(i + 1) * max_problem_num]
                    item_skill = now_skill[i * max_problem_num:(i + 1) * max_problem_num]

                    self.problem_list.append(item_problem)
                    self.ans_list.append(item_ans)
                    self.skill_list.append(item_skill)
            else:
                item_problem = problem
                item_ans = ans
                item_skill = skill
                self.problem_list.append(item_problem)
                self.ans_list.append(item_ans)
                self.skill_list.append(item_skill)

    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):

        now_problem = self.problem_list[index]
        now_problem = np.array(now_problem)

        now_ans = self.ans_list[index]

        # 由于需要统一格式
        use_problem = np.zeros(self.max_problem_num, dtype=int)
        use_ans = np.zeros(self.max_problem_num, dtype=int)
        use_mask = np.zeros(self.max_problem_num, dtype=int)

        num = len(now_problem)
        use_problem[-num:] = now_problem
        use_ans[-num:] = now_ans

        next_ans = use_ans[1:]
        next_problem = use_problem[1:]

        last_ans = use_ans[:-1]
        last_problem = use_problem[:-1]

        use_mask[-num:] = 1
        next_mask = use_mask[1:]

        last_problem = torch.from_numpy(last_problem).to(self.device).long()

        next_problem = torch.from_numpy(next_problem).to(self.device).long()
        last_ans = torch.from_numpy(last_ans).to(self.device).long()
        next_ans = torch.from_numpy(next_ans).to(self.device).float()

        res_mask = torch.from_numpy(next_mask != 0).to(self.device)

        return last_problem, last_ans, next_problem, next_ans, res_mask

def getLoader(problem_max, path, batch_size, is_train, min_problem_num, max_problem_num, dataset,device):
    read_data = getReader(path, dataset)
    problem_list, skill_list, ans_list = read_data.readData()

    dataset = KT_Dataset(problem_max, problem_list, ans_list, skill_list, min_problem_num, max_problem_num,device)

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    return loader