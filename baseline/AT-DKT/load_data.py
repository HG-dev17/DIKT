import torch.utils.data as data
import torch
import numpy as np

from config import BASICConfig


class getReader():
    def __init__(self, path):
        self.path = path

    def readData(self):

        problem_list = []
        skill_list = []
        ans_list = []
        split_char = ','

        with open(self.path, 'r') as read:
            for index, line in enumerate(read):
                if BASICConfig.DATASET in ["assist09","ednet"]:
                    if index % 7 == 0:
                        pass
                    elif index % 7 == 1:
                        pass
                    elif index % 7 == 2:
                        skills = list(line.strip().split(split_char))
                        skill_list.append(skills)
                    elif index % 7 == 3:
                        problems = list(line.strip().split(split_char))
                        problem_list.append(problems)
                    elif index % 7 == 4:
                        ans = list(line.strip().split(split_char))
                        ans_list.append(ans)
                    elif index % 7 == 5:
                        pass
                    elif index % 7 == 6:
                        pass

                elif BASICConfig.DATASET in ["assist12", "junyi"]:
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

                elif BASICConfig.DATASET == "assist17":
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

                elif BASICConfig.DATASET in ["xes3g5m", "eedi"]:
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

    def __init__(self, problem_list, skill_list, ans_list, historyratios_list, min_problem_num, max_problem_num):

        self.problem_list, self.skill_list, self.ans_list, self.historyratios_list = [], [], [], []
        self.min_problem_num = min_problem_num
        self.max_problem_num = max_problem_num

        # 少于min_problem_num丢弃, 多于max_problem_num的分成多个max_problem_num
        for (problem, skill, ans, historyratios) in zip(problem_list, skill_list, ans_list, historyratios_list):
            num = len(problem)
            if num < min_problem_num:
                continue
            elif num > max_problem_num:
                segment = num // max_problem_num
                now_problem = problem[num - segment * max_problem_num:]
                now_skill = skill[num - segment * max_problem_num:]
                now_ans = ans[num - segment * max_problem_num:]
                now_historyratios = historyratios[num - segment * max_problem_num:]

                # 不能整除 除了分割成segment个max_problem_num长度的列表外还有剩余
                if num > segment * max_problem_num:
                    if num - segment * max_problem_num >= min_problem_num:
                        self.problem_list.append(problem[:num - segment * max_problem_num])
                        self.skill_list.append(skill[:num - segment * max_problem_num])
                        self.ans_list.append(ans[:num - segment * max_problem_num])
                        self.historyratios_list.append(historyratios[:num - segment * max_problem_num])

                for i in range(segment):
                    item_problem = now_problem[i * max_problem_num:(i + 1) * max_problem_num]
                    item_skill = now_skill[i * max_problem_num:(i + 1) * max_problem_num]
                    item_ans = now_ans[i * max_problem_num:(i + 1) * max_problem_num]
                    item_historyratios = now_historyratios[i * max_problem_num:(i + 1) * max_problem_num]

                    self.problem_list.append(item_problem)
                    self.skill_list.append(item_skill)
                    self.ans_list.append(item_ans)
                    self.historyratios_list.append(item_historyratios)

            else:
                item_problem = problem
                item_skill = skill
                item_ans = ans
                item_historyratios = historyratios
                self.problem_list.append(item_problem)
                self.skill_list.append(item_skill)
                self.ans_list.append(item_ans)
                self.historyratios_list.append(item_historyratios)

    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):

        now_problem = self.problem_list[index]
        now_skill = self.skill_list[index]
        now_ans = self.ans_list[index]
        now_historyratios = self.historyratios_list[index]

        # 由于需要统一格式（统一长度）
        use_problem = np.zeros(self.max_problem_num, dtype=int)
        use_skill = np.zeros(self.max_problem_num, dtype=int)
        use_ans = np.zeros(self.max_problem_num, dtype=int)
        use_mask = np.zeros(self.max_problem_num, dtype=int)
        use_historyratios = np.zeros(self.max_problem_num, dtype=int)

        num = len(now_problem)
        use_problem[:num] = now_problem
        use_skill[:num] = now_skill
        use_ans[:num] = now_ans
        use_mask[:num] = 1
        use_historyratios[:num] = now_historyratios

        use_problem = torch.from_numpy(use_problem).to(BASICConfig.DEVICE).long()
        use_skill = torch.from_numpy(use_skill).to(BASICConfig.DEVICE).long()
        use_ans = torch.from_numpy(use_ans).to(BASICConfig.DEVICE).float()
        res_mask = torch.from_numpy(use_mask != 0).to(BASICConfig.DEVICE)
        use_historyratios = torch.from_numpy(use_historyratios).to(BASICConfig.DEVICE).float()

        return use_problem, use_skill, use_ans, res_mask, use_historyratios


def getLoader(is_train, path, batch_size, min_problem_num, max_problem_num):
    read_data = getReader(path)
    problem_list, skill_list, ans_list = read_data.readData()

    historyratios = []
    for problems, answers in zip(problem_list, ans_list):
        right = 0
        student_historyratios = []
        for j, ans in enumerate(answers):
            right += int(ans == 1)  # 如果答案为1，right增加1
            student_historyratios.append(right / (j + 1))  # j+1表示当前题目总数
        historyratios.append(student_historyratios)

    dataset = KT_Dataset(problem_list, skill_list, ans_list, historyratios, min_problem_num,
                         max_problem_num)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    return loader
