import numpy as np
import torch
import torch.utils.data as data

from config import BASICConfig, DIMKTConfig


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

    def __init__(self, problem_list, skill_list, ans_list, problem_difficulty_list, skill_difficulty_list,
                 min_problem_num, max_problem_num):

        self.problem_list, self.skill_list, self.ans_list, self.problem_difficulty_list, self.skill_difficulty_list = [], [], [], [], []
        self.min_problem_num = min_problem_num
        self.max_problem_num = max_problem_num

        # 少于min_problem_num丢弃, 多于max_problem_num的分成多个max_problem_num
        for (problem, skill, ans, problem_diff, skill_diff) in zip(problem_list, skill_list, ans_list,
                                                                   problem_difficulty_list, skill_difficulty_list):
            num = len(problem)
            if num < min_problem_num:
                continue
            elif num > max_problem_num:
                segment = num // max_problem_num
                now_problem = problem[num - segment * max_problem_num:]
                now_skill = skill[num - segment * max_problem_num:]
                now_ans = ans[num - segment * max_problem_num:]
                now_problem_diff = problem_diff[num - segment * max_problem_num:]
                now_skill_diff = skill_diff[num - segment * max_problem_num:]

                # 不能整除 除了分割成segment个max_problem_num长度的列表外还有剩余
                if num > segment * max_problem_num:
                    self.problem_list.append(problem[:num - segment * max_problem_num])
                    self.skill_list.append(skill[:num - segment * max_problem_num])
                    self.ans_list.append(ans[:num - segment * max_problem_num])
                    self.problem_difficulty_list.append(problem_diff[:num - segment * max_problem_num])
                    self.skill_difficulty_list.append(skill_diff[:num - segment * max_problem_num])

                for i in range(segment):
                    item_problem = now_problem[i * max_problem_num:(i + 1) * max_problem_num]
                    item_skill = now_skill[i * max_problem_num:(i + 1) * max_problem_num]
                    item_ans = now_ans[i * max_problem_num:(i + 1) * max_problem_num]
                    item_problem_diff = now_problem_diff[i * max_problem_num:(i + 1) * max_problem_num]
                    item_skill_diff = now_skill_diff[i * max_problem_num:(i + 1) * max_problem_num]

                    self.problem_list.append(item_problem)
                    self.skill_list.append(item_skill)
                    self.ans_list.append(item_ans)
                    self.problem_difficulty_list.append(item_problem_diff)
                    self.skill_difficulty_list.append(item_skill_diff)

            else:
                item_problem = problem
                item_skill = skill
                item_ans = ans
                item_problem_diff = problem_diff
                item_skill_diff = skill_diff
                self.problem_list.append(item_problem)
                self.skill_list.append(item_skill)
                self.ans_list.append(item_ans)
                self.problem_difficulty_list.append(item_problem_diff)
                self.skill_difficulty_list.append(item_skill_diff)

    def __len__(self):
        return len(self.problem_list)

    def __getitem__(self, index):

        now_problem = self.problem_list[index]
        now_skill = self.skill_list[index]
        now_ans = self.ans_list[index]
        now_problem_diff = self.problem_difficulty_list[index]
        now_skill_diff = self.skill_difficulty_list[index]

        # 由于需要统一格式（统一长度）
        use_problem = np.zeros(self.max_problem_num, dtype=int)
        use_skill = np.zeros(self.max_problem_num, dtype=int)
        use_ans = np.zeros(self.max_problem_num, dtype=int)
        use_mask = np.zeros(self.max_problem_num, dtype=int)
        use_problem_diff = np.zeros(self.max_problem_num, dtype=int)
        use_skill_diff = np.zeros(self.max_problem_num, dtype=int)

        num = len(now_problem)
        use_problem[:num] = now_problem
        use_skill[:num] = now_skill
        use_ans[:num] = now_ans
        use_mask[1:num] = 1
        use_problem_diff[:num] = now_problem_diff
        use_skill_diff[:num] = now_skill_diff

        use_problem = torch.from_numpy(use_problem).to(BASICConfig.DEVICE).long()
        use_skill = torch.from_numpy(use_skill).to(BASICConfig.DEVICE).long()
        use_ans = torch.from_numpy(use_ans).to(BASICConfig.DEVICE).long()
        res_mask = torch.from_numpy(use_mask != 0).to(BASICConfig.DEVICE)
        use_problem_diff = torch.from_numpy(use_problem_diff).to(BASICConfig.DEVICE).long()
        use_skill_diff = torch.from_numpy(use_skill_diff).to(BASICConfig.DEVICE).long()

        return use_problem, use_skill, use_ans, res_mask, use_problem_diff, use_skill_diff


def getLoader(is_train, path, batch_size, min_problem_num, max_problem_num):
    read_data = getReader(path)
    problem_list, skill_list, ans_list = read_data.readData()

    ### 计算技能难度和问题难度信息
    if is_train:
        problem_difficulty, skill_difficulty = compute_difficulty(problem_list, skill_list, ans_list,
                                                                  diff_level=DIMKTConfig.DIFFICULT_LEVELS)
        torch.save({'problem_difficulty': problem_difficulty, 'skill_difficulty': skill_difficulty},
                   'difficulty_info.pth')
    else:
        difficulty_data = torch.load('difficulty_info.pth', weights_only=True)
        problem_difficulty = difficulty_data['problem_difficulty']
        skill_difficulty = difficulty_data['skill_difficulty']
    problem_difficulty_list = [[problem_difficulty.get(problem_id, 1) for problem_id in problem_group]
                               for problem_group in problem_list]
    skill_difficulty_list = [[skill_difficulty.get(skill_id, 1) for skill_id in skill_group]
                             for skill_group in skill_list]

    dataset = KT_Dataset(problem_list, skill_list, ans_list, problem_difficulty_list, skill_difficulty_list,
                         min_problem_num, max_problem_num)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    return loader


from collections import defaultdict


def compute_difficulty(problem_list, skill_list, ans_list, diff_level):
    problem_correct_count = defaultdict(int)
    problem_total_count = defaultdict(int)
    skill_correct_count = defaultdict(int)
    skill_total_count = defaultdict(int)

    for student_problems, student_skills, student_answers in zip(problem_list, skill_list, ans_list):
        for problem_id, skill_id, ans in zip(student_problems, student_skills, student_answers):
            if problem_id != 0:  # 跳过无效的问题
                problem_total_count[problem_id] += 1
                problem_correct_count[problem_id] += int(ans)

            if skill_id != 0:  # 跳过无效的技能
                skill_total_count[skill_id] += 1
                skill_correct_count[skill_id] += int(ans)

    problem_difficulty = {}
    for problem_id in range(0, BASICConfig.N_PRO + 1):
        if problem_id in problem_total_count:
            total = problem_total_count[problem_id]
            correct = problem_correct_count[problem_id]
            if total < 30 or correct == 0:
                problem_difficulty[problem_id] = 1
            else:
                problem_difficulty[problem_id] = int((correct / total) * diff_level) + 1
        else:
            problem_difficulty[problem_id] = 1

    # 计算技能的难度
    skill_difficulty = {}
    for skill_id in range(0, BASICConfig.N_KNOWS + 1):
        if skill_id in skill_total_count:
            total = skill_total_count[skill_id]
            correct = skill_correct_count[skill_id]
            if total < 30 or correct == 0:
                skill_difficulty[skill_id] = 1
            else:
                skill_difficulty[skill_id] = int((correct / total) * diff_level) + 1
        else:
            skill_difficulty[skill_id] = 1

    return problem_difficulty, skill_difficulty
