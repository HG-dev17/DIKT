import numpy as np
import torch
import torch.utils.data as data
from collections import defaultdict
import random
import math

from config import BASICConfig, CL4KTConfig


class getReader():
    def __init__(self, path, seq_len):
        self.path = path
        self.seq_len = seq_len

    def readData(self):

        problem_list = []
        skill_list = []
        ans_list = []
        split_char = ','

        with open(self.path, 'r') as read:
            for index, line in enumerate(read):
                if BASICConfig.DATASET in ["assist09", "ednet"]:
                    if index % 7 == 0:
                        pass
                    elif index % 7 == 1:
                        pass
                    elif index % 7 == 2:
                        skills = list(line.strip().split(split_char))
                        if len(skills) < 2:
                            continue
                        skills = skills[-self.seq_len:]
                        skills = list(map(int, skills))
                        skill_list.append(skills)
                    elif index % 7 == 3:
                        problems = list(line.strip().split(split_char))
                        if len(problems) < 2:
                            continue
                        problems = problems[-self.seq_len:]
                        problems = list(map(int, problems))
                        problem_list.append(problems)
                    elif index % 7 == 4:
                        ans = list(line.strip().split(split_char))
                        if len(ans) < 2:
                            continue
                        ans = ans[-self.seq_len:]
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
                        skills = list(line.strip().split(split_char))
                        if len(skills) < 2:
                            continue
                        skills = skills[-self.seq_len:]
                        skills = list(map(int, skills))
                        skill_list.append(skills)
                    elif index % 6 == 2:
                        problems = list(line.strip().split(split_char))
                        if len(problems) < 2:
                            continue
                        problems = problems[-self.seq_len:]
                        problems = list(map(int, problems))
                        problem_list.append(problems)
                    elif index % 6 == 3:
                        ans = list(line.strip().split(split_char))
                        if len(ans) < 2:
                            continue
                        ans = ans[-self.seq_len:]
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
                        skills = list(line.strip().split(split_char))
                        if len(skills) < 2:
                            continue
                        skills = skills[-self.seq_len:]
                        skills = list(map(int, skills))
                        skill_list.append(skills)
                    elif index % 7 == 2:
                        problems = list(line.strip().split(split_char))
                        if len(problems) < 2:
                            continue
                        problems = problems[-self.seq_len:]
                        problems = list(map(int, problems))
                        problem_list.append(problems)
                    elif index % 7 == 3:
                        ans = list(line.strip().split(split_char))
                        if len(ans) < 2:
                            continue
                        ans = ans[-self.seq_len:]
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
                        skills = list(line.strip().split(split_char))
                        if len(skills) < 2:
                            continue
                        skills = skills[-self.seq_len:]
                        skills = list(map(int, skills))
                        skill_list.append(skills)
                    elif index % 6 == 3:
                        problems = list(line.strip().split(split_char))
                        if len(problems) < 2:
                            continue
                        problems = problems[-self.seq_len:]
                        problems = list(map(int, problems))
                        problem_list.append(problems)
                    elif index % 6 == 4:
                        ans = list(line.strip().split(split_char))
                        if len(ans) < 2:
                            continue
                        ans = ans[-self.seq_len:]
                        ans = list(map(int, ans))
                        ans_list.append(ans)
                    elif index % 6 == 5:
                        pass

        return problem_list, skill_list, ans_list


class MostRecentQuestionSkillDataset(data.Dataset):
    def __init__(self, problem_list, skill_list, ans_list, seq_len):
        self.seq_len = seq_len

        skill_correct = defaultdict(int)
        skill_count = defaultdict(int)
        for s_list, r_list in zip(skill_list, ans_list):
            for s, r in zip(s_list, r_list):
                skill_correct[s] += int(r)
                skill_count[s] += 1
        skill_difficulty = {
            s: skill_correct[s] / float(skill_count[s]) for s in skill_correct
        }
        ordered_skills = [
            item[0] for item in sorted(skill_difficulty.items(), key=lambda x: x[1])
        ]
        self.easier_skills = {}
        self.harder_skills = {}
        for i, s in enumerate(ordered_skills):
            if i == 0:  # the hardest
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = s
            elif i == len(ordered_skills) - 1:  # the easiest
                self.easier_skills[s] = s
                self.harder_skills[s] = ordered_skills[i - 1]
            else:
                self.easier_skills[s] = ordered_skills[i + 1]
                self.harder_skills[s] = ordered_skills[i - 1]

        self.len = len(problem_list)

        self.padded_q = np.zeros((len(problem_list), self.seq_len), dtype=int)
        self.padded_s = np.zeros((len(skill_list), self.seq_len), dtype=int)
        self.padded_r = np.full((len(ans_list), self.seq_len), -1, dtype=int)
        self.mask = np.zeros((len(skill_list), self.seq_len), dtype=int)

        for i, elem in enumerate(zip(problem_list, skill_list, ans_list)):
            q, s, r = elem
            self.padded_q[i, -len(q):] = q
            self.padded_s[i, -len(s):] = s
            self.padded_r[i, -len(r):] = r
            self.mask[i, -(len(s) - 1):] = 1

    def __getitem__(self, index):

        return {
            "questions": self.padded_q[index],
            "skills": self.padded_s[index],
            "responses": self.padded_r[index],
            "mask": self.mask[index],
        }

    def __len__(self):
        return self.len


class SimCLRDatasetWrapper(data.Dataset):
    def __init__(
            self,
            ds: data.Dataset,
            seq_len: int,
            mask_prob: float,
            crop_prob: float,
            permute_prob: float,
            replace_prob: float,
            negative_prob: float,
            eval_mode=False,
    ):
        super().__init__()
        self.ds = ds
        self.seq_len = seq_len
        self.mask_prob = mask_prob
        self.crop_prob = crop_prob
        self.permute_prob = permute_prob
        self.replace_prob = replace_prob
        self.negative_prob = negative_prob
        self.eval_mode = eval_mode

        # 包含zero值
        self.num_questions = BASICConfig.N_PRO + 1
        self.num_skills = BASICConfig.N_KNOWS + 1
        self.q_mask_id = self.num_questions + 1
        self.s_mask_id = self.num_skills + 1
        self.easier_skills = self.ds.easier_skills
        self.harder_skills = self.ds.harder_skills

    def __len__(self):
        return len(self.ds)

    def __getitem_internal__(self, index):
        original_data = self.ds[index]
        q_seq = original_data["questions"]
        s_seq = original_data["skills"]
        r_seq = original_data["responses"]
        r_mask = original_data["mask"]

        if self.eval_mode:
            q_seq = torch.tensor(q_seq, dtype=torch.int).to(BASICConfig.DEVICE)
            s_seq = torch.tensor(s_seq, dtype=torch.int).to(BASICConfig.DEVICE)
            r_seq = torch.tensor(r_seq, dtype=torch.int).to(BASICConfig.DEVICE)
            r_mask = torch.tensor(r_mask, dtype=torch.int).to(BASICConfig.DEVICE).to(torch.bool)
            return {
                "questions": q_seq,
                "skills": s_seq,
                "responses": r_seq,
                "mask": r_mask,
            }

        else:
            q_seq_list = original_data["questions"].tolist()
            s_seq_list = original_data["skills"].tolist()
            r_seq_list = original_data["responses"].tolist()

            t1 = augment_kt_seqs(
                q_seq_list,
                s_seq_list,
                r_seq_list,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index,
            )

            t2 = augment_kt_seqs(
                q_seq_list,
                s_seq_list,
                r_seq_list,
                self.mask_prob,
                self.crop_prob,
                self.permute_prob,
                self.replace_prob,
                self.negative_prob,
                self.easier_skills,
                self.harder_skills,
                self.q_mask_id,
                self.s_mask_id,
                self.seq_len,
                seed=index + 1,
            )

            aug_q_seq_1, aug_s_seq_1, aug_r_seq_1, negative_r_seq, r_mask_1 = t1
            aug_q_seq_2, aug_s_seq_2, aug_r_seq_2, _, r_mask_2 = t2

            q_seq = torch.tensor(q_seq, dtype=torch.int).to(BASICConfig.DEVICE)
            aug_q_seq_1 = torch.tensor(aug_q_seq_1, dtype=torch.int).to(BASICConfig.DEVICE)
            aug_q_seq_2 = torch.tensor(aug_q_seq_2, dtype=torch.int).to(BASICConfig.DEVICE)

            s_seq = torch.tensor(s_seq, dtype=torch.int).to(BASICConfig.DEVICE)
            aug_s_seq_1 = torch.tensor(aug_s_seq_1, dtype=torch.int).to(BASICConfig.DEVICE)
            aug_s_seq_2 = torch.tensor(aug_s_seq_2, dtype=torch.int).to(BASICConfig.DEVICE)

            r_seq = torch.tensor(r_seq, dtype=torch.int).to(BASICConfig.DEVICE)
            aug_r_seq_1 = torch.tensor(aug_r_seq_1, dtype=torch.int).to(BASICConfig.DEVICE)
            aug_r_seq_2 = torch.tensor(aug_r_seq_2, dtype=torch.int).to(BASICConfig.DEVICE)
            negative_r_seq = torch.tensor(negative_r_seq, dtype=torch.int).to(BASICConfig.DEVICE)

            r_mask = torch.tensor(r_mask, dtype=torch.int).to(BASICConfig.DEVICE).to(torch.bool)
            r_mask_1 = torch.tensor(r_mask_1, dtype=torch.int).to(BASICConfig.DEVICE).to(torch.bool)
            r_mask_2 = torch.tensor(r_mask_2, dtype=torch.int).to(BASICConfig.DEVICE).to(torch.bool)

            ret = {
                "questions": (aug_q_seq_1, aug_q_seq_2, q_seq),
                "skills": (aug_s_seq_1, aug_s_seq_2, s_seq),
                "responses": (aug_r_seq_1, aug_r_seq_2, r_seq, negative_r_seq),
                "mask": (r_mask_1, r_mask_2, r_mask),
            }
            return ret

    def __getitem__(self, index):
        return self.__getitem_internal__(index)


def augment_kt_seqs(
        q_seq,
        s_seq,
        r_seq,
        mask_prob,
        crop_prob,
        permute_prob,
        replace_prob,
        negative_prob,
        easier_skills,
        harder_skills,
        q_mask_id,
        s_mask_id,
        seq_len,
        seed=None,
):
    # masking (random or PMI 등을 활용해서)
    # 구글 논문의 Correlated Feature Masking 등...
    rng = random.Random(seed)
    np.random.seed(seed)

    masked_q_seq = []
    masked_s_seq = []
    masked_r_seq = []
    negative_r_seq = []

    if mask_prob > 0:
        for q, s, r in zip(q_seq, s_seq, r_seq):
            prob = rng.random()
            if prob < mask_prob and s != 0:
                prob /= mask_prob
                if prob < 0.8:
                    masked_q_seq.append(q_mask_id)
                    masked_s_seq.append(s_mask_id)
                elif prob < 0.9:
                    masked_q_seq.append(
                        rng.randint(1, q_mask_id - 1)
                    )  # original BERT처럼 random한 확률로 다른 token으로 대체해줌
                    masked_s_seq.append(
                        rng.randint(1, s_mask_id - 1)
                    )  # randint(start, end) [start, end] 둘다 포함
                else:
                    masked_q_seq.append(q)
                    masked_s_seq.append(s)
            else:
                masked_q_seq.append(q)
                masked_s_seq.append(s)
            masked_r_seq.append(r)  # response는 나중에 hard negatives로 활용 (0->1, 1->0)

            # reverse responses
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  # padding
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)
    else:
        masked_q_seq = q_seq[:]
        masked_s_seq = s_seq[:]
        masked_r_seq = r_seq[:]

        for r in r_seq:
            # reverse responses
            neg_prob = rng.random()
            if neg_prob < negative_prob and r != -1:  # padding
                negative_r_seq.append(1 - r)
            else:
                negative_r_seq.append(r)

    """
    skill difficulty based replace
    """
    # print(harder_skills)
    if replace_prob > 0:
        for i, elem in enumerate(zip(masked_s_seq, masked_r_seq)):
            s, r = elem
            prob = rng.random()
            if prob < replace_prob and s != 0 and s != s_mask_id:
                if (
                        r == 0 and s in harder_skills
                ):  # if the response is wrong, then replace a skill with the harder one
                    masked_s_seq[i] = harder_skills[s]
                elif (
                        r == 1 and s in easier_skills
                ):  # if the response is correct, then replace a skill with the easier one
                    masked_s_seq[i] = easier_skills[s]

    true_seq_len = np.sum(np.asarray(q_seq) != 0)
    if permute_prob > 0:
        reorder_seq_len = math.floor(permute_prob * true_seq_len)
        start_idx = (np.asarray(q_seq) != 0).argmax()
        while True:
            start_pos = rng.randint(start_idx, seq_len - reorder_seq_len)
            if start_pos + reorder_seq_len < seq_len:
                break

        # reorder (permute)
        perm = np.random.permutation(reorder_seq_len)
        masked_q_seq = (
                masked_q_seq[:start_pos]
                + np.asarray(masked_q_seq[start_pos: start_pos + reorder_seq_len])[
                    perm
                ].tolist()
                + masked_q_seq[start_pos + reorder_seq_len:]
        )
        masked_s_seq = (
                masked_s_seq[:start_pos]
                + np.asarray(masked_s_seq[start_pos: start_pos + reorder_seq_len])[
                    perm
                ].tolist()
                + masked_s_seq[start_pos + reorder_seq_len:]
        )
        masked_r_seq = (
                masked_r_seq[:start_pos]
                + np.asarray(masked_r_seq[start_pos: start_pos + reorder_seq_len])[
                    perm
                ].tolist()
                + masked_r_seq[start_pos + reorder_seq_len:]
        )

    # To-Do: check this crop logic!
    if 0 < crop_prob < 1:
        crop_seq_len = math.floor(crop_prob * true_seq_len)
        if crop_seq_len == 0:
            crop_seq_len = 1
        start_idx = (np.asarray(q_seq) != 0).argmax()
        while True:
            start_pos = rng.randint(start_idx, seq_len - crop_seq_len)
            if start_pos + crop_seq_len < seq_len:
                break

        masked_q_seq = masked_q_seq[start_pos: start_pos + crop_seq_len]
        masked_s_seq = masked_s_seq[start_pos: start_pos + crop_seq_len]
        masked_r_seq = masked_r_seq[start_pos: start_pos + crop_seq_len]

    pad_len = seq_len - len(masked_q_seq)

    r_mask = [0] * pad_len + [1] * len(masked_s_seq)
    masked_q_seq = [0] * pad_len + masked_q_seq
    masked_s_seq = [0] * pad_len + masked_s_seq
    masked_r_seq = [-1] * pad_len + masked_r_seq

    return masked_q_seq, masked_s_seq, masked_r_seq, negative_r_seq, r_mask


def getLoader(is_train, path, batch_size, seq_len, mask_prob, crop_prob, permute_prob, replace_prob, negative_prob):
    read_data = getReader(path, seq_len)
    problem_list, skill_list, ans_list = read_data.readData()

    dataset = MostRecentQuestionSkillDataset(problem_list, skill_list, ans_list, seq_len)

    if is_train:
        loader = data.DataLoader(
            SimCLRDatasetWrapper(dataset, seq_len, mask_prob, crop_prob, permute_prob, replace_prob, negative_prob,
                                 eval_mode=False), batch_size=batch_size, shuffle=is_train)
    else:
        loader = data.DataLoader(
            SimCLRDatasetWrapper(dataset, seq_len, 0, 0, 0, 0, 0,
                                 eval_mode=True), batch_size=batch_size, shuffle=is_train)

    return loader
