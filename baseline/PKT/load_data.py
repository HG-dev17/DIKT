from typing import List, Tuple
import numpy as np
import torch
import torch.utils.data as data

from config import PKTConfig,BASICConfig


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
                        if len(skills) < PKTConfig.MIN_SEQ_LEN:
                            continue
                        skill_list.append(skills)
                    elif index % 7 == 3:
                        problems = list(line.strip().split(split_char))
                        if len(problems) < PKTConfig.MIN_SEQ_LEN:
                            continue
                        problem_list.append(problems)
                    elif index % 7 == 4:
                        ans = list(line.strip().split(split_char))
                        if len(ans) < PKTConfig.MIN_SEQ_LEN:
                            continue
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
                        if len(skills) < PKTConfig.MIN_SEQ_LEN:
                            continue
                        skill_list.append(skills)
                    elif index % 6 == 2:
                        problems = line.strip().split(split_char)
                        problems = list(map(int, problems))
                        if len(problems) < PKTConfig.MIN_SEQ_LEN:
                            continue
                        problem_list.append(problems)
                    elif index % 6 == 3:
                        ans = line.strip().split(split_char)
                        ans = list(map(int, ans))
                        if len(ans) < PKTConfig.MIN_SEQ_LEN:
                            continue
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


class KTDataset(data.Dataset):
    def __init__(self, questionID_allSample, skillID_allSample, label_allSample, n_question, n_skill):
        assert len(questionID_allSample) == len(skillID_allSample) == len(label_allSample)
        self.questionID_allSample = questionID_allSample
        self.skillID_allSample = skillID_allSample
        self.correctness_all = label_allSample
        self.n_question = n_question
        self.n_skill = n_skill
        self.eyeSkill = np.eye(n_skill + 1)

    def __len__(self):
        return len(self.questionID_allSample)

    def __getitem__(self, index: int):
        effLen_1Sample = len(self.questionID_allSample[index]) - 1

        questionID_1Sample = np.array(self.questionID_allSample[index], dtype=np.int32)
        skillID_1Sample = np.array(self.skillID_allSample[index], dtype=np.int32)
        label_1Sample = np.array(self.correctness_all[index], dtype=np.int32)

        currQuestionID_1Sample = questionID_1Sample[:-1]
        currSkillID_1Sample = skillID_1Sample[:-1]
        currLabel_1Sample = label_1Sample[:-1]
        currQuestionAddLabel_1Sample = label_1Sample[:-1] * self.n_question + questionID_1Sample[:-1]
        currSkillAddLabel_1Sample = label_1Sample[:-1] * self.n_skill + skillID_1Sample[:-1]
        currSkill_oneHot_1Sample = self.eyeSkill[skillID_1Sample[:-1]]

        nextQuestionID_1Sample = questionID_1Sample[1:]
        nextSkillID_1Sample = skillID_1Sample[1:]
        nextLabel_1Sample = label_1Sample[1:]
        nextSkill_oneHot_1Sample = self.eyeSkill[skillID_1Sample[1:]]

        return torch.LongTensor([effLen_1Sample]), \
            torch.LongTensor(currQuestionAddLabel_1Sample), torch.LongTensor(currQuestionID_1Sample), \
            torch.LongTensor(currSkillAddLabel_1Sample), torch.LongTensor(currSkillID_1Sample), torch.FloatTensor(
            currSkill_oneHot_1Sample), \
            torch.FloatTensor(currLabel_1Sample), \
            torch.LongTensor(nextQuestionID_1Sample), \
            torch.LongTensor(nextSkillID_1Sample), torch.FloatTensor(nextSkill_oneHot_1Sample), \
            torch.FloatTensor(nextLabel_1Sample)


class PadSequence(object):
    def __call__(self, batch: List[Tuple[torch.Tensor]]):
        batch = sorted(batch, key=lambda y: y[0].shape[0], reverse=True)

        effLen = torch.cat([x[0] for x in batch])

        currQuestionAddLabel = torch.nn.utils.rnn.pad_sequence([x[1] for x in batch], batch_first=True, padding_value=0)
        currQuestionID = torch.nn.utils.rnn.pad_sequence([x[2] for x in batch], batch_first=True, padding_value=0)

        currSkillAddLabel = torch.nn.utils.rnn.pad_sequence([x[3] for x in batch], batch_first=True, padding_value=0)
        currSkillID = torch.nn.utils.rnn.pad_sequence([x[4] for x in batch], batch_first=True, padding_value=0)
        currSkill_oneHot = torch.nn.utils.rnn.pad_sequence([x[5] for x in batch], batch_first=True)

        currLabel = torch.nn.utils.rnn.pad_sequence([x[6] for x in batch], batch_first=True, padding_value=0)
        nextQuestionID = torch.nn.utils.rnn.pad_sequence([x[7] for x in batch], batch_first=True, padding_value=0)

        nextSkillID = torch.nn.utils.rnn.pad_sequence([x[8] for x in batch], batch_first=True, padding_value=0)
        nextSkill_oneHot = torch.nn.utils.rnn.pad_sequence([x[9] for x in batch], batch_first=True)

        nextLabel = torch.nn.utils.rnn.pad_sequence([x[10] for x in batch], batch_first=True, padding_value=0)

        return effLen, \
            currQuestionAddLabel, currQuestionID, \
            currSkillAddLabel, currSkillID, currSkill_oneHot, \
            currLabel, \
            nextQuestionID, \
            nextSkillID, nextSkill_oneHot, \
            nextLabel


def getLoader(is_train, path, batch_size, n_question, n_skill):
    read_data = getReader(path)
    problem_list, skill_list, ans_list = read_data.readData()
    dataset = KTDataset(problem_list, skill_list, ans_list, n_question, n_skill)
    loader = data.DataLoader(dataset=dataset, batch_size=batch_size, drop_last=False, collate_fn=PadSequence(),
                             shuffle=is_train)
    return loader
