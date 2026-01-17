"""
数据预处理脚本：将ASSISTments 2009原始CSV数据转换为DIKT所需的格式
"""
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import scipy.sparse as sp
from tqdm import tqdm

def find_csv_file():
    """查找skill_builder相关的CSV文件"""
    # 先检查data目录
    data_dir = 'data'
    if os.path.exists(data_dir):
        for file in os.listdir(data_dir):
            if 'skill' in file.lower() and file.endswith('.csv'):
                filepath = os.path.join(data_dir, file)
                print(f"找到CSV文件: {filepath}")
                return filepath
    
    # 在整个目录中查找
    for root, dirs, files in os.walk('.'):
        if 'data' in root or root == '.':
            for file in files:
                if 'skill' in file.lower() and file.endswith('.csv'):
                    filepath = os.path.join(root, file)
                    print(f"找到CSV文件: {filepath}")
                    return filepath
    return None

def load_assist09_data(csv_path):
    """加载ASSISTments 2009数据
    返回: (df, correct_col) - 处理后的数据框和correct列名
    """
    print(f"正在加载数据: {csv_path}")
    # 尝试不同的编码
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252', 'gbk', 'gb2312']
    df = None
    for encoding in encodings:
        try:
            # 如果第一列是空字符串或看起来像索引，让pandas自动处理
            df = pd.read_csv(csv_path, low_memory=False, encoding=encoding, index_col=False)
            print(f"成功使用编码: {encoding}")
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    if df is None:
        raise ValueError("无法读取CSV文件，尝试了多种编码都失败")
    
    # 如果第一列是空字符串或Unnamed，删除它
    if df.columns[0] == '' or 'Unnamed' in df.columns[0] or df.columns[0].startswith('"'):
        print(f"  检测到第一列为索引列，将删除: {df.columns[0]}")
        df = df.drop(columns=[df.columns[0]])
    
    # 检查必要的列
    print(f"\n数据列: {df.columns.tolist()}")
    print(f"数据形状: {df.shape}")
    print(f"前5行:\n{df.head()}")
    
    # 尝试识别列名
    user_col = None
    problem_col = None
    skill_col = None
    correct_col = None
    time_col = None
    
    # 常见的列名变体
    for col in df.columns:
        col_lower = col.lower()
        if ('user' in col_lower or 'student' in col_lower) and user_col is None:
            user_col = col
        elif ('problem' in col_lower or 'question' in col_lower) and problem_col is None:
            problem_col = col
        elif ('skill' in col_lower or 'kc' in col_lower or 'concept' in col_lower) and skill_col is None:
            skill_col = col
        elif 'correct' in col_lower and correct_col is None:
            correct_col = col
        elif ('time' in col_lower or 'ms' in col_lower or 'timestamp' in col_lower) and time_col is None:
            time_col = col
    
    # 如果还没找到，尝试通过数据特征识别
    if not user_col:
        for col in df.columns:
            if df[col].dtype in ['int64', 'int32'] and df[col].nunique() > 1000:
                user_col = col
                break
    
    print(f"\n识别的列:")
    print(f"  用户列: {user_col}")
    print(f"  题目列: {problem_col}")
    print(f"  技能列: {skill_col}")
    print(f"  正确性列: {correct_col}")
    print(f"  时间列: {time_col}")
    
    if not all([user_col, problem_col, correct_col]):
        raise ValueError("无法识别必要的列，请检查CSV文件格式")
    
    # 处理缺失值
    df = df.dropna(subset=[user_col, problem_col, correct_col])
    
    # 确保correct是0/1
    if correct_col:
        df[correct_col] = pd.to_numeric(df[correct_col], errors='coerce').fillna(0)
        if df[correct_col].max() > 1:
            df[correct_col] = (df[correct_col] > 0).astype(int)
        else:
            df[correct_col] = df[correct_col].astype(int)
    
    # 如果没有skill列，使用problem作为skill
    if not skill_col:
        print("警告: 未找到技能列，使用problem_id作为skill_id")
        skill_col = problem_col
        df[skill_col] = df[problem_col]
    
    # 如果没有时间列，创建虚拟时间
    if not time_col:
        print("警告: 未找到时间列，使用顺序作为时间")
        df = df.sort_values([user_col, problem_col])
        df['virtual_time'] = df.groupby(user_col).cumcount()
        time_col = 'virtual_time'
    
    # 对ID进行重新编码（从0开始）
    df['user_id_orig'] = df[user_col]
    df['problem_id_orig'] = df[problem_col]
    df['skill_id_orig'] = df[skill_col]
    
    # 处理缺失值，将NaN替换为-1，然后重新编码
    df[problem_col] = df[problem_col].fillna(-1)
    df[skill_col] = df[skill_col].fillna(-1)
    
    # 处理skill_id可能是字符串的情况（如"1_13"）
    # 尝试转换为数值，如果失败则使用分类编码
    if df[skill_col].dtype == 'object':
        print(f"  检测到skill_id为字符串格式，将使用分类编码")
        # 将字符串转换为分类编码
        df[skill_col] = df[skill_col].astype(str)
    
    df['user_id'] = pd.Categorical(df[user_col]).codes
    df['problem_id'] = pd.Categorical(df[problem_col]).codes
    df['skill_id'] = pd.Categorical(df[skill_col]).codes
    
    # 过滤掉无效的ID（-1会被编码为某个值，我们需要确保都是非负的）
    # 如果codes中有-1，说明原始数据中有缺失值，需要过滤
    df = df[df['problem_id'] >= 0]
    df = df[df['skill_id'] >= 0]
    
    # 处理时间
    if 'ms' in time_col.lower():
        df['time_sec'] = pd.to_numeric(df[time_col], errors='coerce').fillna(0) / 1000
    else:
        df['time_sec'] = pd.to_numeric(df[time_col], errors='coerce').fillna(0)
    
    # 按用户和时间排序（保持时间顺序，这对知识追踪很重要）
    # 优先使用sequence_id排序（如果存在），因为它通常表示交互的顺序
    # 然后使用时间作为辅助排序
    sort_cols = ['user_id']
    if 'sequence_id' in df.columns:
        # 确保sequence_id是数值类型
        df['sequence_id'] = pd.to_numeric(df['sequence_id'], errors='coerce').fillna(0)
        sort_cols.append('sequence_id')
        print(f"  使用sequence_id进行排序")
    if 'time_sec' in df.columns:
        sort_cols.append('time_sec')
        print(f"  使用time_sec作为辅助排序")
    
    df = df.sort_values(sort_cols)
    print(f"  排序字段: {sort_cols}")
    
    print(f"\n数据统计:")
    print(f"  用户数: {df['user_id'].nunique()}")
    print(f"  题目数: {df['problem_id'].nunique()}")
    print(f"  技能数: {df['skill_id'].nunique()}")
    print(f"  总记录数: {len(df)}")
    
    return df, correct_col

def format_data_for_dikt(df, correct_col):
    """将数据格式化为DIKT所需的格式"""
    # 按用户分组
    user_data = []
    
    for user_id, group in tqdm(df.groupby('user_id'), desc="格式化数据"):
        # 按sequence_id和时间排序（保持时间顺序）
        # 优先使用sequence_id，因为它通常表示交互的顺序
        sort_cols = []
        if 'sequence_id' in group.columns:
            sort_cols.append('sequence_id')
        if 'time_sec' in group.columns:
            sort_cols.append('time_sec')
        
        if sort_cols:
            group = group.sort_values(sort_cols)
        else:
            # 如果没有排序信息，保持原始顺序（按索引）
            group = group.sort_index()
        
        problems = group['problem_id'].tolist()
        skills = group['skill_id'].tolist()
        # 使用传入的correct_col列名
        answers = group[correct_col].tolist()
        times = group['time_sec'].tolist()
        
        # 确保所有列表长度一致
        min_len = min(len(problems), len(skills), len(answers), len(times))
        problems = problems[:min_len]
        skills = skills[:min_len]
        answers = answers[:min_len]
        times = times[:min_len]
        
        if len(problems) >= 3:  # 至少3个交互
            user_data.append({
                'user_id': user_id,
                'problems': problems,
                'skills': skills,
                'answers': answers,
                'times': times
            })
    
    return user_data

def write_data_file(user_data, output_path):
    """将数据写入DIKT格式的文件"""
    # DIKT格式：每7行为一组（对于assist09）
    # 行0: 空
    # 行1: 空
    # 行2: skills (逗号分隔)
    # 行3: problems (逗号分隔)
    # 行4: answers (逗号分隔)
    # 行5: 空
    # 行6: times (逗号分隔)
    
    with open(output_path, 'w') as f:
        for data in user_data:
            f.write('\n')  # 行0
            f.write('\n')  # 行1
            f.write(','.join(map(str, data['skills'])) + '\n')  # 行2
            f.write(','.join(map(str, data['problems'])) + '\n')  # 行3
            f.write(','.join(map(str, data['answers'])) + '\n')  # 行4
            f.write('\n')  # 行5
            f.write(','.join(map(str, data['times'])) + '\n')  # 行6

def build_graph(train_data, num_problems, num_skills):
    """构建图数据"""
    # 构建问题-技能关系图
    edge_list = []
    
    for data in train_data:
        problems = data['problems']
        skills = data['skills']
        for p, s in zip(problems, skills):
            # 过滤掉无效的索引
            if 0 <= p < num_problems and 0 <= s < num_skills:
                edge_list.append((p, s))
    
    # 创建邻接矩阵（问题-技能二分图）
    if edge_list:
        rows, cols = zip(*edge_list)
        graph = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), 
                             shape=(num_problems, num_skills))
        graph_array = graph.toarray()
    else:
        graph_array = np.zeros((num_problems, num_skills))
    
    return graph_array

def main():
    # 查找CSV文件
    csv_path = find_csv_file()
    if not csv_path:
        raise FileNotFoundError("未找到skill_builder相关的CSV文件，请确保文件在项目目录或data目录下")
    
    # 加载数据
    df, correct_col = load_assist09_data(csv_path)
    
    # 创建题目-技能映射
    ques_skill = df[['problem_id', 'skill_id']].drop_duplicates()
    
    # 格式化数据
    user_data = format_data_for_dikt(df, correct_col)
    
    # 5折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    user_ids = [d['user_id'] for d in user_data]
    unique_users = sorted(set(user_ids))
    
    num_problems = df['problem_id'].max() + 1
    num_skills = df['skill_id'].max() + 1
    
    print(f"\n开始5折交叉验证数据分割...")
    print(f"总用户数: {len(unique_users)}")
    print(f"题目数: {num_problems}")
    print(f"技能数: {num_skills}")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(unique_users)):
        print(f"\n处理Fold {fold}...")
        
        train_users = set([unique_users[i] for i in train_idx])
        test_users = set([unique_users[i] for i in test_idx])
        
        train_data = [d for d in user_data if d['user_id'] in train_users]
        test_data = [d for d in user_data if d['user_id'] in test_users]
        
        # 创建目录
        fold_dir = f'pre_process_data/assist09/{fold}'
        os.makedirs(f'{fold_dir}/train_test', exist_ok=True)
        os.makedirs(f'{fold_dir}/graph', exist_ok=True)
        
        # 写入训练和测试数据
        write_data_file(train_data, f'{fold_dir}/train_test/train_question.txt')
        write_data_file(test_data, f'{fold_dir}/train_test/test_question.txt')
        
        # 写入题目-技能映射
        ques_skill.to_csv(f'{fold_dir}/graph/ques_skill.csv', index=False, header=False)
        
        # 构建图数据
        graph = build_graph(train_data, num_problems, num_skills)
        np.savez(f'{fold_dir}/graph/train_graphs.npz', adj_matrix=graph)
        
        print(f"  Fold {fold}完成: 训练样本{len(train_data)}, 测试样本{len(test_data)}")
    
    print("\n数据预处理完成！")

if __name__ == '__main__':
    main()
