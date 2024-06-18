import networkx as nx
import numpy as np
import torch
import os

# 确保目录存在
os.makedirs('./graph', exist_ok=True)
os.makedirs('./embedding', exist_ok=True)
os.makedirs('./weight', exist_ok=True)

def generate_random_graph(v_num, e_num):
    G = nx.gnm_random_graph(v_num, e_num, directed=True)
    edges = list(G.edges())
    return edges

def save_graph_to_file(edges, v_num, e_num, file_path):
    with open(file_path, 'w') as f:
        f.write(f"{v_num} {e_num}\n")
        for src, dst in edges:
            f.write(f"{src} {dst}\n")

def generate_random_features(v_num, F0):
    features = torch.rand(v_num, F0, dtype=torch.float32)
    return features

def save_features_to_file(features, file_path):
    features.numpy().tofile(file_path)

def generate_random_weights(in_dim, out_dim):
    weights = torch.rand(in_dim, out_dim, dtype=torch.float32)
    return weights

def save_weights_to_file(weights, file_path):
    weights.numpy().tofile(file_path)

# 参数
v_num = 20480  # 顶点数量
e_num = 102400  # 边数量
F0 = 64  # 输入顶点特征长度
F1 = 16  # 第一层顶点特征长度
F2 = 8   # 第二层顶点特征长度

# 生成并保存图结构文件
edges = generate_random_graph(v_num, e_num)
save_graph_to_file(edges, v_num, e_num, './graph/large_example_graph_v20480_e102400.txt')

# 生成并保存顶点特征矩阵
features = generate_random_features(v_num, F0)
save_features_to_file(features, './embedding/large_features_v20480_f64.bin')

# 生成并保存权重矩阵
weights1 = generate_random_weights(F0, F1)
save_weights_to_file(weights1, './weight/W_64_16_v20480.bin')

weights2 = generate_random_weights(F1, F2)
save_weights_to_file(weights2, './weight/W_16_8_v20480.bin')
