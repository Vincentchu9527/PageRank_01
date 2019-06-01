# -*- coding:utf-8 -*-
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 加载邮件数据文件
emails = pd.read_csv('./data/Emails.csv', encoding='ANSI')
# print(emails.head(5))
# 删除'MetadataTo'或者 'MetadataFrom'为空的数据
emails = emails.dropna(subset=['MetadataTo', 'MetadataFrom'], how='any')
# print(emails.info())

# 读取别名文件 生成对应字典
aliases_file = pd.read_csv('./data/Aliases.csv', encoding='ANSI')
aliases = {}
for index, row in aliases_file.iterrows():
    aliases[row['Alias']] = row['PersonId']
# 读取人名文件 生成对应字典
persons_file = pd.read_csv('./data/Persons.csv', encoding='ANSI')
persons = {}
for index, row in persons_file.iterrows():
    persons[row['Id']] = row['Name']


# 定义别名转换函数
def unify_name(name):
    # 统一name为小写字母
    name = str(name).lower()
    # 去掉,和; 以及 @ 后面的内容
    name = name.replace(',', '')
    name = name.replace(';', '').split('@')[0]
    # 别名转换
    if name in aliases.keys():
        return persons[aliases[name]]
    else:
        return name


# 定义画图函数
def show_graph(graph, layout='spring_layout'):
    # spring_layout 中心散射状布局, circular_layout 圆环状布局
    if layout == 'circular_layout':
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph)
    # 设置网络图中的节点大小, *10000 因为 pagerank 值很小
    nodesize = [x['pagerank'] * 10000 for v, x in graph.nodes(data=True)]
    # 设置网络图中的边长度 用权重衡量
    edgesize = [np.sqrt(e[2]['weight']) for e in graph.edges(data=True)]
    # 绘制节点
    nx.draw_networkx_nodes(graph, pos, node_size=nodesize, alpha=0.4)
    # 绘制边
    nx.draw_networkx_edges(graph, pos, edge_size=edgesize, alpha=0.2)
    # 绘制节点的 label
    nx.draw_networkx_labels(graph, pos, font_size=10)
    plt.show()


# 将邮件数据中寄件人和收件人的姓名进行规范化
emails.MetadataFrom = emails.MetadataFrom.apply(unify_name)
emails.MetadataTo = emails.MetadataTo.apply(unify_name)

# 设置权重等于发邮件的次数
edges_weights_temp = defaultdict(list)
for row in zip(emails.MetadataFrom, emails.MetadataTo):
    temp = (row[0], row[1])
    if temp not in edges_weights_temp:
        edges_weights_temp[temp] = 1
    else:
        edges_weights_temp[temp] = edges_weights_temp[temp] + 1
# 转化格式 (from, to), weight => from, to, weight
edges_weights = [(key[0], key[1], val) for key, val in edges_weights_temp.items()]
# print(edges_weights)

# 创建一个有向图
graph = nx.DiGraph()
# 设置有向图中的路径及权重 (from, to, weight)
graph.add_weighted_edges_from(edges_weights)
# 计算每个节点的PR值并作为节点的pagerank属性
pagerank = nx.pagerank(graph)
nx.set_node_attributes(graph, name='pagerank', values=pagerank)
# 画图
show_graph(graph)

# 将完整的图谱进行精简
# 设置 PR 值的阈值，筛选大于阈值的重要核心节点
pagerank_threshold = 0.005
# 复制一份计算好的图
small_graph = graph.copy()
# 删除PR值小于pagerank_threshold的节点
for n, p_rank in graph.nodes(data=True):
    if p_rank['pagerank'] < pagerank_threshold:
        small_graph.remove_node(n)
# 画网络图, 采用 circular_layout 布局
show_graph(small_graph, 'circular_layout')