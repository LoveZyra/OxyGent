#!/usr/bin/env python3
"""
使用 Google OR-Tools 解决最短路径问题的示例

这个示例展示了如何使用 OR-Tools 的图算法来解决城市间的最短路径问题。
"""

from ortools.graph.python import min_cost_flow
import time
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from oxygent import oxy

shortest_path_tools = oxy.FunctionHub(
    name="shortest_path_tools")
column_data = {}


@shortest_path_tools.tool(description="Update city and distance information based on Excel.")
async def info_update(file_path, sheet_name=0):
    # 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(df)

    # 遍历数据框中的每一列
    for column in df.columns:
        # 将每一列的数据存储到列表中
        column_data[column] = df[column].dropna().tolist()

    print(column_data)
    if len(column_data) > 0:
        return "File Read Success!"

    return "File is Empty"


@shortest_path_tools.tool(description="A tool that can calculate the shortest path between different points")
async def shortest_path(start_city: str, end_city):
    # 城市列表
    city_to_index = {city: i for i, city in enumerate(column_data['cities'])}
    print(start_city, end_city)

    cities = column_data['cities']
    start_cities = column_data['start_cities']
    end_cities = column_data['end_cities']
    distances = column_data['distances']
    # 转换城市名称为索引
    start_nodes = [city_to_index[city] for city in start_cities]
    end_nodes = [city_to_index[city] for city in end_cities]
    # 创建有向图求解器
    sp_func = min_cost_flow.SimpleMinCostFlow()

    # 添加每条边到图中 (注意：我们需要添加双向边，因为城市之间的道路是双向的)
    for i in range(len(start_nodes)):
        sp_func.add_arc_with_capacity_and_unit_cost(
            start_nodes[i], end_nodes[i], 1, distances[i])
        sp_func.add_arc_with_capacity_and_unit_cost(
            end_nodes[i], start_nodes[i], 1, distances[i])

    # 设置起点和终点的供应/需求
    sp_func.set_node_supply(city_to_index[start_city], 1)  # 起点
    sp_func.set_node_supply(city_to_index[end_city], -1)  # 终点

    # 求解最短路径
    start_time = time.time()
    status = sp_func.solve()
    end_time = time.time()

    # 构建结果
    result = {}
    if status == min_cost_flow.SimpleMinCostFlow.OPTIMAL:
        result["status"] = "optimal"
        result["distance"] = sp_func.optimal_cost()
        result["solve_time"] = end_time - start_time

        # 遍历所有边，找出流量为 1 的边（即最短路径上的边）
        path = []
        path_cities = []
        for i in range(sp_func.num_arcs()):
            if sp_func.flow(i) > 0:
                tail = sp_func.tail(i)
                head = sp_func.head(i)
                path.append((tail, head))
                path_cities.append(f"{cities[tail]} -> {cities[head]}")

        result["path"] = path
        result["path_cities"] = path_cities
        # 可视化城市路径
        visualize_city_path(cities, start_cities, end_cities, distances, path)
    else:
        result["status"] = "not_optimal"

    return result


def visualize_city_path(cities, start_cities, end_cities, distances, path):
    """
    可视化城市图和最短路径

    参数:
        cities: 城市列表
        start_cities: 起始城市列表
        end_cities: 终止城市列表
        distances: 距离列表
        path: 最短路径上的边列表 (使用城市索引)
    """
    try:
        # 创建图
        G = nx.Graph()

        # 添加节点
        for city in cities:
            G.add_node(city)

        # 添加边和权重
        for i in range(len(start_cities)):
            G.add_edge(start_cities[i], end_cities[i], weight=distances[i])

        # 创建位置字典 - 使用中国主要城市的大致地理位置
        # 这些坐标是近似值，用于可视化目的
        # 创建位置字典
        city_positions = nx.spring_layout(G, seed=42)

        # 绘制图
        plt.figure(figsize=(12, 10))

        # 绘制所有边
        nx.draw_networkx_edges(G, city_positions, alpha=0.3, width=1)

        # 高亮显示最短路径上的边
        path_edges = []
        for u, v in path:
            path_edges.append((cities[u], cities[v]))

        nx.draw_networkx_edges(G, city_positions, edgelist=path_edges, width=3, edge_color='r')

        # 绘制节点
        nx.draw_networkx_nodes(G, city_positions, node_size=700, node_color='lightblue')

        # 绘制节点标签
        nx.draw_networkx_labels(G, city_positions, font_size=12, font_family='SimHei')

        # 绘制边权重
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, city_positions, edge_labels=edge_labels, font_size=8)

        plt.title("中国城市间最短路径", fontsize=16, fontfamily='SimHei')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig("city_shortest_path.png")
        print("\n城市路径图已保存为 'city_shortest_path.png'")
    except Exception as e:
        print(f"可视化过程中出错: {e}")
        print("跳过可视化步骤...")
