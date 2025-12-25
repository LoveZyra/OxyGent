# 概述
shortest_path_demo.py是一个基于 OxyGent 框架的多代理系统并且专注于网络最短路径计算与优化的示例，结合Google OR-Tools的求解能力与JoyCode的编码能力共同构建。该系统能够高效地解决各种网络拓扑中的最短路径问题，特别适用于通信网络、交通路线规划等场景。系统支持从Excel/CSV文件导入网络拓扑数据，计算指定节点间的最短路径，并提供可视化展示功能。
# 核心功能
- **多智能体系统**：基于oxygent框架实现的多智能体协作系统
- **最短路径计算**：基于Google OR-Tools的min_cost_flow算法实现高效的最短路径计算
- **网络数据导入**：支持从Excel/CSV文件导入网络拓扑数据
- **路径可视化**：使用matplotlib和networkx生成网络拓扑图和最短路径可视化
# 核心组件
1. 最短路径算法模块 (shortest_path.py)
    - 基于Google OR-Tools的min_cost_flow实现
    - 支持基本最短路径、城市间最短路径和带约束的最短路径计算
    - 提供路径可视化功能
2. 智能体配置 (shortest_path_demo.py)
    - 配置多智能体系统
    - 设置LLM模型连接
    - 启动应用
# 技术栈
**Python**：核心编程语言
**Google OR-Tools**：提供最短路径算法实现
**Oxygent**：多智能体系统框架
**Pandas**：数据处理和Excel文件读取
**Matplotlib & NetworkX**：网络可视化

# 数据模型
系统处理的网络拓扑数据主要包含以下字段：
cities：节点/城市名称列表
start_cities：边的起始节点
end_cities：边的终止节点
distances：边的距离/权重
costs：边的成本（可选）
如表格图片所示，一个网络拓扑例子（NSFNET）
![image.png](https://s3.cn-north-1.jdcloud-oss.com/shendengbucket1/2025-10-30-15-018nX1LG0YP410rXyC.png)
# 实现细节
## 最短路径算法与可视化
目前，最短路径算法实现的方案较多，启发式、线性规划等，这里基于Google OR-Tools的min_cost_flow方法实现最短路径的精确求解。此外，shortest_path.py 是一个工具文件，目的是为了在OxyGent中注册为Agent可以随时调用的工具，因此，shortest_path.py主要包含以下功能：
**数据导入功能**：info_update()函数从Excel文件读取网络拓扑数据
```python
@shortest_path_tools.tool(description="根据excel更新cityies和distances信息")
async def info_update(file_path, sheet_name=0):
    # 读取 Excel 文件
    df = pd.read_excel(file_path, sheet_name=sheet_name)
    print(df)

    # 遍历数据框中的每一列
    for column in df.columns:
        # 将每一列的数据存储到列表中
        column_data[column] = df[column].dropna().tolist()

    print(column_data)
```
**最短路径计算**：shortest_path()函数计算指定起点和终点间的最短路径
```python
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
```
**路径可视化**：visualize_city_path()函数生成网络拓扑和路径的可视化图像
```python
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

```

## 智能体配置
需要了解OxyGent如何使用的同学可以参考这篇神灯文章：[使用Ollama服务本地启动OxyGent演示案例](http://xingyun.jd.com/shendeng/article/detail/52491?forumId=0&jdme_router=jdme://web/202206081297?url%3Dhttp%3A%2F%2Fsd.jd.com%2Farticle%2F52491)。这里不再赘述OxyGent的配置方法。本示例使用了三个agent，分别为：
- **shortest_path_agent**：负责最短路径计算。
- **excel_agent**：负责excel文件读取操作，并且更新信息。
- **master_agent**：作为主控智能体协调其他智能体。
核心代码如下：
```python
import os
from oxygent import MAS, Config, oxy, preset_tools
from tools.shortest_path import shortest_path_tools

## 注册LLM地址
Config.set_agent_llm_model("default_llm")

## 创建最短路径计算智能体
def create_optimal_agent():
    return oxy.ReActAgent(
        name="shortest_path_agent",
        desc="Agent for computing shortest path between different cities",
        category="agent",
        class_name="ReActAgent",
        tools=["shortest_path_tools"],
        llm_model="default_llm",
        is_entrance=False,
        is_permission_required=False,
        is_save_data=True,
        timeout=30,
        retries=3,
        delay=1,
        is_multimodal_supported=False,
        semaphore=2,
    )
## 注册智能体
oxy_space = [
    oxy.HttpLLM(
        name="default_llm",
        api_key=os.getenv("DEFAULT_LLM_API_KEY"),
        base_url=os.getenv("DEFAULT_LLM_BASE_URL"),
        model_name=os.getenv("DEFAULT_LLM_MODEL_NAME"),
    ),
    shortest_path_tools,
    create_optimal_agent(),
    oxy.ReActAgent(
        name="excel_agent",
        desc="A tool that can operate the file system",
        tools=["shortest_path_tools"],
    ),
    oxy.ReActAgent(
        is_master=True,
        name="master_agent",
        sub_agents=["excel_agent","shortest_path_agent"],
    ),
]
## web启动
async def main():
    async with MAS(oxy_space=oxy_space) as mas:
        await mas.start_web_service(first_query="What is the shortest path between N0 and N3 now?")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```
# 示例
启动应用：**python shortest_path_demo.py**
使用的网络是是美国骨干网络拓扑USnet，包含24个节点（N0-N23），43条边连接。如下图所示，excel_agent读取在usnet.xlsx文件并且更新节点和网络路径信息。
![image.png](https://s3.cn-north-1.jdcloud-oss.com/shendengbucket1/2025-10-30-15-28fjB22ccS6BOT28uWE.png)
获取网络信息后，可以通过人为询问的方式获取两个节点之间的最短路径。
![image.png](https://s3.cn-north-1.jdcloud-oss.com/shendengbucket1/2025-10-30-15-30kyChkV0zeL6jwnL.png)
同时，在项目的文件夹下，也会生成对应的最短路径图。
![image.png](https://s3.cn-north-1.jdcloud-oss.com/shendengbucket1/2025-10-30-15-33xnCpUXqpD33jesp15.png)


# 总结
OxyGent结合JoyCode的高效开发为大模型的应用提供了更多可能性。基于不同能力需求，用户可以随时打造出自己的千军万马！**shortest_path_demo**提供一个简单的应用场景示例，现实优化场景可能会面临较多限制与不确定因素，在大模型的辅助下，可最大化帮助开发人员简化问题难度，并降低操作学习成本。