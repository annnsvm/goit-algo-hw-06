import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate


def run_dfs(graph):
    visited = set()

    def dfs(node, path=[]):
        if node not in visited:
            visited.add(node)
            path.append(node)
            for neighbor in graph.neighbors(node):
                dfs(neighbor, path)
        return path

    paths = []
    for node in graph.nodes:
        if node not in visited:
            paths.append(dfs(node))

    return paths


def run_bfs(graph):
    visited = set()
    paths = []

    def bfs(node):
        queue = [(node, [node])]
        while queue:
            current_node, path = queue.pop(0)
            if current_node not in visited:
                visited.add(current_node)
                for neighbor in graph.neighbors(current_node):
                    queue.append((neighbor, path + [neighbor]))
        paths.append(path)

    for node in graph.nodes:
        if node not in visited:
            bfs(node)

    return paths


def run_dijkstra(graph, start_node):
    distances = {node: float("inf") for node in graph.nodes}
    distances[start_node] = 0
    visited = set()

    while len(visited) < graph.number_of_nodes():
        current_node = min(
            (node for node in graph.nodes if node not in visited),
            key=lambda x: distances[x],
        )
        visited.add(current_node)

        for neighbor in graph.neighbors(current_node):
            new_distance = distances[current_node] + graph[current_node][neighbor].get(
                "weight", 1
            )
            distances[neighbor] = min(distances[neighbor], new_distance)

    # Print results
    result_table = []
    for node, distance in distances.items():
        result_table.append([f"Від {start_node} до {node}", distance])

    headers = ["Маршрут", "Відстань"]
    print(tabulate(result_table, headers=headers, tablefmt="pipe"))

    # Visualization
    edge_labels = {
        (node, neighbor): graph[node][neighbor].get("weight", 1)
        for node in distances
        for neighbor in graph.neighbors(node)
    }

    pos = nx.spring_layout(graph, seed=42)
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=600,
        node_color="skyblue",
        font_size=10,
        font_color="black",
        arrowsize=20,
        font_weight="bold",
    )

    nx.draw_networkx_edge_labels(
        graph, pos, edge_labels=edge_labels, font_color="red", font_size=8
    )

    plt.show()


def graf_algo_tests(matrix, points=[], weighted=False, info=False):
    G = nx.DiGraph()

    for i, (point, duration) in enumerate(zip(points, [sum(row) for row in matrix])):
        G.add_node(i, label=point, duration=duration)

    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] > 0:
                G.add_edge(i, j, weight=matrix[i][j])

    # Анализ основных характеристик
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    avg_degree = sum(dict(G.degree()).values()) / num_nodes

    if info:
        table_data = [
            ["Характеристика", "Значення"],
            ["Кількість вершин", num_nodes],
            ["Кількість ребер", num_edges],
            ["Середній ступінь вершин", avg_degree],
        ]

        print(tabulate(table_data, headers="firstrow", tablefmt="pipe"))
        print("========================================================")

    if weighted == False:
        dfs_paths = run_dfs(G)
        bfs_paths = run_bfs(G)

        paths_table = [
            ["DFS Paths"] + [path for path in dfs_paths],
            ["BFS Paths"] + [path for path in bfs_paths],
        ]

        print(tabulate(paths_table, headers="firstrow", tablefmt="pipe"))
        print("========================================================")

        pos = nx.spring_layout(G, seed=42)
        edge_labels = {(i, j): label["weight"] for i, j, label in G.edges(data=True)}

        plt.figure(figsize=(10, 8))
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=600,
            node_color="skyblue",
            font_size=10,
            font_color="black",
            arrowsize=20,
            font_weight="bold",
        )
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, font_color="red", font_size=8
        )
        nx.draw_networkx_edges(
            G, pos, edgelist=edge_labels.keys(), edge_color="red", width=2, arrowsize=20
        )

        plt.show()
    else:
        start_node = 0
        run_dijkstra(G, start_node)


def main():
    points = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"]

    m_sm_1 = [
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    ]

    m_sm_2 = [
        [0, 10, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [10, 0, 0, 8, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0],
        [5, 0, 0, 0, 6, 0, 7, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 8, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 6, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 4, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0],
        [0, 0, 7, 0, 0, 3, 0, 0, 0, 6, 0, 0, 0, 0, 0],
        [0, 5, 0, 0, 7, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0, 9, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0, 7, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 6, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0],
    ]

    graf_algo_tests(m_sm_1, points, info=True)
    graf_algo_tests(m_sm_2, points, weighted=True)


if __name__ == "__main__":
    main()