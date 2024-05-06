import timeit
import random
import sys
from tabulate import tabulate


class DisjointSet:
    def __init__(self, n):
        # Initialize the disjoint set with each node as its own parent and rank 0
        self.parent = [i for i in range(n)]
        self.rank = [0] * n

    def find(self, u):
        # Find operation with path compression
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        # Union operation with rank optimization
        pu, pv = self.find(u), self.find(v)
        if pu == pv:
            return False
        if self.rank[pu] < self.rank[pv]:
            self.parent[pu] = pv
        elif self.rank[pu] > self.rank[pv]:
            self.parent[pv] = pu
        else:
            self.parent[pv] = pu
            self.rank[pu] += 1
        return True


def kruskal(graph):
    # Convert the graph into a list of edges and sort them by weight
    edges = [(u, v, w) for u in graph for v, w in graph[u]]
    edges.sort(key=lambda x: x[2])

    n = len(graph)
    mst = []
    ds = DisjointSet(n)

    # Iterate through sorted edges and add them to MST if they don't form a cycle
    for u, v, w in edges:
        if ds.union(u, v):
            mst.append((u, v, w))
            if len(mst) == n - 1:
                break

    return mst


def generate_random_graph(num_nodes, num_edges):
    # Generate a random graph with the given number of nodes and edges
    graph = {}
    for i in range(num_nodes):
        graph[i] = []
    edges = set()
    while len(edges) < num_edges:
        u, v = random.randint(0, num_nodes - 1), random.randint(0, num_nodes - 1)
        if u != v and (u, v) not in edges and (v, u) not in edges:
            weight = random.randint(1, 100)
            edges.add((u, v))
            graph[u].append((v, weight))
            graph[v].append((u, weight))
    return graph


# Analyze performance, space complexity, and graph density for increasing graph sizes
results = []
for graph_size in [100, 1000, 10000, 100000]:
    num_edges = graph_size * 5  # Adjust edge density as needed
    graph = generate_random_graph(graph_size, num_edges)

    # Measure execution time
    setup_code = "from __main__ import kruskal, graph"
    execution_time = timeit.timeit("kruskal(graph)", setup=setup_code, number=1)

    # Calculate space complexity
    graph_space = sys.getsizeof(graph)  # Space occupied by the input graph
    disjoint_set_space = graph_size * (2 * 8)  # Each node has 2 integers (8 bytes each) in disjoint set
    total_space = graph_space + disjoint_set_space

    # Calculate graph density
    density = (2 * num_edges) / (graph_size * (graph_size - 1))

    results.append([graph_size, num_edges, density, execution_time, f"{total_space} bytes"])

# Print results in tabular format
print(tabulate(results, headers=["Graph Size", "Num Edges", "Density", "Execution Time (seconds)", "Space Complexity"],
               tablefmt="grid"))