
import sys
import math

class PriorityQueue:
    """
    Manual implementation of a min-heap priority queue for Dijkstra's algorithm.
    Uses a list to store (distance, vertex) pairs and maintains heap property.
    """

    def __init__(self):
        self.heap = []
        self.size = 0

    def parent(self, i):
        """Return parent index of node at index i"""
        return (i - 1) // 2

    def left_child(self, i):
        """Return left child index of node at index i"""
        return 2 * i + 1

    def right_child(self, i):
        """Return right child index of node at index i"""
        return 2 * i + 2

    def swap(self, i, j):
        """Swap elements at indices i and j"""
        self.heap[i], self.heap[j] = self.heap[j], self.heap[i]

    def heapify_up(self, i):
        """Maintain heap property by moving element up"""
        while i > 0 and self.heap[self.parent(i)][0] > self.heap[i][0]:
            self.swap(i, self.parent(i))
            i = self.parent(i)

    def heapify_down(self, i):
        """Maintain heap property by moving element down"""
        while self.left_child(i) < self.size:
            min_child = self.left_child(i)

            # Find the smaller child
            if (self.right_child(i) < self.size and
                    self.heap[self.right_child(i)][0] < self.heap[min_child][0]):
                min_child = self.right_child(i)

            # If current node is smaller than both children, heap property is satisfied
            if self.heap[i][0] <= self.heap[min_child][0]:
                break

            self.swap(i, min_child)
            i = min_child

    def insert(self, distance, vertex):
        """Insert a new (distance, vertex) pair into the priority queue"""
        self.heap.append((distance, vertex))
        self.size += 1
        self.heapify_up(self.size - 1)

    def extract_min(self):
        """Remove and return the minimum element from the priority queue"""
        if self.size == 0:
            return None

        min_element = self.heap[0]
        self.heap[0] = self.heap[self.size - 1]
        self.size -= 1
        self.heap.pop()

        if self.size > 0:
            self.heapify_down(0)

        return min_element

    def is_empty(self):
        """Check if the priority queue is empty"""
        return self.size == 0

    def decrease_key(self, vertex, new_distance):
        """Decrease the key of a vertex in the priority queue"""
        for i in range(self.size):
            if self.heap[i][1] == vertex:
                if new_distance < self.heap[i][0]:
                    self.heap[i] = (new_distance, vertex)
                    self.heapify_up(i)
                break

class Graph:
    """
    Graph class that represents a weighted directed graph using adjacency list.
    Supports both directed and undirected graphs for Dijkstra's algorithm.
    """

    def __init__(self, vertices, directed=False):
        self.V = vertices  # Number of vertices
        self.directed = directed  # Graph type (directed or undirected)
        # Adjacency list: each vertex maps to list of (neighbor, weight) pairs
        self.graph = [[] for _ in range(vertices)]
        self.vertex_names = {}  # Map vertex indices to names (optional)

    def add_edge(self, u, v, weight):
        """
        Add an edge from vertex u to vertex v with given weight.
        For undirected graphs, adds edge in both directions.
        """
        if u < 0 or u >= self.V or v < 0 or v >= self.V:
            raise ValueError(f"Vertex indices must be between 0 and {self.V-1}")

        if weight < 0:
            raise ValueError("Dijkstra's algorithm requires non-negative edge weights")

        # Add edge u -> v
        self.graph[u].append((v, weight))

        # For undirected graphs, add edge v -> u
        if not self.directed:
            self.graph[v].append((u, weight))

    def set_vertex_name(self, vertex, name):
        """Set a custom name for a vertex (optional feature)"""
        if 0 <= vertex < self.V:
            self.vertex_names[vertex] = name

    def get_vertex_name(self, vertex):
        """Get the name of a vertex, or return the index if no name is set"""
        return self.vertex_names.get(vertex, str(vertex))

    def dijkstra(self, src):
        """
        Implement Dijkstra's shortest path algorithm from start vertex.
        Returns distances and previous vertices for path reconstruction.

        Time Complexity: O((V + E) log V) where V is vertices and E is edges
        Space Complexity: O(V) for storing distances and previous vertices
        """
        if src < 0 or src >= self.V:
            raise ValueError(f"Start vertex must be between 0 and {self.V-1}")

        # Initialize distances to infinity and previous vertices to None
        distances = [float('inf')] * self.V
        previous = [None] * self.V
        visited = [False] * self.V

        # Distance from start to itself is 0
        distances[src] = 0

        # Create priority queue and add start vertex
        pq = PriorityQueue()
        pq.insert(0, src)

        print(f"\nRunning Dijkstra's algorithm from vertex {self.get_vertex_name(src)}...")
        print("Step-by-step execution:")
        step = 1

        while not pq.is_empty():
            # Extract vertex with minimum distance
            current_dist, current_vertex = pq.extract_min()

            # Skip if already visited (handles duplicate entries in priority queue)
            if visited[current_vertex]:
                continue

            visited[current_vertex] = True

            print(f"Step {step}: Processing vertex {self.get_vertex_name(current_vertex)} "
                  f"with distance {current_dist}")

            # Examine all neighbors of current vertex
            for neighbor, weight in self.graph[current_vertex]:
                if not visited[neighbor]:
                    # Calculate new distance through current vertex
                    new_distance = distances[current_vertex] + weight

                    # If new path is shorter, update distance and previous vertex
                    if new_distance < distances[neighbor]:
                        distances[neighbor] = new_distance
                        previous[neighbor] = current_vertex
                        pq.insert(new_distance, neighbor)

                        print(f"  Updated distance to {self.get_vertex_name(neighbor)}: "
                              f"{new_distance} (via {self.get_vertex_name(current_vertex)})")

            step += 1

        return distances, previous

    def reconstruct_path(self, previous, src, dest):
        """
        Reconstruct the shortest path from start to end
        using the previous vertices array from Dijkstra's algorithm.
        """
        if previous[dest] is None and src != dest:
            return None  # No path exists

        path = []
        current = dest

        # Trace back from end to start
        while current is not None:
            path.append(current)
            current = previous[current]

        # Reverse to get path from start to end
        path.reverse()
        return path

    def print_shortest_paths(self, src, distances, previous):
        """
        Print a formatted table showing shortest distances and paths
        from start to all other vertices.
        """
        print(f"\n{'='*80}")
        print(f"SHORTEST PATHS FROM VERTEX {self.get_vertex_name(src)}")
        print(f"{'='*80}")

        # Print table header
        print(f"{'Destination':<12} {'Distance':<10} {'Path':<50}")
        print(f"{'-'*12} {'-'*10} {'-'*50}")

        # Print results for each vertex
        for dest in range(self.V):
            dest_name = self.get_vertex_name(dest)

            if distances[dest] == float('inf'):
                distance_str = "∞"
                path_str = "No path"
            else:
                distance_str = str(distances[dest])
                path = self.reconstruct_path(previous, src, dest)
                if path:
                    path_names = [self.get_vertex_name(v) for v in path]
                    path_str = " → ".join(path_names)
                else:
                    path_str = "No path"

            print(f"{dest_name:<12} {distance_str:<10} {path_str:<50}")

        print(f"{'='*80}")

    def print_graph_info(self):
        """Print information about the graph structure"""
        print(f"\nGraph Information:")
        print(f"Type: {'Directed' if self.directed else 'Undirected'}")
        print(f"Vertices: {self.V}")
        print(f"Edges: {sum(len(adj_list) for adj_list in self.graph) // (1 if self.directed else 2)}")

        print(f"\nAdjacency List Representation:")
        for vertex in range(self.V):
            neighbors = []
            for neighbor, weight in self.graph[vertex]:
                neighbors.append(f"{self.get_vertex_name(neighbor)}({weight})")

            if neighbors:
                print(f"  {self.get_vertex_name(vertex)}: {', '.join(neighbors)}")
            else:
                print(f"  {self.get_vertex_name(vertex)}: No outgoing edges")

def get_user_input():
    """
    Get graph input from user with comprehensive error handling
    and support for various input formats.
    """
    print("="*60)
    print("DIJKSTRA'S SHORTEST PATH ALGORITHM")
    print("="*60)

    # Get number of vertices
    while True:
        try:
            vertices = int(input("\nEnter the number of vertices: "))
            if vertices <= 0:
                print("Number of vertices must be positive.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    # Ask for graph type
    while True:
        graph_type = input("\nIs the graph directed? (y/n): ").lower().strip()
        if graph_type in ['y', 'yes', '1']:
            directed = True
            break
        elif graph_type in ['n', 'no', '0']:
            directed = False
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")

    # Create graph
    graph = Graph(vertices, directed)

    # Option to set vertex names
    name_vertices = input("\nWould you like to assign names to vertices? (y/n): ").lower().strip()
    if name_vertices in ['y', 'yes', '1']:
        print("Enter names for each vertex (press Enter to use default index):")
        for i in range(vertices):
            name = input(f"Vertex {i}: ").strip()
            if name:
                graph.set_vertex_name(i, name)

    # Get number of edges
    while True:
        try:
            edges = int(input(f"\nEnter the number of edges: "))
            if edges < 0:
                print("Number of edges cannot be negative.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    # Get edge information
    print(f"\nEnter edges in format: start, end, distance")
    print(f"Vertices should be numbered from 0 to {vertices-1}")

    for i in range(edges):
        while True:
            try:
                edge_input = input(f"Edge {i+1}: ").strip().split()
                if len(edge_input) != 3:
                    print("Please enter exactly 3 values: start, end, distance")
                    continue

                u, v, weight = int(edge_input[0]), int(edge_input[1]), float(edge_input[2])
                graph.add_edge(u, v, weight)
                break

            except ValueError:
                print("Please enter valid numbers for start, end, distance .")
            except Exception as e:
                print(f"Error: {e}")

    # Get start vertex
    while True:
        try:
            src = int(input(f"\nEnter start vertex (0 to {vertices-1}): "))
            if src < 0 or src >= vertices:
                print(f"Start vertex must be between 0 and {vertices-1}.")
                continue
            break
        except ValueError:
            print("Please enter a valid integer.")

    return graph, src

def run_additional_tests(graph):
    """
    Run additional test cases to demonstrate robustness
    """
    print(f"\n{'='*80}")
    print("ADDITIONAL TESTING")
    print(f"{'='*80}")

    # Test all possible start vertices
    print("\nTesting algorithm with different start vertices:")

    for src in range(graph.V):
        print(f"\n--- Start: {graph.get_vertex_name(src)} ---")
        distances, previous = graph.dijkstra(src)

        # Show only reachable vertices to keep output manageable
        reachable = [v for v in range(graph.V) if distances[v] != float('inf')]

        if len(reachable) <= 5:  # Show all if small graph
            graph.print_shortest_paths(src, distances, previous)
        else:  # Show summary for large graphs
            print(f"Reachable vertices: {len(reachable)}")
            print(f"Average distance: {sum(distances[v] for v in reachable) / len(reachable):.2f}")

def create_sample_graph():
    """
    Create a sample graph for demonstration purposes
    """
    print("\nCreating sample graph for demonstration...")

    # Create a sample weighted graph
    graph = Graph(6, directed=False)

    # Set vertex names for better readability
    vertex_names = ['A', 'B', 'C', 'D', 'E', 'F']
    for i, name in enumerate(vertex_names):
        graph.set_vertex_name(i, name)

    # Add edges (creating a connected graph with interesting paths)
    edges = [
        (0, 1, 4),   # A-B: 4
        (0, 2, 2),   # A-C: 2
        (1, 2, 1),   # B-C: 1
        (1, 3, 5),   # B-D: 5
        (2, 3, 8),   # C-D: 8
        (2, 4, 10),  # C-E: 10
        (3, 4, 2),   # D-E: 2
        (3, 5, 6),   # D-F: 6
        (4, 5, 3)    # E-F: 3
    ]

    for u, v, weight in edges:
        graph.add_edge(u, v, weight)

    return graph

def main():
    """
    Main function that orchestrates the program execution
    """
    try:
        print("Welcome to Dijkstra's Shortest Path Algorithm Implementation!")
        print("This program finds the shortest paths from a start vertex to all other vertices.")

        # Ask user for input method
        while True:
            choice = input("\nChoose input method:\n1. Enter custom graph\n2. Use sample graph\nChoice (1/2): ").strip()

            if choice == '1':
                graph, src = get_user_input()
                break
            elif choice == '2':
                graph = create_sample_graph()
                src = 0  # Start from vertex A
                break
            else:
                print("Please enter 1 or 2.")

        # Display graph information
        graph.print_graph_info()

        # Run Dijkstra's algorithm
        distances, previous = graph.dijkstra(src)

        # Display results
        graph.print_shortest_paths(src, distances, previous)

        # Ask if user wants to test with different start point
        while True:
            test_more = input("\nWould you like to test with a different s"
                              " vertex? (y/n): ").lower().strip()
            if test_more in ['y', 'yes', '1']:
                while True:
                    try:
                        new_src = int(input(f"Enter new start vertex (0 to {graph.V-1}): "))
                        if 0 <= new_src < graph.V:
                            distances, previous = graph.dijkstra(new_src)
                            graph.print_shortest_paths(new_src, distances, previous)
                            break
                        else:
                            print(f"Start vertex must be between 0 and {graph.V-1}.")
                    except ValueError:
                        print("Please enter a valid integer.")
            elif test_more in ['n', 'no', '0']:
                break
            else:
                print("Please enter 'y' for yes or 'n' for no.")

        # Option to run comprehensive tests
        comprehensive_test = input("\nWould you like to run comprehensive tests? (y/n): ").lower().strip()
        if comprehensive_test in ['y', 'yes', '1']:
            run_additional_tests(graph)

        print("\nThank you for using Dijkstra's Algorithm Implementation!")

    except KeyboardInterrupt:
        print("\n\nProgram terminated by user.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        print("Please check your input and try again.")

if __name__ == "__main__":
    main()
