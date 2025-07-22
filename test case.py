"""
Test Cases for Dijkstra's Algorithm Implementation
CSC2103 Group Project - Problem 2

This file contains various test cases to validate the correctness
and robustness of the Dijkstra's algorithm implementation.
"""

# Import the main implementation (assuming it's in dijkstra.py)
# from dijkstra import Graph, PriorityQueue
from djikstra import Graph, PriorityQueue

def test_basic_functionality():
    """Test basic functionality with a simple connected graph"""
    print("="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)

    # Create a simple 5-vertex graph
    graph = Graph(5, directed=False)

    # Set vertex names for clarity
    names = ['A', 'B', 'C', 'D', 'E']
    for i, name in enumerate(names):
        graph.set_vertex_name(i, name)

    # Add edges to create a connected graph
    edges = [
        (0, 1, 10),  # A-B: 10
        (0, 4, 5),   # A-E: 5
        (1, 2, 1),   # B-C: 1
        (1, 4, 2),   # B-E: 2
        (2, 3, 4),   # C-D: 4
        (3, 0, 7),   # D-A: 7
        (3, 2, 6),   # D-C: 6
        (4, 1, 3),   # E-B: 3
        (4, 2, 9),   # E-C: 9
        (4, 3, 2)    # E-D: 2
    ]

    for u, v, weight in edges:
        graph.add_edge(u, v, weight)

    # Test from vertex A (0)
    print("Testing shortest paths from vertex A:")
    distances, previous = graph.dijkstra(0)
    graph.print_shortest_paths(0, distances, previous)

    # Verify expected results
    expected_distances = [0, 8, 9, 7, 5]
    expected_paths = [
        [0],           # A to A
        [0, 4, 1],     # A to B via E
        [0, 4, 1, 2],  # A to C via E, B
        [0, 4, 3],     # A to D via E
        [0, 4]         # A to E
    ]

    print("\nVerification:")
    for i, (expected_dist, expected_path) in enumerate(zip(expected_distances, expected_paths)):
        actual_path = graph.reconstruct_path(previous, 0, i)
        if distances[i] == expected_dist and actual_path == expected_path:
            print(f"✓ Vertex {names[i]}: PASS")
        else:
            print(f"✗ Vertex {names[i]}: FAIL - Expected {expected_dist}, got {distances[i]}")

def test_disconnected_graph():
    """Test with a disconnected graph"""
    print("\n" + "="*60)
    print("TEST 2: Disconnected Graph")
    print("="*60)

    # Create a disconnected graph with 6 vertices
    graph = Graph(6, directed=False)

    # Component 1: vertices 0, 1, 2
    graph.add_edge(0, 1, 3)
    graph.add_edge(1, 2, 2)

    # Component 2: vertices 3, 4
    graph.add_edge(3, 4, 1)

    # Vertex 5 is isolated

    print("Testing shortest paths from vertex 0:")
    distances, previous = graph.dijkstra(0)
    graph.print_shortest_paths(0, distances, previous)

    # Verify unreachable vertices
    unreachable = [i for i in range(6) if distances[i] == float('inf')]
    print(f"\nUnreachable vertices from 0: {unreachable}")
    assert 3 in unreachable and 4 in unreachable and 5 in unreachable
    print("✓ Disconnected graph handling: PASS")

def test_directed_graph():
    """Test with a directed graph"""
    print("\n" + "="*60)
    print("TEST 3: Directed Graph")
    print("="*60)

    # Create a directed graph
    graph = Graph(4, directed=True)

    # Set vertex names
    names = ['S', 'A', 'B', 'T']
    for i, name in enumerate(names):
        graph.set_vertex_name(i, name)

    # Add directed edges
    edges = [
        (0, 1, 1),   # S -> A: 1
        (0, 2, 4),   # S -> B: 4
        (1, 2, 2),   # A -> B: 2
        (1, 3, 6),   # A -> T: 6
        (2, 3, 3)    # B -> T: 3
    ]

    for u, v, weight in edges:
        graph.add_edge(u, v, weight)

    print("Testing shortest paths in directed graph from S:")
    distances, previous = graph.dijkstra(0)
    graph.print_shortest_paths(0, distances, previous)

    # Verify shortest path to T is S -> A -> B -> T with distance 6
    path_to_t = graph.reconstruct_path(previous, 0, 3)
    expected_path = [0, 1, 2, 3]
    expected_distance = 6

    if distances[3] == expected_distance and path_to_t == expected_path:
        print("✓ Directed graph shortest path: PASS")
    else:
        print("✗ Directed graph shortest path: FAIL")

def test_single_vertex():
    """Test with a single vertex graph"""
    print("\n" + "="*60)
    print("TEST 4: Single Vertex Graph")
    print("="*60)

    graph = Graph(1, directed=False)
    graph.set_vertex_name(0, 'ONLY')

    print("Testing single vertex graph:")
    distances, previous = graph.dijkstra(0)
    graph.print_shortest_paths(0, distances, previous)

    # Verify distance to self is 0
    assert distances[0] == 0
    assert graph.reconstruct_path(previous, 0, 0) == [0]
    print("✓ Single vertex graph: PASS")

def test_self_loops():
    """Test with self-loops"""
    print("\n" + "="*60)
    print("TEST 5: Self-Loops")
    print("="*60)

    graph = Graph(3, directed=False)

    # Add regular edges
    graph.add_edge(0, 1, 5)
    graph.add_edge(1, 2, 3)

    # Add self-loops (should be ignored in shortest path)
    graph.add_edge(0, 0, 10)
    graph.add_edge(1, 1, 2)

    print("Testing graph with self-loops:")
    distances, previous = graph.dijkstra(0)
    graph.print_shortest_paths(0, distances, previous)

    # Self-loops should not affect shortest paths
    assert distances[0] == 0
    assert distances[1] == 5
    assert distances[2] == 8
    print("✓ Self-loops handling: PASS")

def test_zero_weight_edges():
    """Test with zero-weight edges"""
    print("\n" + "="*60)
    print("TEST 6: Zero-Weight Edges")
    print("="*60)

    graph = Graph(4, directed=False)

    # Add edges with zero weights
    edges = [
        (0, 1, 0),   # Free edge
        (1, 2, 3),   # Regular edge
        (2, 3, 0),   # Another free edge
        (0, 3, 10)   # Direct but expensive
    ]

    for u, v, weight in edges:
        graph.add_edge(u, v, weight)

    print("Testing graph with zero-weight edges:")
    distances, previous = graph.dijkstra(0)
    graph.print_shortest_paths(0, distances, previous)

    # Should take the free path: 0 -> 1 -> 2 -> 3 (distance 3)
    # instead of direct path: 0 -> 3 (distance 10)
    assert distances[3] == 3
    expected_path = [0, 1, 2, 3]
    actual_path = graph.reconstruct_path(previous, 0, 3)
    assert actual_path == expected_path
    print("✓ Zero-weight edges: PASS")

def test_priority_queue():
    """Test the priority queue implementation separately"""
    print("\n" + "="*60)
    print("TEST 7: Priority Queue Implementation")
    print("="*60)

    pq = PriorityQueue()

    # Test insertions
    test_data = [(5, 'E'), (1, 'A'), (3, 'C'), (2, 'B'), (4, 'D')]

    print("Inserting elements:", test_data)
    for dist, vertex in test_data:
        pq.insert(dist, vertex)

    # Test extractions (should come out in sorted order)
    extracted = []
    while not pq.is_empty():
        extracted.append(pq.extract_min())

    print("Extracted in order:", extracted)

    # Verify correct order
    expected = [(1, 'A'), (2, 'B'), (3, 'C'), (4, 'D'), (5, 'E')]
    assert extracted == expected
    print("✓ Priority queue implementation: PASS")

def test_error_handling():
    """Test error handling and edge cases"""
    print("\n" + "="*60)
    print("TEST 8: Error Handling")
    print("="*60)

    graph = Graph(3, directed=False)

    # Test 1: Invalid vertex indices
    try:
        graph.add_edge(-1, 0, 5)
        print("✗ Should have raised error for negative vertex")
    except ValueError as e:
        print(f"✓ Negative vertex error caught: {e}")

    try:
        graph.add_edge(0, 5, 5)
        print("✗ Should have raised error for out-of-bounds vertex")
    except ValueError as e:
        print(f"✓ Out-of-bounds vertex error caught: {e}")

    # Test 2: Negative weights
    try:
        graph.add_edge(0, 1, -5)
        print("✗ Should have raised error for negative weight")
    except ValueError as e:
        print(f"✓ Negative weight error caught: {e}")

    # Test 3: Invalid source vertex for Dijkstra
    try:
        graph.dijkstra(-1)
        print("✗ Should have raised error for invalid source")
    except ValueError as e:
        print(f"✓ Invalid source error caught: {e}")

    try:
        graph.dijkstra(5)
        print("✗ Should have raised error for out-of-bounds source")
    except ValueError as e:
        print(f"✓ Out-of-bounds source error caught: {e}")

def test_large_graph():
    """Test with a larger graph to verify scalability"""
    print("\n" + "="*60)
    print("TEST 9: Large Graph Performance")
    print("="*60)

    import time

    # Create a larger graph (complete graph with 100 vertices)
    n = 100
    graph = Graph(n, directed=False)

    print(f"Creating complete graph with {n} vertices...")

    # Add edges to create a complete graph
    edge_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            weight = abs(i - j)  # Simple weight function
            graph.add_edge(i, j, weight)
            edge_count += 1

    print(f"Added {edge_count} edges")

    # Test algorithm performance
    start_time = time.time()
    distances, previous = graph.dijkstra(0)
    end_time = time.time()

    print(f"Algorithm completed in {end_time - start_time:.4f} seconds")

    # Verify some results
    print(f"Distance to vertex 50: {distances[50]}")
    print(f"Distance to vertex 99: {distances[99]}")

    # In a complete graph, direct edges should be shortest
    assert distances[50] == 50  # Direct edge weight
    assert distances[99] == 99  # Direct edge weight
    print("✓ Large graph performance: PASS")

def run_all_tests():
    """Run all test cases"""
    print("DIJKSTRA'S ALGORITHM - COMPREHENSIVE TEST SUITE")
    print("=" * 80)

    test_functions = [
        test_basic_functionality,
        test_disconnected_graph,
        test_directed_graph,
        test_single_vertex,
        test_self_loops,
        test_zero_weight_edges,
        test_priority_queue,
        test_error_handling,
        test_large_graph
    ]

    passed = 0
    total = len(test_functions)

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            print(f"\n✓ {test_func.__name__}: PASSED")
        except Exception as e:
            print(f"\n✗ {test_func.__name__}: FAILED - {e}")

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    print("=" * 80)

if __name__ == "__main__":
    # Import the main classes (you'll need to adjust the import based on your file structure)
    # For demonstration, assuming the classes are in the same file

    run_all_tests()
