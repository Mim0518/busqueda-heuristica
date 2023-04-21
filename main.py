import networkx as nx
from collections import deque
import heapq

tec = nx.Graph()
tec.add_nodes_from(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
                    'K', 'L', 'M', 'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
                    'V', 'W', 'X', 'Y', 'Z', 'E1', 'E2', 'E3'])
tec.add_weighted_edges_from([('A', 'B', 1), ('A', 'E1', 1), ('B', 'E1', 2), ('B', 'D', 1), ('B', 'C', 1),
                             ('E1', 'C', 1), ('E1', 'E2', 2), ('D', 'C', 1), ('D', 'E', 1), ('C', 'F', 1),
                             ('C', 'E2', 1), ('E', 'F', 1), ('F', 'E2', 1), ('E', 'G', 1), ('F', 'G', 1),
                             ('F', 'I', 1), ('E2', 'I', 1), ('E2', 'J', 1), ('E2', 'P', 1), ('G', 'H', 1),
                             ('G', 'I', 2), ('H', 'K', 1), ('H', 'L', 2), ('I', 'L', 3), ('I', 'J', 1), ('K', 'M', 1),
                             ('K', 'L', 2), ('L', 'M', 1), ('L', 'J', 2), ('J', 'P', 2), ('M', 'N', 1), ('M', 'O', 2),
                             ('N', 'U', 4), ('N', 'O', 1), ('O', 'Q', 1), ('O', 'Ñ', 1), ('P', 'Ñ', 1), ('P', 'S', 3),
                             ('P', 'E3', 3), ('Q', 'Ñ', 1), ('Q', 'R', 1), ('Ñ', 'R', 2), ('Ñ', 'S', 4), ('U', 'R', 1),
                             ('U', 'T', 2), ('U', 'W', 2), ('R', 'T', 1), ('R', 'S', 1), ('S', 'T', 1), ('S', 'V', 2),
                             ('S', 'E3', 1), ('T', 'W', 2), ('T', 'V', 2), ('V', 'W', 1), ('V', 'X', 2), ('V', 'E3', 1),
                             ('E3', 'W', 2), ('E3', 'X', 1), ('W', 'Y', 1), ('W', 'X', 1), ('Y', 'X', 2), ('Y', 'Z', 2),
                             ('X', 'Z', 4)])


def bfs_search(graph, start_node, target_node):
    visited = set()
    queue = deque([(start_node, [start_node])])

    while queue:
        node, path = queue.popleft()
        if node not in visited:
            visited.add(node)
            if node == target_node:
                return path
            for neighbor in graph.neighbors(node):
                if neighbor not in visited:
                    queue.append((neighbor, path + [neighbor]))

    return None


def dfs_search(graph, start_node, target_node):
    visited = []
    stack = [start_node]

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.append(node)
            if node == target_node:
                return visited
            neighbors = list(graph.neighbors(node))
            stack.extend(neighbors[::-1])

    return []


def greedy_search(graph, start_node, target_node):
    visited = [start_node]
    while visited[-1] != target_node:
        neighbors = list(graph.neighbors(visited[-1]))
        h = {n: graph[visited[-1]][n]['weight'] for n in neighbors if n not in visited}
        if not h:
            break
        next_node = min(h, key=h.get)
        visited.append(next_node)
    return visited


def dijkstra_search(graph, start_node, target_node):
    # Inicializamos las distancias de cada nodo a un valor "infinito"
    distances = {node: float('inf') for node in graph}
    # La distancia del nodo de inicio a sí mismo es 0
    distances[start_node] = 0

    # Usamos una cola de prioridad (heap) para seleccionar el nodo con la distancia mínima en cada iteración
    priority_queue = [(0, start_node)]

    # Creamos un diccionario para mantener el registro de los nodos visitados
    visited = {}
    # Para mantener el registro del camino más corto desde el nodo de inicio al nodo actual,
    # guardamos cada nodo visitado en un diccionario, donde la clave es el nodo actual y el valor es el nodo previo
    # en el camino más corto
    shortest_path = {start_node: None}

    while len(priority_queue) > 0:
        # Sacamos el nodo con la distancia mínima de la cola de prioridad
        current_distance, current_node = heapq.heappop(priority_queue)

        # Si el nodo actual ya ha sido visitado, lo saltamos
        if current_node in visited:
            continue

        # Marcamos el nodo actual como visitado
        visited[current_node] = True

        # Si llegamos al nodo objetivo, regresamos el camino más corto
        if current_node == target_node:
            path = []
            while current_node is not None:
                path.append(current_node)
                current_node = shortest_path[current_node]
            path.reverse()
            # return path, distances[target_node]
            return path

        # Para cada vecino del nodo actual, actualizamos las distancias si es necesario
        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight['weight']
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                shortest_path[neighbor] = current_node
                heapq.heappush(priority_queue, (distance, neighbor))

    # Si no encontramos un camino al nodo objetivo, regresamos None
    return None, None


if __name__ == '__main__':
    print("Arruina tu vida por Hitler")
    print("Anchura")
    camino = bfs_search(tec, 'A', 'Z')
    print(camino)
    print("Profundidad")
    camino = dfs_search(tec, 'A', 'Z')
    print(camino)
    print("Greedy")
    camino = greedy_search(tec, 'A', 'Z')
    print(camino)
    print("Dijkstra")
    camino = dijkstra_search(tec, 'A', 'Z')
    print(camino)
