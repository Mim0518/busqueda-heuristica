import tkinter as tk
from tkinter import ttk
import networkx as nx
from collections import deque
import heapq
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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


def a_search(graph, start_node, target_node):
    frontier = [(0, start_node)]
    cost_so_far = {start_node: 0}
    came_from = {start_node: None}
    while frontier:
        current_cost, current_node = heapq.heappop(frontier)
        if current_node == target_node:
            break
        for neighbor in graph[current_node]:
            new_cost = cost_so_far[current_node] + graph[current_node][neighbor]['weight']
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                cost_estimate = 0
                total_cost_estimate = new_cost + cost_estimate
                heapq.heappush(frontier, (total_cost_estimate, neighbor))
                came_from[neighbor] = current_node
    if target_node not in came_from:
        return None
    path = [target_node]
    current = target_node
    while current != start_node:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


# La función toma como entrada un grafo graph, un nodo de inicio start y un nodo de destino end.
def branch_and_bound(graph, start, end):
    # Crea una cola de prioridad (heap) inicializada con una tupla que contiene el costo del camino (inicialmente 0) y
    # una lista que contiene el nodo de inicio.
    queue = [(0, [start])]
    # Crea un conjunto vacío para almacenar los nodos visitados.
    visited = set()
    # Mientras la cola no esté vacía, extrae el elemento con el menor costo del camino desde la cola y comprueba si el
    # nodo actual es el nodo de destino. Si es así, devuelve el costo y el camino.
    while queue:
        (cost, path) = heapq.heappop(queue)
        node = path[-1]
        if node == end:
            return path
        # Si el nodo actual no ha sido visitado, lo marca como visitado y añade sus vecinos a la cola con los costos y
        # caminos actualizados. Los vecinos que ya han sido visitados no se añaden a la cola.
        if node not in visited:
            visited.add(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    new_cost = cost + graph[node][neighbor]['weight']
                    new_path = path + [neighbor]
                    heapq.heappush(queue, (new_cost, new_path))
    # Si no se ha encontrado un camino hasta el nodo de destino, devuelve None.                
    return None


def draw_graph(graph, path=None):
    figure.clear()  # Limpiar el gráfico existente
    ax = figure.add_subplot(1, 1, 1)  # Agregar un nuevo eje
    pos = nx.spring_layout(graph)

    # Dibujar nodos y aristas del grafo
    nx.draw_networkx_nodes(graph, pos, ax=ax)
    nx.draw_networkx_labels(graph, pos, ax=ax)
    nx.draw_networkx_edges(graph, pos, ax=ax)

    # Si se proporciona un camino, resaltarlo con aristas rojas más gruesas
    if path:
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(graph, pos, edgelist=edges, edge_color="r", width=2, ax=ax)

    ax.axis("off")
    canvas.draw()


# Realiza la búsqueda y muestra el resultado en el cuadro de texto
def run_search():
    start_node = start_var.get()
    target_node = target_var.get()
    algorithm = algorithm_var.get()
    start_node = start_node.upper()
    target_node = target_node.upper()

    if algorithm == "BFS":
        path = bfs_search(tec, start_node, target_node)
    elif algorithm == "DFS":
        path = dfs_search(tec, start_node, target_node)
    elif algorithm == "Greedy":
        path = greedy_search(tec, start_node, target_node)
    elif algorithm == "Dijkstra":
        path = dijkstra_search(tec, start_node, target_node)
    elif algorithm == "A*":
        path = a_search(tec, start_node, target_node)
    elif algorithm == "Branch & bound":
        path = branch_and_bound(tec, start_node, target_node)
    else:
        path = None

    result_text.delete(1.0, tk.END)
    result_text.insert(tk.END, f"Camino: {path}\n")
    draw_graph(tec, path)


# Para la interfaz gráfica usaremos Tkinter
# Crear ventana principal
root = tk.Tk()
root.title("Busqueda en Grafos")
root.resizable(False, False)

# Crear letrero, caja de texto y botón para la búsqueda
start_label = ttk.Label(root, text="Nodo inicial:")
start_var = tk.StringVar()
start_entry = ttk.Entry(root, textvariable=start_var)

target_label = ttk.Label(root, text="Nodo objetivo:")
target_var = tk.StringVar()
target_entry = ttk.Entry(root, textvariable=target_var)

algorithm_label = ttk.Label(root, text="Algoritmo:")
algorithm_var = tk.StringVar()
algorithm_combobox = ttk.Combobox(root, textvariable=algorithm_var, state="readonly")
algorithm_combobox["values"] = ["BFS", "DFS", "Greedy", "Dijkstra", "A*", "Branch & bound"]
algorithm_combobox.current(0)

search_button = ttk.Button(root, text="Buscar", command=run_search)

result_text = tk.Text(root, wrap=tk.WORD, width=50, height=10)

# Crear el gráfico y el lienzo
figure = plt.Figure(figsize=(5, 5), dpi=100)
canvas = FigureCanvasTkAgg(figure, root)
canvas.get_tk_widget().grid(row=0, column=2, rowspan=5)

# Colocar widgets en la ventana
start_label.grid(row=0, column=0, sticky="w")
start_entry.grid(row=0, column=1, sticky="ew")

target_label.grid(row=1, column=0, sticky="w")
target_entry.grid(row=1, column=1, sticky="ew")

algorithm_label.grid(row=2, column=0, sticky="w")
algorithm_combobox.grid(row=2, column=1, sticky="ew")

search_button.grid(row=3, column=0, columnspan=2)

result_text.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

# Dibujar el grafo inicial
draw_graph(tec)

# Iniciar el bucle principal de eventos
root.mainloop()
