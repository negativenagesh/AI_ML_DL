from manimlib import *
import networkx as nx
import numpy as np

class PreferentialAttachmentGrowth(Scene):
    def construct(self):
        # Generate a small starting network
        initial_nodes = 5
        total_nodes = 100
        m_edges = 2  # New edges per node

        G = nx.barabasi_albert_graph(total_nodes, m_edges)
        pos = nx.spring_layout(G, seed=42)

        # Draw initial nodes
        node_circles = {}
        for node in G.nodes():
            x, y = pos[node]
            circle = Circle(radius=0.1).move_to([x * 3, y * 3, 0])  # Scaling for better visualization
            circle.set_fill(BLUE, opacity=0.8)
            node_circles[node] = circle
            self.add(circle)

        self.wait(1)

        # Animate adding new nodes and edges
        for new_node in range(initial_nodes, total_nodes):
            # Position new node
            x, y = np.random.uniform(-2, 2), np.random.uniform(-2, 2)
            pos[new_node] = (x, y)
            new_circle = Circle(radius=0.1).move_to([x * 3, y * 3, 0])
            new_circle.set_fill(RED, opacity=0.8)
            node_circles[new_node] = new_circle

            # Animate node appearance
            self.play(GrowFromCenter(new_circle), run_time=0.3)

            # Connect edges based on preferential attachment
            targets = list(G.nodes())[:new_node]
            degrees = np.array([G.degree(t) for t in targets])
            probabilities = degrees / degrees.sum()

            connected_nodes = np.random.choice(targets, size=m_edges, replace=False, p=probabilities)

            # Animate connecting edges
            for target in connected_nodes:
                G.add_edge(new_node, target)
                edge = Line(node_circles[new_node].get_center(), node_circles[target].get_center(), color=WHITE)
                self.play(ShowCreation(edge), run_time=0.2)

        self.wait(2)
