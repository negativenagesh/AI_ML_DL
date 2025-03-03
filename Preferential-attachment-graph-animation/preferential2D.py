from manimlib import *
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

class PreferentialAttachmentDynamicGraph(Scene):
    def construct(self):
        # Set background color to darker shade
        self.camera.background_color = "#1A1A1A"
        
        # Initialize the graph
        initial_nodes = 5
        total_nodes = 200  # Increased number of nodes
        m_edges = 3  # New edges per node

        G = nx.empty_graph(initial_nodes)
        pos = nx.kamada_kawai_layout(G)  # More natural, free-flowing layout

        # Normalize positions to fit within Manim frame
        pos = self.normalize_positions(pos)

        # Draw initial nodes and edges
        node_circles = {}
        edge_lines = {}

        # Color scheme - gradient based on node degree
        color_low = "#3498DB"  # Light blue
        color_high = "#E74C3C"  # Red
        edge_color = "#AAAAAA"  # Light gray for edges

        for node in G.nodes():
            x, y, z = pos[node]  # Unpack 3D coordinates
            circle = Circle(radius=0.1, stroke_width=2).move_to([x, y, z])
            circle.set_fill(color_low, opacity=0.8)
            circle.set_stroke(WHITE, opacity=0.6, width=0.5)
            node_circles[node] = circle
            self.add(circle)

        self.wait(1)

        # Add new nodes dynamically
        for new_node in range(initial_nodes, total_nodes):
            G.add_node(new_node)

            # Connect preferentially based on degree
            targets = list(G.nodes())[:-1]
            degrees = np.array([G.degree(t) for t in targets])
            probabilities = degrees / degrees.sum() if degrees.sum() > 0 else np.ones(len(degrees)) / len(degrees)

            connected_nodes = np.random.choice(targets, size=m_edges, replace=False, p=probabilities)

            for target in connected_nodes:
                G.add_edge(new_node, target)

            # Update node positions using Kamada-Kawai layout and normalize
            pos = nx.kamada_kawai_layout(G)
            pos = self.normalize_positions(pos)

            # Add and animate new node
            x, y, z = pos[new_node]  # Unpack 3D coordinates
            node_degree = G.degree(new_node)
            node_size = 0.08 + 0.018 * node_degree
            
            # Determine color based on degree (from blue to red)
            max_degree = max(G.degree(n) for n in G.nodes())
            color_factor = min(1.0, node_degree / (max_degree * 0.7))  # Scale to get more color variation
            node_color = self.interpolate_color(color_low, color_high, color_factor)
            
            new_circle = Circle(radius=node_size, stroke_width=2).move_to([x, y, z])
            new_circle.set_fill(node_color, opacity=0.8)
            new_circle.set_stroke(WHITE, opacity=0.6, width=0.5)
            node_circles[new_node] = new_circle
            self.play(GrowFromCenter(new_circle), run_time=0.2)

            # Add dynamically updating edges
            for target in connected_nodes:
                def create_edge_func(source_circle, target_circle):
                    return lambda: Line(
                        source_circle.get_center(),
                        target_circle.get_center(),
                        color=edge_color,
                        stroke_width=0.8 + 0.3 * min(G.degree(new_node), G.degree(target)) / max_degree
                    )

                edge = always_redraw(create_edge_func(node_circles[new_node], node_circles[target]))
                edge_lines[(new_node, target)] = edge
                self.add(edge)

            # Animate node movements based on updated positions
            animations = []
            for node in G.nodes():
                x, y, z = pos[node]  # Unpack 3D coordinates
                new_position = np.array([x, y, z])
                node_degree = G.degree(node)
                node_size = 0.08 + 0.018 * node_degree
                
                # Update node color based on its current degree
                color_factor = min(1.0, node_degree / (max_degree * 0.7))
                node_color = self.interpolate_color(color_low, color_high, color_factor)
                
                animations.append(
                    node_circles[node].animate.move_to(new_position)
                                           .scale(node_size / node_circles[node].get_width())
                                           .set_fill(node_color)
                )

            # Play animation for node and edges together
            self.play(*animations, run_time=0.4)

        self.wait(2)

        # Plot the power-law degree distribution using matplotlib
        self.plot_degree_distribution(G)
    
    def interpolate_color(self, color1, color2, factor):
        """Interpolate between two hex colors"""
        c1 = np.array([int(color1[i:i+2], 16) for i in (1, 3, 5)])
        c2 = np.array([int(color2[i:i+2], 16) for i in (1, 3, 5)])
        c = c1 * (1-factor) + c2 * factor
        return rgb_to_hex(c.astype(int))
    
    def normalize_positions(self, pos):
        """Normalize positions to fit within Manim's frame and center the graph."""
        pos_array = np.array([pos[node][:2] for node in pos])  # Extract 2D coordinates
        min_x, max_x = np.min(pos_array[:, 0]), np.max(pos_array[:, 0])
        min_y, max_y = np.min(pos_array[:, 1]), np.max(pos_array[:, 1])
        
        # Find the center of the graph
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        
        # Calculate ranges
        x_range = max_x - min_x
        y_range = max_y - min_y
        max_range = max(x_range, y_range) if max(x_range, y_range) > 0 else 1.0
        
        # Scale to fit within [-6, 6] x [-6, 6] with some margin
        scale = 5.0 / max_range  # Slightly smaller scale to ensure it fits
        
        normalized_pos = {}
        for node in pos:
            x, y = pos[node][:2]  # Take only x, y from the original 2D pos
            # Center and scale
            x = (x - center_x) * scale
            y = (y - center_y) * scale
            normalized_pos[node] = [x, y, 0]  # Add z=0 for 3D compatibility
        
        return normalized_pos

    def plot_degree_distribution(self, G):
        # Calculate degree frequency
        degrees = [G.degree(n) for n in G.nodes()]
        degree_count = Counter(degrees)
        deg, cnt = zip(*sorted(degree_count.items()))

        # Plot degree distribution on a log-log scale with improved styling
        plt.figure(figsize=(10, 8), facecolor='#F8F9F9')
        plt.plot(deg, cnt, color='#2980B9', marker='o', markersize=6, 
                 linestyle='-', linewidth=2, markerfacecolor='#3498DB', markeredgecolor='#1A5276')
        plt.xscale('log')
        plt.yscale('log')
        plt.title("Power-Law Degree Distribution", fontsize=18, fontweight='bold')
        plt.xlabel("Node Degree (log scale)", fontsize=14)
        plt.ylabel("Frequency (log scale)", fontsize=14)
        plt.grid(True, which="both", linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        plt.show()

def rgb_to_hex(rgb):
    """Convert RGB values to hex color code"""
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"

# Run with: manimgl preferential2D.py PreferentialAttachmentDynamicGraph -o preferential_attachment_video