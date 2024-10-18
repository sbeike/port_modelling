import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to check if two line segments intersect
def do_intersect(p1, p2, q1, q2):
    """ 
    Check if the line segment (p1, p2) intersects with (q1, q2)
    """
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if val == 0:
            return 0  # Collinear
        return 1 if val > 0 else 2  # Clockwise or Counterclockwise

    def on_segment(p, q, r):
        if min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1]):
            return True
        return False

    o1 = orientation(p1, p2, q1)
    o2 = orientation(p1, p2, q2)
    o3 = orientation(q1, q2, p1)
    o4 = orientation(q1, q2, p2)

    # General case
    if o1 != o2 and o3 != o4:
        return True

    # Special cases (collinear points)
    if o1 == 0 and on_segment(p1, q1, p2): return True
    if o2 == 0 and on_segment(p1, q2, p2): return True
    if o3 == 0 and on_segment(q1, p1, q2): return True
    if o4 == 0 and on_segment(q1, p2, q2): return True

    return False

# Function to check if a line between two points intersects any container
def is_feasible_arc(node1, node2, containers):
    """
    Check if the edge between node1 and node2 intersects any container.
    Containers are defined as rectangles with two diagonal corners.
    """
    for container in containers:
        (cx0, cy0), (cx1, cy1) = container  # Get the diagonal corners of the container

        # Define the edges of the container (rectangle)
        rectangle_edges = [
            ((cx0, cy0), (cx1, cy0)),  # Top edge
            ((cx0, cy1), (cx1, cy1)),  # Bottom edge
            ((cx0, cy0), (cx0, cy1)),  # Left edge
            ((cx1, cy0), (cx1, cy1))   # Right edge
        ]
        
        # Check if the edge between node1 and node2 intersects any rectangle edge
        for rect_edge in rectangle_edges:
            if do_intersect(node1, node2, rect_edge[0], rect_edge[1]):
                return False  # The edge intersects a container, so it's not feasible
    
    return True  # No intersections, the arc is feasible

# Load the image
image_path = './data/port_close_image.png'  # Replace with the correct path
img = cv2.imread(image_path)

# List of container corners (you can manually define or mark them using mouse events)
container_corners = [((447, 135), (467, 199)), ((468, 72), (477, 91)), ((470, 12), (479, 66)), ((426, 68), (435, 88)), ((420, 87), (432, 110)), ((440, 35), (451, 15)), ((432, 240), (467, 229)), ((437, 216), (470, 202)), ((447, 249), (464, 272)), ((431, 269), (461, 317)), ((438, 320), (456, 387)), ((381, 467), (422, 488)), ((375, 511), (411, 528)), ((370, 548), (410, 565))]

# List of node positions
nodes = [(362, 243), (410, 244), (443, 103), (476, 103), (456, 34), (397, 332), (373, 446), (367, 483),
         (363, 525), (356, 565), (351, 607), (339, 651), (328, 680), (329, 706), (322, 763), (373, 772),
         (300, 845), (422, 707), (427, 659), (427, 617), (429, 578), (429, 537), (429, 496), (441, 453)]

# Function to calculate the distance between two nodes
def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

# Find feasible arcs between adjacent nodes
feasible_arcs = []
for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if i != j:
            # Calculate the distance between the two nodes
            distance = calculate_distance(node1, node2)
            # Set a distance threshold for adjacent nodes (e.g., 300 pixels)
            if distance < 100:
                # Check if the arc is feasible (doesn't intersect any containers)
                if is_feasible_arc(node1, node2, container_corners):
                    feasible_arcs.append((node1, node2))  # Add the arc if it's feasible

# Now you have the feasible arcs (edges) for your graph
print(f"Feasible arcs: {feasible_arcs}")

# Optionally, draw the arcs on the image to visualize
for arc in feasible_arcs:
    cv2.line(img, arc[0], arc[1], (0, 255, 0), 2)

# Show the image with feasible arcs
cv2.imshow("Feasible Arcs", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
