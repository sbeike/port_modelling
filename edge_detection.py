import cv2
import numpy as np
import matplotlib.pyplot as plt


import cv2

Length_standard_container = 12.19

# Initialize variables to store container corners
container_corners = []
current_container = []

container_length_pixels = 0
reference_container_list = []

def calculate_distance(point1, point2):
    # Use Euclidean distance formula to calculate the distance between two points
    distance = np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return distance

# Mouse callback function to mark corners of container piles
def mark_container(event, x, y, flags, param):
    global current_container, container_corners
    
    # When left mouse button is clicked, mark a corner
    if event == cv2.EVENT_LBUTTONDOWN:
        current_container.append((x, y))
        if len(current_container) == 2:
            # When two points are marked, save the container
            container_corners.append(tuple(current_container))
            current_container = []  # Reset for next container
            
    
def reference_container(event, x, y, flags, param):
    global container_length_pixels
    
    # When left mouse button is clicked, mark a corner
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Reference point: ({x}, {y})")
        reference_container_list.append((x, y))
        if len(reference_container_list) == 2:
            container_length_pixels = calculate_distance(reference_container_list[0], reference_container_list[1])

# Load the image
image_path = './data/port_close_image.png'  # Replace with the correct path
img = cv2.imread(image_path)
marked_img = img.copy()

# Create a window and set the mouse callback function
cv2.namedWindow('Reference Container')
cv2.setMouseCallback('Reference Container', reference_container)

# Loop to display the image and allow marking
while True:
    # Display the image
    cv2.imshow('Reference Container', marked_img)

    
    # Wait for 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, close the window
cv2.destroyAllWindows()


# Create a window and set the mouse callback function
cv2.namedWindow('Mark Containers')
cv2.setMouseCallback('Mark Containers', mark_container)

# Loop to display the image and allow marking
while True:
    # Display the image
    cv2.imshow('Mark Containers', marked_img)

    # Draw the marked containers so far
    for corners in container_corners:
        # Draw a rectangle using the two diagonal corners
        cv2.rectangle(marked_img, corners[0], corners[1], (0, 255, 0), 2)

    # Wait for 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, close the window
cv2.destroyAllWindows()

# Output the marked container corners
print(f"Marked container corners: {container_corners}")


# Load the image
image_path = './data/port_close_image.png'  # Replace with the correct path
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Create a copy of the original image for plotting nodes
node_image = np.copy(marked_img)

# List to store node coordinates
nodes = []

# Mouse callback function to capture node positions
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Add node position to list
        nodes.append((x, y))
        # Draw a red circle at the clicked position
        cv2.circle(node_image, (x, y), 5, (0, 0, 255), -1)
        # Display updated image
        cv2.imshow("Manual Node Placement", node_image)

# Display the image and set up mouse event handler
cv2.imshow("Manual Node Placement", node_image)
cv2.setMouseCallback("Manual Node Placement", click_event)

# Wait for the user to press any key, and then close the window
cv2.waitKey(0)
cv2.destroyAllWindows()

# Print the node coordinates (optional, for debugging)
print("Node coordinates:", nodes)

# Use nodes in further graph construction
# Now you can use the manually placed nodes to create your graph
# For example, construct the edges based on distance between the nodes or other logic








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

# Example list of containers (replace with actual container coordinates and sizes)

import math

def direction_edge(point1, point2):
    """
    Determines the cardinal direction (North, South, East, West) of the edge
    between two points.

    Args:
        point1: Tuple (x1, y1) representing the first point.
        point2: Tuple (x2, y2) representing the second point.

    Returns:
        A string representing the direction (e.g., "North", "South", "East", "West").
    """
    x1, y1 = point1
    x2, y2 = point2
    
    # Calculate the angle in radians between the two points
    delta_x = x2 - x1
    delta_y = y2 - y1
    angle = math.atan2(delta_y, delta_x)  # atan2 handles all quadrants and returns radians
    
    # Convert the angle to degrees for easier comparison
    angle_degrees = math.degrees(angle)
    
    # Normalize the angle to be between 0 and 360 degrees
    if angle_degrees < 0:
        angle_degrees += 360
    
    # Determine the direction based on the angle
    if 45 <= angle_degrees < 135:
        return "North"
    elif 135 <= angle_degrees < 225:
        return "West"
    elif 225 <= angle_degrees < 315:
        return "South"
    else:
        return "East"

# Example usage:
p1 = (300, 400)
p2 = (400, 300)

direction = direction_edge(p1, p2)
print(f"Direction: {direction}")


# List of node positions (you already have these)

# Find feasible arcs between adjacent nodes
feasible_arcs = {}

for i, node1 in enumerate(nodes):
    for j, node2 in enumerate(nodes):
        if i != j and i < j:
            # Calculate the distance between the two nodes
            distance = calculate_distance(node1, node2)
            # You can set a distance threshold for adjacent nodes (e.g., 200 pixels)
            if distance < 300:  # You can adjust the threshold
                if is_feasible_arc(node1, node2, container_corners):
                    direction_of_edge = direction_edge(node1, node2)
                    if i not in feasible_arcs:
                        
                        feasible_arcs[i] = {direction_of_edge: ((node1, node2))}
                    else:
                        if direction_of_edge not in feasible_arcs[i]:
                            feasible_arcs[i][direction_of_edge] = ((node1, node2))
                        else:
                            if calculate_distance(node1, node2) < calculate_distance(feasible_arcs[i][direction_of_edge][0], feasible_arcs[i][direction_of_edge][1]):
                                feasible_arcs[i][direction_of_edge] = ((node1, node2))

# Now you have the feasible arcs (edges) for your graph
print(f"Feasible arcs: {feasible_arcs}")

# Optionally, draw the arcs on the image to visualize
for node, directions in feasible_arcs.items():
    for direction, arc in directions.items():
        cv2.line(node_image, arc[0], arc[1], (0, 0, 255), 2)

neighbor_nodes = {}
        
for node, direction in feasible_arcs.items():
    for dir, arc in direction.items():
        if node not in neighbor_nodes:
            neighbor_nodes[node] = [nodes.index(arc[1])]
        else:
            neighbor_nodes[node].append(nodes.index(arc[1]))


print("Neighbor nodes:", neighbor_nodes)

#Save neighbor nodes and nodes as csv files
import csv

#Set first node to coordinates 0,0 and rest to relative to first node
first_node = nodes[0]
nodes = [(node[0] - first_node[0], node[1] - first_node[1]) for node in nodes]


length_one_pixel_meter = Length_standard_container / container_length_pixels

#Scale alle nodes to meters
nodes = [(node[0] * length_one_pixel_meter, node[1] * length_one_pixel_meter) for node in nodes]


# Save the nodes to a CSV file
with open('data/nodes.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Node ID', 'X', 'Y'])
    for i, node in enumerate(nodes):
        writer.writerow([i, node[0], node[1]])
    
    
# Save the neighbor nodes to a CSV file
with open('data/neighbor_nodes.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Node ID', 'Neighbor IDs'])
    for node, neighbors in neighbor_nodes.items():
        writer.writerow([node, neighbors])
        

# Show the image with feasible arcs
cv2.imshow("Feasible Arcs", node_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
