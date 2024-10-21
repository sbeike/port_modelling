import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import csv

class ContainerMapping:
    def __init__(self, image_path, length_standard_container):
        self.image_path = image_path
        self.length_standard_container = length_standard_container
        self.current_container = []
        self.container_corners = []
        self.reference_container_list = []
        self.container_length_pixels = 0
        self.nodes = []
        self.feasible_arcs = {}
        self.neighbor_nodes = {}

    def calculate_distance(self, point1, point2):
        return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def mark_container(self, event, x, y, flags, param):
            
        # When left mouse button is clicked, mark a corner
        if event == cv2.EVENT_LBUTTONDOWN:
            self.current_container.append((x, y))
            if len(self.current_container) == 2:
                # When two points are marked, save the container
                self.container_corners.append(tuple(self.current_container))
                self.current_container = []  # Reset for next container
    
    def reference_container(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.reference_container_list.append((x, y))
            if len(self.reference_container_list) == 2:
                self.container_length_pixels = self.calculate_distance(
                    self.reference_container_list[0], self.reference_container_list[1]
                )

    def click_event(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.nodes.append((x, y))
            cv2.circle(self.img, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Manual Node Placement", self.img)

    def direction_edge(self, point1, point2):
        delta_x = point2[0] - point1[0]
        delta_y = point2[1] - point1[1]
        angle_degrees = math.degrees(math.atan2(delta_y, delta_x)) % 360
        if 45 <= angle_degrees < 135:
            return "North"
        elif 135 <= angle_degrees < 225:
            return "West"
        elif 225 <= angle_degrees < 315:
            return "South"
        else:
            return "East"

    def do_intersect(self, p1, p2, q1, q2):
        def orientation(p, q, r):
            val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
            return 0 if val == 0 else 1 if val > 0 else 2

        def on_segment(p, q, r):
            return min(p[0], r[0]) <= q[0] <= max(p[0], r[0]) and min(p[1], r[1]) <= q[1] <= max(p[1], r[1])

        o1, o2 = orientation(p1, p2, q1), orientation(p1, p2, q2)
        o3, o4 = orientation(q1, q2, p1), orientation(q1, q2, p2)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(p1, q1, p2): return True
        if o2 == 0 and on_segment(p1, q2, p2): return True
        if o3 == 0 and on_segment(q1, p1, q2): return True
        if o4 == 0 and on_segment(q1, p2, q2): return True

        return False

    def is_feasible_arc(self, node1, node2):
        for container in self.container_corners:
            rect_edges = [
                (container[0], (container[0][0], container[1][1])),
                ((container[0][0], container[1][1]), container[1]),
                (container[1], (container[1][0], container[0][1])),
                ((container[1][0], container[0][1]), container[0])
            ]
            if any(self.do_intersect(node1, node2, edge[0], edge[1]) for edge in rect_edges):
                return False
        return True

    def compute_feasible_arcs(self):
        for i, node1 in enumerate(self.nodes):
            for j, node2 in enumerate(self.nodes):
                if i != j and i < j and self.calculate_distance(node1, node2) < 300:
                    if self.is_feasible_arc(node1, node2):
                        direction = self.direction_edge(node1, node2)
                        if i not in self.feasible_arcs:
                            self.feasible_arcs[i] = {direction: (node1, node2)}
                        elif direction not in self.feasible_arcs[i] or self.calculate_distance(node1, node2) < self.calculate_distance(
                                *self.feasible_arcs[i][direction]):
                            self.feasible_arcs[i][direction] = (node1, node2)

    def compute_neighbor_nodes(self):
        for node, directions in self.feasible_arcs.items():
            for direction, arc in directions.items():
                if node not in self.neighbor_nodes:
                    self.neighbor_nodes[node] = [self.nodes.index(arc[1])]
                else:
                    self.neighbor_nodes[node].append(self.nodes.index(arc[1]))

    def scale_nodes(self):
        first_node = self.nodes[0]
        length_one_pixel_meter = self.length_standard_container / self.container_length_pixels
        self.nodes = [(node[0] - first_node[0], node[1] - first_node[1]) for node in self.nodes]
        self.nodes = [(node[0] * length_one_pixel_meter, node[1] * length_one_pixel_meter) for node in self.nodes]

    def save_to_csv(self):
        with open('data/nodes.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Node ID', 'X', 'Y'])
            for i, node in enumerate(self.nodes):
                writer.writerow([i, node[0], node[1]])

        with open('data/neighbor_nodes.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Node ID', 'Neighbor IDs'])
            for node, neighbors in self.neighbor_nodes.items():
                writer.writerow([node, neighbors])
                
    def draw_lines(self):
        for node, directions in self.feasible_arcs.items():
            for direction, arc in directions.items():
                cv2.line(self.img, arc[0], arc[1], (0, 0, 255), 2)

    def process_image(self):
        self.img = cv2.imread(self.image_path)
        self.node_image = np.copy(self.img)

        # Mark reference container
        cv2.namedWindow('Reference Container')
        cv2.setMouseCallback('Reference Container', self.reference_container)
        while len(self.reference_container_list) < 2:
            cv2.imshow('Reference Container', self.img)
            if (cv2.waitKey(1) & 0xFF == ord('q')) or len(self.reference_container_list) == 2:
                break
        cv2.destroyAllWindows()

        # Mark containers
        cv2.namedWindow('Mark Containers')
        cv2.setMouseCallback('Mark Containers', self.mark_container)
        while True:
            cv2.imshow('Mark Containers', self.img)
            for corners in self.container_corners:
                cv2.rectangle(self.img, corners[0], corners[1], (0, 255, 0), 2)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

        # Place nodes manually
        cv2.imshow("Manual Node Placement", self.img)
        cv2.setMouseCallback("Manual Node Placement", self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        self.compute_feasible_arcs()
        self.compute_neighbor_nodes()
        
        self.draw_lines()
        
        # Show the image with feasible arcs
        cv2.imshow("Feasible Arcs", self.img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        
        self.scale_nodes()
        self.save_to_csv()

def main():
    image_path = './data/port_close_image.png'
    length_standard_container = 12.19  # Example length in meters
    container_mapping = ContainerMapping(image_path, length_standard_container)
    container_mapping.process_image()

if __name__ == "__main__":
    main()
