import numpy as np
import networkx as nx
import random
import math
import csv
import os
import pandas as pd
import time


class HospitalModel:
    def __init__(self, num_hospitals, mean_daily_patients, std_daily_patients, mean_visits, simulation_days,
                 stay_probability):
        self.num_hospitals = num_hospitals  # number of hospitals (nodes)
        self.mean_visits = mean_visits  # maximum number of referrals
        self.simulation_days = simulation_days  # number of simulation days
        self.mean_daily_patients = mean_daily_patients  # mean daily new patients
        self.std_daily_patients = std_daily_patients  # std of daily new patients
        self.stay_probability = stay_probability  # probability of continuing hospitalization
        self.G = self.generate_hospital_network(node_data_file, edge_data_file)

    # (1) Generate hospital network
    def generate_hospital_network(self, node_data_file, edge_data_file):
        node_data = pd.read_csv(node_data_file, header=0)
        edge_data = pd.read_csv(edge_data_file, header=0)

        G = nx.DiGraph()

        # Add nodes with attributes
        for _, row in node_data.iterrows():
            G.add_node(
                row['Node'],
                grade=int(row['grade']),
                cure_rate=float(row['cure_rate']),
                location=row['location'],
                visit_limit=int(row['visit_limit']),
                VP_initial=float(row['VP_initial']),
                initial_distance=float(row['initial_distance']),
                VP_referral=float(row['VP_referral']),
                referral_distance=float(row['referral_distance']),
                LP=float(row['LP']),
                cure_LP=float(row['cure_LP']),
                free_capacity=int(row['free_capacity']),
            )

        # Add edges with attributes
        for _, row in edge_data.iterrows():
            G.add_edge(
                row['Source'],
                row['Target'],
                weight=float(row['weight']),
                distance=float(row['distance']),
                preference=float(row['preference']),
            )

        for node in G.nodes:
            G.nodes[node]['location'] = G.nodes[node].get('location')
            G.nodes[node]['cure_rate'] = G.nodes[node].get('cure_rate')
        for u, v in G.edges:
            G[u][v]['new_weight'] = 0

        return G

    # (2) Generate number of patients per day
    def daily_patient_counts(self):
        daily_patient_count = np.random.normal(self.mean_daily_patients, self.std_daily_patients, self.simulation_days)
        daily_patient_count = [max(0, int(count)) for count in daily_patient_count]
        return daily_patient_count

    # (3) Referral decision logic based on weight, distance, and hospital grade
    def choose_next_hospital(self, current_node):
        all_neighbors = [target for _, target, data in self.G.out_edges(current_node, data=True)
                         if data.get('weight', 0) > 0]
        referral_distance = 0
        neighbors = [
            neighbor for neighbor in all_neighbors
            if self.G.nodes[neighbor].get('free_capacity', 0) > 0
        ]

        A = 0.3  # historical weight coefficient
        B = 0.3  # distance coefficient
        C = 1 - A - B  # grade coefficient

        if neighbors:
            try:
                weights = np.array([
                    self.G.edges[current_node, neighbor]['weight'] for neighbor in neighbors
                ])
                distances = np.array([
                    1 / self.G.edges[current_node, neighbor]['distance'] for neighbor in neighbors
                ])
                grades = np.array([
                    self.G.edges[current_node, neighbor]['preference'] for neighbor in neighbors
                ])

                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.zeros_like(weights)
                distances = distances / np.sum(distances) if np.sum(distances) > 0 else np.zeros_like(distances)
                grades = grades / np.sum(grades) if np.sum(grades) > 0 else np.zeros_like(grades)

                mix = A * weights + B * distances + C * grades
                next_hospital = np.random.choice(neighbors, p=mix / np.sum(mix))

                referral_distance = self.G.edges[current_node, next_hospital]['distance']
            except Exception as e:
                print(f"Error: {e}, falling back to random selection")
                next_hospital = np.random.choice(list(self.G.nodes))
                referral_distance = self.G.edges[current_node, next_hospital].get('distance', float('inf'))
        else:
            valid_nodes = [node for node in self.G.nodes if
                           self.G.nodes[node].get('free_capacity', 0) > 0 and node != current_node]
            if valid_nodes:
                next_hospital = np.random.choice(valid_nodes)
                if self.G.has_edge(current_node, next_hospital):
                    referral_distance = self.G.edges[current_node, next_hospital].get('distance', None)
                else:
                    print('Distance error!')
            else:
                print("Warning: No hospital has free capacity, cannot refer")
                next_hospital = current_node
                referral_distance = 0

        return next_hospital, referral_distance

    # (4) Assign patient to the nearest hospital
    def min_distance(self):
        min_distance = float('inf')
        nearest_hospital = None

        L = 200
        x_p = random.uniform(0, L)
        y_p = random.uniform(0, L)

        for node in self.G.nodes:
            hospital_location = self.G.nodes[node].get('location')
            if isinstance(hospital_location, str):
                try:
                    hospital_location = tuple(map(float, hospital_location.strip("()").split(",")))
                except ValueError:
                    continue

            if not isinstance(hospital_location, (tuple, list)) or len(hospital_location) != 2:
                continue
            x_h, y_h = hospital_location
            if hospital_location:
                distance = round(math.sqrt((x_p - x_h) ** 2 + (y_p - y_h) ** 2), 3)
                if distance < min_distance:
                    min_distance = distance
                    nearest_hospital = node

        return nearest_hospital, min_distance

    # Simulate a single patient's visit process
    def process_patient_visit(self):
        nearest_hospital, min_distance = self.min_distance()
        current_node = nearest_hospital

        j = 0
        num_referrals = 0
        num_distance = min_distance
        R_distance = 0
        LP = VP_initial = VP_referral = cure_LP = 0

        while j <= self.mean_visits:
            if self.G.nodes[current_node]['free_capacity'] > 0:
                self.G.nodes[current_node]['free_capacity'] -= 1
                if np.random.rand() < float(self.G.nodes[current_node]['cure_rate']):
                    if j == 0:
                        VP_initial = 1
                        self.G.nodes[current_node]['VP_initial'] += 1
                        num_distance += R_distance
                        self.G.nodes[current_node]['initial_distance'] += num_distance
                    else:
                        VP_referral = 1
                        self.G.nodes[current_node]['VP_referral'] += 1
                        num_distance += R_distance
                        self.G.nodes[current_node]['referral_distance'] += num_distance
                    num_referrals = j
                    break
                else:
                    if np.random.rand() < self.stay_probability:
                        self.G.nodes[current_node]['cure_LP'] += 1
                        cure_LP += 1
                        continue

            j += 1
            if j > self.mean_visits:
                LP = 1
                self.G.nodes[current_node]['LP'] += 1
                num_referrals = self.mean_visits
                break

            next_node, referral_distance = self.choose_next_hospital(current_node)
            if current_node == next_node:
                self.G.nodes[current_node]['LP'] += 1
                num_referrals += j
                LP += 1
                R_distance = referral_distance
            else:
                R_distance = referral_distance
                num_referrals += 1
                self.G.out_edges[current_node, next_node]['new_weight'] += 1
                current_node = next_node

        return num_referrals, num_distance, LP, VP_initial, VP_referral, cure_LP

    # Simulate all patients in a single day
    def simulate_single_day(self, daily_patient_count):
        num_R_daily = 0
        num_LP_daily = 0
        num_cure_LP_daily = 0
        num_VP_initial_daily = 0
        num_VP_referral_daily = 0
        num_distance_daily = 0

        for patient in range(daily_patient_count):
            num_referrals, num_distance, LP, VP_initial, VP_referral, cure_LP = self.process_patient_visit()
            if math.isinf(num_distance) or math.isnan(num_distance):
                print(f"Warning: Invalid distance for patient {patient}: {num_distance}, skipping")
                num_distance = 0
            num_R_daily += num_referrals
            num_LP_daily += LP
            num_cure_LP_daily += cure_LP
            num_VP_initial_daily += VP_initial
            num_VP_referral_daily += VP_referral
            num_distance_daily += int(num_distance)

    # Reset hospital capacity each day
    def reset_hospital_capacity(self):
        for node in self.G.nodes:
            self.G.nodes[node]['free_capacity'] = self.G.nodes[node]['visit_limit'] - self.G.nodes[node]['cure_LP']

    # Simulate all days
    def simulate_patient_visits(self):
        num_LP_daily_dict = {}
        daily_patient_counts = self.daily_patient_counts()
        TP = sum(daily_patient_counts)

        for day in range(1, self.simulation_days + 1):
            daily_patient_count = random.choice(daily_patient_counts)
            daily_patient_counts.remove(daily_patient_count)
            delta = daily_patient_count
            if day > 1:
                delta += num_LP_daily_dict.get(day - 1, 0)
            self.reset_hospital_capacity()
            self.simulate_single_day(daily_patient_count)

    # Save graph to CSV
    def save_graph_to_csv(self, node_file, edge_file):
        os.makedirs(os.path.dirname(node_file), exist_ok=True)
        os.makedirs(os.path.dirname(edge_file), exist_ok=True)

        with open(node_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ['Node'] + list(next(iter(self.G.nodes(data=True)))[1].keys())
            writer.writerow(header)
            for node, attrs in self.G.nodes(data=True):
                row = [node] + list(attrs.values())
                writer.writerow(row)

        with open(edge_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ['Source', 'Target'] + list(next(iter(self.G.edges(data=True)))[2].keys())
            writer.writerow(header)
            for u, v, attrs in self.G.edges(data=True):
                row = [u, v] + list(attrs.values())
                writer.writerow(row)


if __name__ == '__main__':
    start = time.time()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'Starting simulation at {start_time}')
    node_data_file = "../original_data/0513nodes(0.01).csv"
    edge_data_file = "../original_data/edges(T500_network_completed).csv"
    node_file = "../simulation_results/model_results(0.01)(10)/nodes.csv"
    edge_file = "../simulation_results/model_results(0.01)(10)/edges.csv"
    model = HospitalModel(num_hospitals=200, mean_daily_patients=41000, std_daily_patients=200, mean_visits=4,
                          simulation_days=500, stay_probability=0)

    model.simulate_patient_visits()
    model.save_graph_to_csv(node_file, edge_file)
    end = time.time()
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'Simulation ended at {end_time}')
    print("Total time used: {} hours".format((end - start) / 3600))
    print("Working directory:", os.getcwd())
