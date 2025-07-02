import numpy as np
import networkx as nx
import random
import math
import csv
import os
import pandas as pd
import time


class HospitalModel:
    def __init__(self, num_hospitals, mean_daily_patients, std_daily_patients, mean_visits, simulation_days, stay_probability):
        self.num_hospitals = num_hospitals  # Number of nodes
        self.mean_visits = mean_visits  # Maximum number of referrals
        self.simulation_days = simulation_days  # Number of simulation days
        self.mean_daily_patients = mean_daily_patients  # Mean of new patients per time step
        self.std_daily_patients = std_daily_patients  # Std dev of new patients per time step
        self.stay_probability = stay_probability  # Probability of continued hospitalization
        self.G = self.generate_hospital_network(node_data_file, edge_data_file)

    # (1) Generate hospital network
    def generate_hospital_network(self, node_data_file, edge_data_file):
        # Load network data
        node_data = pd.read_csv(node_data_file, header=0)
        edge_data = pd.read_csv(edge_data_file, header=0)

        # Initialize directed weighted graph
        G = nx.DiGraph()

        # Add nodes and their attributes (converted to appropriate data types)
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

        # Add edges and their attributes (converted to appropriate data types)
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

        return G

    # (2) Generate daily patient counts
    def daily_patient_counts(self):
        daily_patient_count = np.random.normal(self.mean_daily_patients, self.std_daily_patients, self.simulation_days)
        daily_patient_count = [max(0, int(count)) for count in daily_patient_count]
        return daily_patient_count

    # (3) Referral decision: based on edge weight, distance, and hospital grade (only if capacity exists)
    def choose_next_hospital(self, current_node):
        all_neighbors = [target for _, target in self.G.out_edges(current_node)]
        referral_distance = 0
        no_free_capacity = 0

        neighbors = [
            neighbor for neighbor in all_neighbors
            if self.G.nodes[neighbor].get('free_capacity', 0) > 0
        ]

        A = 0.3  # Weight for historical referrals
        B = 0.3  # Weight for distance
        C = 1 - A - B  # Weight for hospital grade

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
                print(f"Error: {e}, defaulting to random hospital")
                next_hospital = np.random.choice(list(self.G.nodes))
                referral_distance = self.G.edges[current_node, next_hospital].get('distance', float('inf'))
        else:
            no_free_capacity += 1
            next_hospital = current_node
            referral_distance = 0

        return next_hospital, referral_distance

    # (5) Patient chooses the nearest hospital
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

    # Simulate one patient's visit process
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
                    if num_referrals == j:
                        num_referrals = j
                    else:
                        print('Logic error: num_referrals does not match j!')
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
                num_referrals += 1
                LP += 1
                R_distance = referral_distance
                break
            else:
                R_distance = referral_distance
                num_referrals += 1
                self.G.out_edges[current_node, next_node]['weight'] += 1
                current_node = next_node

        return num_referrals, num_distance, LP, VP_initial, VP_referral, cure_LP

    # Simulate one day's patients
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

        daily_FC_dict = {node: self.G.nodes[node]['free_capacity'] for node in self.G.nodes}
        FC_daily = sum(daily_FC_dict.values())

        return num_R_daily, num_LP_daily, num_cure_LP_daily, num_VP_initial_daily, num_VP_referral_daily, num_distance_daily, FC_daily
    # Hospital bed capacity reset
    def reset_hospital_capacity(self):
        for node in self.G.nodes:
            self.G.nodes[node]['free_capacity'] = self.G.nodes[node]['visit_limit'] - self.G.nodes[node]['cure_LP']

    # Simulate the patient visit process over the entire time span (daily iteration)
    def simulate_patient_visits(self):
        # Daily totals of R, LP, VP, CLP, FC
        num_R_daily_dict = {}
        num_LP_daily_dict = {}
        num_cure_LP_daily_dict = {}
        num_VP_initial_daily_dict = {}
        num_VP_referral_daily_dict = {}
        num_distance_daily_dict = {}
        FC_daily_dict = {}

        # Daily records for each hospital of R, LP, VP, CLP, FC
        hospital_daily_stats = {
            node: {
                "LP": {}, "VP_initial": {}, "initial_distance": {}, "VP_referral": {},
                "referral_distance": {}, "cure_LP": {}, "visit_limit": {}, "free_capacity": {}
            } for node in self.G.nodes
        }

        delta_dict = {}  # Actual patient count per day (δ)
        daily_patient_count_dict = {}  # Expected (theoretical) patient count per day
        daily_patient_counts = self.daily_patient_counts()  # Generate patient count list per day
        TP = sum(daily_patient_counts)  # Total number of patients

        for day in range(1, self.simulation_days + 1):
            daily_patient_count = random.choice(daily_patient_counts)
            daily_patient_counts.remove(daily_patient_count)
            delta = daily_patient_count
            if day > 1:
                delta += num_LP_daily_dict.get(day - 1, 0)

            # Simulate for the day
            num_R_daily, num_LP_daily, num_cure_LP_daily, num_VP_initial_daily, num_VP_referral_daily, num_distance_daily, FC_daily = self.simulate_single_day(delta)

            # Record daily statistics
            num_R_daily_dict[day] = num_R_daily
            num_LP_daily_dict[day] = num_LP_daily
            num_cure_LP_daily_dict[day] = num_cure_LP_daily
            num_VP_initial_daily_dict[day] = num_VP_initial_daily
            num_VP_referral_daily_dict[day] = num_VP_referral_daily
            num_distance_daily_dict[day] = num_distance_daily

            delta_dict[day] = delta
            daily_patient_count_dict[day] = daily_patient_count
            FC_daily_dict[day] = FC_daily

            # Record per-hospital statistics
            for node in self.G.nodes:
                hospital_daily_stats[node]["LP"][day] = self.G.nodes[node]['LP']
                hospital_daily_stats[node]["VP_initial"][day] = self.G.nodes[node]['VP_initial']
                hospital_daily_stats[node]["initial_distance"][day] = self.G.nodes[node]['initial_distance']
                hospital_daily_stats[node]["VP_referral"][day] = self.G.nodes[node]['VP_referral']
                hospital_daily_stats[node]["referral_distance"][day] = self.G.nodes[node]['referral_distance']
                hospital_daily_stats[node]["cure_LP"][day] = self.G.nodes[node]['cure_LP']
                hospital_daily_stats[node]["visit_limit"][day] = self.G.nodes[node]['visit_limit']
                hospital_daily_stats[node]["free_capacity"][day] = self.G.nodes[node]['free_capacity']

            self.reset_hospital_capacity()

        return {
            "num_R_daily_dict": num_R_daily_dict,
            "num_LP_daily_dict": num_LP_daily_dict,
            "num_cure_LP_daily_dict": num_cure_LP_daily_dict,
            "num_VP_initial_daily_dict": num_VP_initial_daily_dict,
            "num_VP_referral_daily_dict": num_VP_referral_daily_dict,
            "num_distance_daily_dict": num_distance_daily_dict,
            "FC_daily_dict": FC_daily_dict,
            "delta_dict": delta_dict,
            "daily_patient_count_dict": daily_patient_count_dict,
            "hospital_daily_stats": hospital_daily_stats,
            "TP": TP
        }

    # Save node and edge attributes to CSV
    def save_graph_to_csv(self, node_file, edge_file):
        os.makedirs(os.path.dirname(node_file), exist_ok=True)
        os.makedirs(os.path.dirname(edge_file), exist_ok=True)

        # Save node attributes
        with open(node_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ['Node'] + list(next(iter(self.G.nodes(data=True)))[1].keys())
            writer.writerow(header)
            for node, attrs in self.G.nodes(data=True):
                row = [node] + list(attrs.values())
                writer.writerow(row)

        # Save edge attributes
        with open(edge_file, mode='a', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            header = ['Source', 'Target'] + list(next(iter(self.G.edges(data=True)))[2].keys())
            writer.writerow(header)
            for u, v, attrs in self.G.edges(data=True):
                row = [u, v] + list(attrs.values())
                writer.writerow(row)

    # Save daily metrics: R, LP, VP, cLP, distance, FC, delta, daily patient count
    def save_results_to_csv(self, results, output_dir):
        os.makedirs(output_dir, exist_ok=True)

        daily_data = {
            "Day": list(results["num_R_daily_dict"].keys()),
            "R": list(results["num_R_daily_dict"].values()),
            "LP": list(results["num_LP_daily_dict"].values()),
            "cure_LP": list(results["num_cure_LP_daily_dict"].values()),
            "VP_initial": list(results["num_VP_initial_daily_dict"].values()),
            "VP_referral": list(results["num_VP_referral_daily_dict"].values()),
            "Distance": list(results["num_distance_daily_dict"].values()),
            "FC": list(results["FC_daily_dict"].values()),
            "Delta": list(results["delta_dict"].values()),
            "Daily Patient Count": list(results["daily_patient_count_dict"].values()),
        }
        daily_df = pd.DataFrame(daily_data)
        daily_file = f"{output_dir}/daily_statistics.csv"
        daily_df.to_csv(daily_file, mode='a', header=not os.path.exists(daily_file), index=False)

        hospital_stats = results.get("hospital_daily_stats", {})
        hospital_frames = []
        for hospital, stats in hospital_stats.items():
            for metric, day_data in stats.items():
                df = pd.DataFrame.from_dict(day_data, orient='index', columns=[metric])
                df.index.name = 'Day'
                df['Hospital'] = hospital
                df.reset_index(inplace=True)
                hospital_frames.append(df)

        if hospital_frames:
            hospital_df = pd.concat(hospital_frames, ignore_index=True)
            hospital_df.to_csv(f"{output_dir}/hospital_daily_stats.csv", index=False)


if __name__ == '__main__':
    start = time.time()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'Starting simulation, time = {start_time}')

    # Specify save paths
    node_data_file = "../original_data/0513nodes(cure).csv"
    edge_data_file = "../original_data/0513edges(cure).csv"
    node_file = "../simulation_results/network_generate/T500(7)/nodes.csv"
    edge_file = "../simulation_results/network_generate/T500(7)/edges.csv"

    model = HospitalModel(num_hospitals=200, mean_daily_patients=41000, std_daily_patients=200, mean_visits=4, simulation_days=500, stay_probability=0)

    # Run simulation
    simulation_results = model.simulate_patient_visits()

    # Save graph data
    model.save_graph_to_csv(node_file, edge_file)
    model.save_results_to_csv(simulation_results, "../simulation_results/network_generate/T500(7)")

    end = time.time()
    end_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'Simulation ended, time = {end_time}')
    print("Total run time: {:.2f} hours".format((end - start) / 3600))
    print("Current working directory:", os.getcwd())
