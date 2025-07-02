import numpy as np
import networkx as nx
import random
import math
import pandas as pd


def generate_hospital_network(num_hospitals=200):
    G = nx.complete_graph(num_hospitals, create_using=nx.DiGraph())

    # (1) Hospital grade and quantity: ratio of Grade 1, 2, and 3 is 5:4:1
    num_grade_1 = int(num_hospitals * 5 / 10)
    num_grade_2 = int(num_hospitals * 4 / 10)
    num_grade_3 = num_hospitals - num_grade_1 - num_grade_2
    grades = [1] * num_grade_1 + [2] * num_grade_2 + [3] * num_grade_3
    random.shuffle(grades)

    cure_rates = {1: 0.75, 2: 0.85, 3: 0.95}

    for i, node in enumerate(G.nodes):
        G.nodes[node]["grade"] = grades[i]
        G.nodes[node]["cure_rate"] = cure_rates.get(grades[i], 0.0)

    # (2) Generate hospital geographical locations
    L = 200  # unit: kilometers
    locations = []
    for node in G.nodes:
        while True:
            x = random.uniform(0, L)
            y = random.uniform(0, L)
            valid = all(math.sqrt((x - loc[0]) ** 2 + (y - loc[1]) ** 2) >= 1 for loc in locations)
            if valid:
                locations.append((x, y))
                G.nodes[node]['location'] = (x, y)
                break

    # Compute edge attributes: geographic distance, edge weight, grade preference probability
    for u, v in G.edges:
        a = 1
        g_ij = G.nodes[u]["grade"] - G.nodes[v]["grade"]
        G[u][v]['preference'] = round(1 / (1 + np.exp(a * g_ij)), 3)
        G[u][v]['weight'] = 1
        x1, y1 = G.nodes[u]['location']
        x2, y2 = G.nodes[v]['location']
        G[u][v]['distance'] = round(math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2), 3)

    # (3) Hospital capacity attributes
    total_free_capacity = 0
    for node in G.nodes:
        grade = G.nodes[node]['grade']
        if grade == 1:
            visit_limit = np.random.randint(50, 101)
        elif grade == 2:
            visit_limit = np.random.randint(101, 501)
        else:
            visit_limit = np.random.randint(501, 1201)

        G.nodes[node]['visit_limit'] = visit_limit
        G.nodes[node]['VP_initial'] = 0
        G.nodes[node]['initial_distance'] = 0
        G.nodes[node]['VP_referral'] = 0
        G.nodes[node]['referral_distance'] = 0
        G.nodes[node]['LP'] = 0
        G.nodes[node]['cure_LP'] = 0
        G.nodes[node]['free_capacity'] = visit_limit
        total_free_capacity += visit_limit

    return G, total_free_capacity


def save_network_to_csv(G, node_file=r'C:\Users\Administrator\Desktop\data\original_data\0513nodes(cure).csv',
                        edge_file=r'C:\Users\Administrator\Desktop\data\original_data\0513edges(cure).csv'):
    node_data = []
    for node, data in G.nodes(data=True):
        node_data.append([
            node, data['grade'], data['cure_rate'], data['location'],
            data['visit_limit'], data['VP_initial'], data['initial_distance'], data['VP_referral'], data['referral_distance'], data['LP'], data['cure_LP'], data['free_capacity']
        ])

    node_df = pd.DataFrame(node_data, columns=[
        'Node', 'grade', 'cure_rate', 'location', 'visit_limit', 'VP_initial', 'initial_distance', 'VP_referral', 'referral_distance', 'LP', 'cure_LP', 'free_capacity'
    ])
    node_df.to_csv(node_file, index=False)

    edge_data = []
    for u, v, data in G.edges(data=True):
        edge_data.append([u, v, data['preference'], data['weight'], data['distance']])

    edge_df = pd.DataFrame(edge_data, columns=['Source', 'Target', 'preference', 'weight', 'distance'])
    edge_df.to_csv(edge_file, index=False)


# Example run
G, total_free_capacity = generate_hospital_network()
save_network_to_csv(G)
