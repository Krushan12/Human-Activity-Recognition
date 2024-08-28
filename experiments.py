import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
import os

np.random.seed(42)

def create_fake_data(N, M, decision_tree_type):
    # Real Input Real Output
    if decision_tree_type == "RIRO":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
    # Real Input Discrete Output
    elif decision_tree_type == "RIDO":
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(2, size=N), dtype="category")
    # Discrete input Discrete Output
    elif decision_tree_type == "DIDO":
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(2, size=N), dtype="category")
    # Discrete Input Real Output
    elif decision_tree_type == "DIRO":    
        X = pd.DataFrame({i: pd.Series(np.random.randint(2, size=N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
    return X, y

def calculate_time_fit(func, X, y):
    start_time = time.time()
    func(X, y)
    end_time = time.time()
    return end_time - start_time

def calculate_time_predict(func, X):
    start_time = time.time()
    func(X)
    end_time = time.time()
    return end_time - start_time

def run_experiment(N_values, M_values, criteria, decision_tree_type):
    results = {'N': [], 'M': [], 'fit_time': [], 'predict_time': []}

    for N in N_values:
        for M in M_values:
            # Create fake data
            X, y = create_fake_data(N, M, decision_tree_type)

            # Initialize a new DecisionTree for each iteration
            dt = DecisionTree(criterion=criteria)

            # Measure fit time
            time_fit = calculate_time_fit(dt.fit, X, y)
            results['N'].append(N)
            results['M'].append(M) 
            results['fit_time'].append(time_fit)

            # Measure predict time
            test_data = create_fake_data(N, M, decision_tree_type)[0]
            time_predict = calculate_time_predict(dt.predict, test_data)
            results['predict_time'].append(time_predict)

    return pd.DataFrame(results)

def plot_results(results, M_values, decision_tree_type, criteria, output_folder):
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Plot and save Fit Time
    plt.figure(figsize=(10, 6))
    for M in M_values:
        subset = results[results['M'] == M]
        plt.plot(subset['N'], subset['fit_time'], label=f'M={M}')

    plt.title(f'Decision Tree {decision_tree_type} - {criteria} - Fit Time')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Fit Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'{decision_tree_type}_{criteria}_fit_time.png'))
    plt.close()

    # Plot and save Predict Time
    plt.figure(figsize=(10, 6))
    for M in M_values:
        subset = results[results['M'] == M]
        plt.plot(subset['N'], subset['predict_time'], label=f'M={M}')

    plt.title(f'Decision Tree {decision_tree_type} - {criteria} - Predict Time')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Predict Time (seconds)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, f'{decision_tree_type}_{criteria}_predict_time.png'))
    plt.close()

# Define the values of N and M to experiment with
N_values = np.arange(10, 35)
M_values = np.arange(5, 9)

avg_fit = {}
std_fit = {}

avg_pred = {}
std_pred = {}

for decision_tree_type in ["RIRO", "RIDO", "DIDO", "DIRO"]:
    for criteria in ["information_gain", "gini_index"]:
        # Run experiment
        results = run_experiment(N_values, M_values, criteria, decision_tree_type)
        # Plot results
        plot_results(results, M_values, decision_tree_type, criteria,'Graphs_Q4')
        # average and standard deviation
        avg_fit[f'({decision_tree_type},{criteria})'] = np.mean(results['fit_time'])
        std_fit[f'({decision_tree_type},{criteria})'] = np.std(results['fit_time'])
        avg_pred[f'({decision_tree_type},{criteria})'] = np.mean(results['predict_time'])
        std_pred[f'({decision_tree_type},{criteria})'] = np.std(results['predict_time'])

print(avg_fit)
print(avg_pred)
print(std_fit)
print(std_pred)
