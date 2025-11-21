""" Helper script to gather results from multiple runs and save them to a single csv file. """

import argparse
import os
import pandas as pd

def clean_results():
    file_path = 

def gather_results(run_dir):
    """ Gather results from a single run and save them to a csv file. """
    results_dir = os.path.join(run_dir, "results")
    results_files = os.listdir(results_dir)
    results_files = [f for f in results_files if f.endswith(".csv")]
    results_files = [os.path.join(results_dir, f) for f in results_files]
    results_files = [pd.read_csv(f) for f in results_files]
    results_df = pd.concat(results_files)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gather results from multiple runs and save them to a single csv file.")
    parser.add_argument("--run_dir", type=str, default="./runs", help="Directory containing the results to gather.")
    args = parser.parse_args()

    gather_results(args.run_dir)