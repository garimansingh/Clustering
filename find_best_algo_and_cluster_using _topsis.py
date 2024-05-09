import sys
import pandas as pd
import numpy as np

def check_input_parameters():
    # Check if the correct number of command-line arguments is provided
    if len(sys.argv) != 5:
        print("Usage: python <program.py> <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

def load_and_validate_data(input_file):
    try:
        # Load the data from the specified CSV file
        data = pd.read_csv(input_file)
    except FileNotFoundError:
        print("Error: File not found. Please provide a valid input file.")
        sys.exit(1)

    # Validate the structure of the input data
    if len(data.columns) < 3:
        print("Error: Input file must contain three or more columns.")
        sys.exit(1)

    # Check if the columns from 2nd to last contain numeric values only
    if not data.iloc[:, 1:].applymap(lambda x: np.issubdtype(type(x), np.number)).all().all():
        print("Error: Columns from 2nd to last must contain numeric values only.")
        sys.exit(1)

    return data

def validate_weights_and_impacts(weights, impacts, num_columns):
    weights_list = list(map(int, weights.split(',')))
    impacts_list = impacts.split(',')

    # Validate the number of weights, impacts, and columns
    if len(weights_list) != num_columns - 1 or len(impacts_list) != num_columns - 1:
        print("Error: Number of weights, impacts, and columns must be the same.")
        sys.exit(1)

    # Validate that impacts are either '+' or '-'
    if not all(impact in ['+', '-'] for impact in impacts_list):
        print("Error: Impacts must be either +ve or -ve.")
        sys.exit(1)

def normalize_data(data):
    # Normalize the data using Euclidean normalization
    normalized_data = data.iloc[:, 1:].apply(lambda x: x / np.sqrt(np.sum(x**2)), axis=0)
    return normalized_data

def calculate_topsis_score(data, weights, impacts):
    normalized_data = normalize_data(data)
    weighted_normalized_data = normalized_data * list(map(int, weights.split(',')))

    # Determine the ideal best and ideal worst values based on impacts
    ideal_best = weighted_normalized_data.max() if impacts[0] == '+' else weighted_normalized_data.min()
    ideal_worst = weighted_normalized_data.min() if impacts[0] == '+' else weighted_normalized_data.max()

    # Calculate TOPSIS score
    topsis_score = np.sqrt(np.sum((weighted_normalized_data - ideal_best)**2, axis=1)) / (
            np.sqrt(np.sum((weighted_normalized_data - ideal_best)**2, axis=1)) +
            np.sqrt(np.sum((weighted_normalized_data - ideal_worst)**2, axis=1))
    )
    return topsis_score

def main():
    # Validate command-line input
    check_input_parameters()

    # Extract command-line arguments
    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    result_file = sys.argv[4]

    # Load and validate input data
    data = load_and_validate_data(input_file)

    # Validate weights and impacts
    validate_weights_and_impacts(weights, impacts, len(data.columns))

    # Calculate TOPSIS score
    topsis_score = calculate_topsis_score(data, weights, impacts)

    # Add TOPSIS score and rank to the data
    data['Topsis Score'] = topsis_score
    data['Rank'] = data['Topsis Score'].rank(ascending=False)

    # Save results to a CSV file
    data.to_csv(result_file, index=False)
    print("TOPSIS Implementation Completed")
    print("Results saved to", result_file)

if __name__ == "__main__":
    main()