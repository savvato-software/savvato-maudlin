
import importlib
import subprocess
import pandas as pd

def load_function_from_file(file_path, function_name):
    """Load a function dynamically from a given .py file."""
#    print(f"**** Loading external function {function_name} from {file_path}")
    spec = importlib.util.spec_from_file_location("external_module", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, function_name)

def read_csv_into_dataframe(file_path, n):
    """
    Reads the last N rows of a CSV file using `wc -l` to count total lines.

    Args:
        file_path (str): Path to the CSV file.
        n (int): Number of rows to read from the end.

    Returns:
        pd.DataFrame: A DataFrame containing the last N rows of the CSV file.
    """
    # Use subprocess to execute `wc -l` and get the total number of lines in the file
    result = subprocess.run(['wc', '-l', file_path], stdout=subprocess.PIPE, text=True, check=True)
    total_lines = int(result.stdout.split()[0])  # Extract the total line count

    # Calculate rows to skip (leave at least the header row)
    rows_to_skip = max(total_lines - n, 1)  # Ensure the header is not skipped

    # Read only the last N rows of the file
    df = pd.read_csv(file_path, skiprows=range(1, rows_to_skip))  # Skip rows except the header

    return df

