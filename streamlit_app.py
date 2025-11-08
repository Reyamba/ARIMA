import pandas as pd
import requests
from io import StringIO

# --- Configuration ---

# IMPORTANT: You must use the RAW GitHub URL for the CSV file.
# To get the raw URL:
# 1. Navigate to your CSV file on GitHub.
# 2. Click the 'Raw' button.
# 3. Copy the URL from your browser's address bar.

# Example of a RAW GitHub URL (replace this with your actual raw URL):
# If your file is at: https://github.com/user/repo/blob/main/data/my_data.csv
# The RAW URL should look like this:
csv_url = "https://raw.githubusercontent.com/pandas-dev/pandas/main/doc/data/titanic.csv"

# --- Function to Load Data ---

def load_csv_from_github(url):
    """
    Reads a CSV file from a raw GitHub URL directly into a pandas DataFrame.

    Args:
        url (str): The raw URL of the CSV file on GitHub.

    Returns:
        pd.DataFrame or None: The DataFrame containing the CSV data, or None if an error occurred.
    """
    print(f"Attempting to load data from: {url}")
    try:
        # pandas read_csv can handle direct HTTP/HTTPS URLs
        df = pd.read_csv(url)
        print("\nSuccessfully loaded the data!")
        print(f"Shape of the DataFrame: {df.shape}")
        return df
    except requests.exceptions.RequestException as e:
        print(f"\n[ERROR] Network or request error: {e}")
        print("Please ensure the URL is correct and you have an internet connection.")
    except pd.errors.EmptyDataError:
        print("\n[ERROR] The URL returned empty data or an invalid file.")
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        
    return None

# --- Main Execution ---

if __name__ == "__main__":
    # Load the data
    data_frame = load_csv_from_github(csv_url)

    # Check if loading was successful and display the first few rows
    if data_frame is not None:
        print("\n--- First 5 Rows of the DataFrame ---")
        print(data_frame.head())
        
        print("\n--- Column Data Types ---")
        print(data_frame.dtypes)

        # You can now proceed with your data analysis, e.g.,
        # print(f"\nAverage Age: {data_frame['Age'].mean():.2f}")
