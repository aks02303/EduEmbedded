import pandas as pd

# Load the CSV files into pandas DataFrames
triples_df = pd.read_csv('triples.csv')
filtered_triples_df = pd.read_csv('filtered_triples.csv')

# Merge the two DataFrames
merged_df = pd.concat([triples_df, filtered_triples_df], ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged.csv', index=False)

print("Merged file 'merged.csv' has been created.")
