import pandas as pd

file_path = '/home/alpha/Desktop/Calibaration_Sensor_Force_1.csv'

# Reading the data file
data_df = pd.read_csv(file_path)

# Displaying the first few rows of the data to understand its structure
print(data_df.head())
