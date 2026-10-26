import pandas as pd
input_file = "student-por.csv"
output_file = "processed_student_data.csv"
# student-por.csv is comma-separated
df = pd.read_csv(input_file)
print("Input shape:", df.shape)
print("Input columns:", df.columns.tolist())
# Save as a standard comma-separated CSV
df.to_csv(output_file, index=False)
print("Data Preprocessing Completed")
