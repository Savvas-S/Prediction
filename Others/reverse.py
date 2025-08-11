import csv

input_file = "joker_results2.csv"
output_file = "joker_results2_reversed.csv"

# Read all rows
with open(input_file, "r", encoding="utf-8") as f:
    reader = list(csv.reader(f))
    header = reader[0]
    rows = reader[1:]

# Reverse order
rows.reverse()

# Write back to new file (or overwrite original if you want)
with open(output_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(rows)

print(f"Reversed order saved to {output_file}")
