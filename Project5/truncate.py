import csv
def truncate_csv(input_path, output_path, line_number):
    with open(input_path, 'r', newline='', encoding='utf-8') as infile:
        reader = list(csv.reader(infile))
        # Keep rows up to the given line_number (1-based)
        truncated_rows = reader[:line_number]

    with open(output_path, 'w', newline='', encoding='utf-8') as outfile:
        writer = csv.writer(outfile)
        writer.writerows(truncated_rows)

# Example usage:
# Keep only the first 5 lines (including header, if present)
truncate_csv('DOW.csv', 'output.csv', 137675)