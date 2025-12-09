from datetime import datetime, timedelta
import csv
import sys
import os

HOLIDAYS_RAW = [
    "2021-02-15 till 2021-02-19",
    "2021-04-01 till 2021-04-16",
    "2021-05-03",
    "2021-05-31 till 2021-06-04",
    "2021-07-22 till 2021-08-31",
    "2021-08-02 till 2021-08-31",
    "2021-10-25 till 2021-10-29",
    "2021-12-20 till 2022-01-03",
    "2022-02-21 till 2022-02-25",
    "2022-04-11 till 2022-04-22",
    "2022-05-02",
    "2022-05-30 till 2022-06-03",
    "2022-07-21 till 2022-07-29",
    "2022-08-01 till 2022-09-01",
    "2022-10-24 till 2022-10-28",
    "2022-12-21 till 2023-01-03",
    "2023-02-13 till 2023-02-17",
    "2023-04-03 till 2023-04-14",
    "2023-05-01",
    "2023-05-29 till 2023-06-02",
    "2023-07-24 till 2023-07-31",
    "2023-08-01 till 2023-08-31",
    "2023-10-23 till 2023-10-27",
    "2023-12-21 till 2024-01-05",
    "2024-02-12 till 2024-02-16",
    "2024-03-29 till 2024-04-12",
    "2024-05-06",
    "2024-05-27 till 2024-05-31",
    "2024-07-25 till 2024-09-02",
    "2024-10-28 till 2024-11-1",
    "2024-12-23 till 2025-01-3",
    "2025-02-12 till 2025-02-21",
    "2025-04-07 till 2025-04-21",
    "2025-05-05",
    "2025-05-26 till 2025-12-30",
    "2025-07-24 till 2025-09-02"
]

def generate_holiday_set(holidays_raw):
    holiday_set = set()
    for h in holidays_raw:
        if "till" in h:
            start_str, end_str = h.split(" till ")
            start = datetime.strptime(start_str, "%Y-%m-%d").date()
            end = datetime.strptime(end_str, "%Y-%m-%d").date()
            current = start
            while current <= end:
                holiday_set.add(current)
                current += timedelta(days=1)
        else:
            holiday_set.add(datetime.strptime(h, "%Y-%m-%d").date())
    return holiday_set

HOLIDAYS = generate_holiday_set(HOLIDAYS_RAW)

def classify_date(date_str):
    try:
        date = datetime.strptime(date_str, "%d/%m/%Y %H:%M:%S").date()
    except ValueError:
        raise ValueError(f"Invalid date format: {date_str}")

    if date.weekday() >= 5:
        return "off day"

    if date in HOLIDAYS:
        return "off day"

    return "school day"

def process_csv(input_file, output_file):
    with open(input_file, mode='r', newline='', encoding='utf-8') as f_in:
        reader = csv.DictReader(f_in)
        rows = list(reader)

    fieldnames = reader.fieldnames + ["Day Type"]

    with open(output_file, mode='w', newline='', encoding='utf-8') as f_out:
        writer = csv.DictWriter(f_out, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            row["Day Type"] = classify_date(row["Report Date"])
            writer.writerow(row)


def process_folder(folder_path):
    folder_path = os.path.abspath(folder_path)

    if not os.path.isdir(folder_path):
        print(f"ERROR: '{folder_path}' is not a folder.")
        return

    parent = os.path.dirname(folder_path)
    out_folder = os.path.join(parent, os.path.basename(folder_path) + "-classified")

    os.makedirs(out_folder, exist_ok=True)

    print(f"\nInput folder: {folder_path}")
    print(f"Output folder: {out_folder}\n")

    csv_files = [f for f in os.listdir(folder_path) if f.lower().endswith(".csv")]

    if not csv_files:
        print("No CSV files found.")
        return

    for file in csv_files:
        in_path = os.path.join(folder_path, file)

        name, _ = os.path.splitext(file)
        out_file = f"{name}-classified.csv"

        out_path = os.path.join(out_folder, out_file)

        print(f"Processing: {file} → {out_file}")
        process_csv(in_path, out_path)

    print("\nDone. All classified files saved.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python classify_dataset.py /path/to/traffic-folder")
        sys.exit(1)

    process_folder(sys.argv[1])
