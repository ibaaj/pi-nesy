import csv
import glob
import json
import os
import sys
import time
from typing import List, Dict, Any



def read_csv_file(file_name: str) -> List[Dict[str, Any]]:
    """
    Reads a CSV file and processes its data into a list of dictionaries.

    Parameters:
    - file_name: The name of the CSV file to be read.

    Returns:
    - A list of dictionaries where each dictionary represents a row from the CSV file.
    """
    with open(file_name, mode='r', encoding='utf-8') as file:
        data = csv.reader(file)
        headers = next(data)
        records = [dict(zip(headers, [
            float(value) if header not in ['id', 'label', 'predicted_label', 'train', 'original_id'] else int(value) for
            header, value in zip(headers, row)])) for row in data]

        # Determine mode (4 out_features or 10) based on the number of columns
        mode_full = len(records[0]) == 15

        for record in records:
            
            for key in ['id', 'label', 'predicted_label', 'train', 'original_id']:
                record[key] = int(record.get(key, 0))

            # Create probability distribution dict
            keys_to_remove = range(10) if mode_full else range(4)
            record["probdist"] = {k: record.get(str(k), 0.0) for k in keys_to_remove}
            for key in keys_to_remove:
                del record[str(key)]

        return records



