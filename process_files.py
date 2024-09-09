import csv
import re

import numpy as np

from dbscan import db_scan

eps = .0001  # equivalent to ~10m
min_samples = 1  # samples for DB scan
min_unique = 1  # how much consecutive time at a location to count as unique
min_gap = 10  # how much noise gap between the same location to count twice

from os import listdir
from os.path import isfile, join

# Get files from data directory
files = [join("data", f) for f in listdir("data") if isfile(join("data", f))]
output_data = list()
for f in files:
    # Run DBScan on files
    output = db_scan(f, eps, min_samples, min_unique, min_gap)
    row = []
    [row.extend([k, v]) for k, v in output.items()]  # Expand data to a single row
    pid = re.findall(r"\d+", f)[0]  # Extract pid from filename
    row.insert(0, pid)
    output_data.append(row)

with open('output.csv', 'w', newline='\n') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(output_data)
