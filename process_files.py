import csv
import re

import numpy as np
import pandas as pd

from dbscan import db_scan

eps = .0001  # equivalent to ~10m
min_samples = 2  # samples for DB scan
min_unique = 2  # how much consecutive time at a location to count as unique
min_gap = 10  # how much noise gap between the same location to count twice

from os import listdir
from os.path import isfile, join

# Get files from data directory
files = [join("data", f) for f in listdir("data") if isfile(join("data", f))]
output_data = list()
for f in files:
    # Run DBScan on files
    pid = re.findall(r"\d+", f)[0]  # Extract pid from filename
    output = db_scan(f, eps, min_samples, min_unique, min_gap, pid)
    output_data.append(output)

df = pd.concat(output_data)
df.rename(columns={0: "Lat", 1: "Long", 2:"Timestamp"}, inplace=True)

df.to_csv("data.csv", index=False)
