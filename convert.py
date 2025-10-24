import csv
import datetime
import re

from os import listdir
from os.path import isfile, join, basename

import pandas as pd
from dateutil import parser

files = [join("data", f) for f in listdir("data") if isfile(join("data", f))]
output_data = list()
for f in files:
    # Run DBScan on files
    data = pd.read_csv(f, header=None, parse_dates=[2])  # Read file in with pandas to get proper date times
    data[3] = data[2].dt.normalize()  # Create date column to collate by day
    dates = data[3].unique()
    day = 1
    output = []

    # Start Day Nov 4
    start = parser.parse("2025-01-27 00:00:00")
    # End Day Dec 15
    end = parser.parse("2025-03-09 23:59:59")

    current_index = 0
    current_dt = data[2][0]
    next_dt = data[2][1]
    current_output = data[2][0]
    if current_output < start:
        current_output = start
    output_index = 0
    while current_output < end:

        if current_dt <= current_output < next_dt:
            lat = data[0][current_index]
            long = data[1][current_index]
            output.append([lat, long, current_output])
            current_output = current_output + datetime.timedelta(minutes=2)
        else:
            current_index += 1
            if current_index > len(data) - 2:
                current_dt = data[2][current_index]
                next_dt = parser.parse("3000-01-01 00:00:00")
            else:
                current_dt = data[2][current_index]
                next_dt = data[2][current_index + 1]

    filename = basename(f)
    print(filename)
    with open(join("output", filename), "w+", newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for line in output:
            wr.writerow(line)