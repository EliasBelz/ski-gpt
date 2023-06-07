# import json


# # Convert the data to JSON format
# data = json.loads(Path(file_path).read_text())
# json_data = json.dumps(data, indent=4)

# # Write the JSON data to a file
# with open("data.txt", "w") as file:
#     file.write(json_data)
import json
import csv

# Load JSON data from a file
with open('snowboardData200.json') as json_file:
    data = json.load(json_file)

# Extracting headers from the details field
headers = data[0]['details'].keys()

# Opening a CSV file for writing
with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)

    # Writing headers
    writer.writerow(['productType', 'productName', 'price', 'url'] + list(headers))

    # Writing data rows
    for item in data:
        row = [item['productType'], item['productName'], item['price'], item['url']]
        details = item['details']
        row += [details[key] for key in headers]
        writer.writerow(row)
