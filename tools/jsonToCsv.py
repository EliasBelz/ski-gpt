import json
import csv

with open('snowboardData200.json') as json_file:
    data = json.load(json_file)

headers = data[0]['details'].keys()

with open('output.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['productType', 'productName', 'price', 'url'] + list(headers))
    for item in data:
        row = [item['productType'], item['productName'], item['price'], item['url']]
        details = item['details']
        row += [details[key] for key in headers]
        writer.writerow(row)
