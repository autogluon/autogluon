import csv

__all__ = ['generate_csv']

def generate_csv(inds, path):
    with open(path, 'w') as csvFile:
        row = ['id', 'category']
        writer = csv.writer(csvFile)
        writer.writerow(row)
        id = 1
        for ind in inds:
            row = [id, ind.asscalar()]
            writer = csv.writer(csvFile)
            writer.writerow(row)
            id += 1
    csvFile.close()
