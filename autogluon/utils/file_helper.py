import csv

__all__ = ['generate_csv']

def generate_csv(inds, path):
    with open(path, 'w') as csvFile:
        for ind in inds:
            row = [ind.asscalar()]
            writer = csv.writer(csvFile)
            writer.writerow(row)
    csvFile.close()