#!/usr/bin/env python3
import numpy as np

if __name__ == "__main__":
    # Load data distribution, each data point on a line
    ldata = []
    #with open("numpy_entropy_eval_examples/numpy_entropy_data_3.txt", "r") as data:
    with open("numpy_entropy_data.txt", "r") as data:
        for line in data:
            line = line.rstrip("\n")
            # TODO: process the line
            ldata.append(line)
    # TODO: Create a NumPy array containing the data distribution
    data = np.array(ldata)
    unique, counts = np.unique(data, return_counts=True)
    datamodel = dict(zip(unique, counts / np.sum(counts)))
    #print(datamodel)

    # Load model distribution, each line `word \t probability`, creating
    # a NumPy array containing the model distribution
    modeldist = {}
    #with open("numpy_entropy_eval_examples/numpy_entropy_model_3.txt", "r") as model:
    with open("numpy_entropy_model.txt", "r") as model:
        for line in model:
            line = line.rstrip("\n")
            char = line.split('\t')[0]
            value = float(line.split('\t')[1])
            modeldist[char] = value

    # TODO: Compute and print entropy H(data distribution)
    mergedKeys = set(datamodel.keys()) | set(modeldist.keys())  # pipe is union
    for key in mergedKeys:
        if not(key in modeldist):
            modeldist[key] = 0

    entropy = np.sum([prob * np.log(prob) for prob in datamodel.values()]) * -1
    print("{:.2f}".format(entropy))

    cross_entropy =  np.sum([datamodel[key] * np.log(modeldist[key]) for key in datamodel.keys()]) * -1
    print("{:.2f}".format(cross_entropy))
    print("{:.2f}".format(cross_entropy-entropy))


    # TODO: Compute and print cross-entropy H(data distribution, model distribution)
    # and KL-divergence D_KL(data distribution, model_distribution)
