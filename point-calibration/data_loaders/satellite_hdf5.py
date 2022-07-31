import numpy as np
import h5py
import pandas as pd

countries = ["tanzania", "mozambique", "malawi", "uganda", "rwanda", "zimbabwe"]

for country in countries:
    filename = "satellite.h5"
    hf = h5py.File("data/{}/satellite.h5".format(country), "w")
    labels = pd.read_csv("data/{}/data.csv".format(country))
    print(country, len(labels))
    for i in range(len(labels)):
        g1 = hf.create_group("{}".format(i))
        img = np.genfromtxt("data/{}/imgs/img_{}.csv".format(country, i))
        g1.create_dataset("image", data=img)
        g1.create_dataset("label", data=labels["label"][i])
        if i % 100 == 0:
            print("finished {}".format(i))

# with h5py.File(filename, "r") as f:
# List all groups
#    print("Keys: %s" % f.keys())
#    a_group_key = list(f.keys())[0]

# Get the data
#    data = f[a_group_key]["image"]
#    print(data)
