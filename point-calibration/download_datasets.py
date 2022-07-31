import requests
import os
import zipfile

MAIN_DIREC = "data_loaders/data/uci/"
if not os.path.isdir("data_loaders/data"):
    os.mkdir("data_loaders/data")
if not os.path.isdir("data_loaders/data/uci/"):
    os.mkdir(MAIN_DIREC)

directories = [
    "protein",
    "energy",
]

simple_urls = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
    "https://github.com/Srceh/DistCal/blob/master/Dataset/Appliances_energy_prediction.csv?raw=true",
]

file_names = [
    "CASP.csv",
    "energydata_complete.csv",
]


for i in range(len(directories)):
    print("Downloading {}".format(directories[i]))
    os.mkdir(MAIN_DIREC + directories[i])
    r = requests.get(simple_urls[i], allow_redirects=True)
    open(MAIN_DIREC + "{}/{}".format(directories[i], file_names[i]), "wb").write(
        r.content
    )

zip_directories = ["naval"]
zip_urls = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip",
]

zip_file_names = ["/CCPP.zip", "/UCI_CBM_Dataset.zip", "/YearPredictionMSD.zip"]

for i in range(len(zip_directories)):
    print("Downloading {}".format(zip_directories[i]))
    path = MAIN_DIREC + zip_directories[i]
    os.mkdir(path)
    r = requests.get(zip_urls[i], allow_redirects=True)
    open(path + zip_file_names[i], "wb").write(r.content)
    with zipfile.ZipFile(path + zip_file_names[i], "r") as zip_ref:
        zip_ref.extractall(MAIN_DIREC + zip_directories[i])

directory = "crime"
crime_urls = [
    "https://raw.githubusercontent.com/ShengjiaZhao/Individual-Calibration/master/data/communities.data",
    "https://raw.githubusercontent.com/ShengjiaZhao/Individual-Calibration/master/data/names",
]
print("Downloading {}".format(directory))
os.mkdir(MAIN_DIREC + directory)
file_names = ["communities.data", "names"]

for i in range(len(crime_urls)):
    r = requests.get(crime_urls[i], allow_redirects=True)
    open(MAIN_DIREC + "{}/{}".format(directory, file_names[i]), "wb").write(r.content)
