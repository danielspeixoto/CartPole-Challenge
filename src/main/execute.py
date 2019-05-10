from sklearn.ensemble import IsolationForest
from sklearn.utils import validation

from src.main import validate
from src.main.data import get_data, save_data
from src.main.population import populate, test, model_decision
from src.main.train import algorithms, train, validator, fit
from termcolor import colored
import matplotlib.pyplot as plt

requirements = [10, 20, 30, 40, 50]
attempts = 10

results = {}
for alg in algorithms:
    results[alg["name"]] = []

for i in range(attempts):
    print(colored("Attempt: " + str(i + 1) + "\n", "red", attrs=["reverse", "bold"]))
    save_data(requirements, 1000)
    for alg in algorithms:
        name = alg["name"]

        all_points = []
        for req in requirements:
            X, y = get_data(req)

            print("")
            print(colored("Requirement: " + str(req) + "\n", "cyan", attrs=["reverse", "bold"]))
            print(colored("Training: " + name, "green"))
            pipeline = train(alg["algorithm"])
            fitted = fit(pipeline, X, y)
            decide = model_decision(fitted)
            print(colored("Testing: " + name, "green"))
            points = test(decide=decide, games=1000, visualize=False)
            print(colored(name + ": " + str(points) + " points", "blue"))

            # results = validate.k_fold(10, X, y, fitted)
            #
            # print("Accuracy: %.2f" % results[0])
            # print("Precision: %.2f" % results[1])
            # print("Recall: %.2f" % results[2])
            # print("F-Score: %.2f" % results[3])

            all_points.append(points)

        results[alg["name"]].append(all_points)

lines = []
plt.figure()
for alg in algorithms:
    all_points = []
    for idx in range(len(requirements)):
        total = 0
        for occ in results[alg["name"]]:
            total += occ[idx]
        avg = total / attempts
        all_points.append(avg)

    line, = plt.plot(requirements, all_points, label=str(alg["name"]))
    lines.append(line)
plt.legend(handles=lines)
plt.show()


