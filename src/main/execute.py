from sklearn.ensemble import IsolationForest
from sklearn.utils import validation

from src.main import validate
from src.main.data import get_data, save_data
from src.main.population import populate, test, model_decision
from src.main.train import algorithms, train, validator, fit
from termcolor import colored
import matplotlib.pyplot as plt
import seaborn as sns
# requirements = [30, 40]
requirements = [10, 20, 30, 40]
attempts = 100

info = {}
metrics = {}
ponct = {}
for alg in algorithms:
    info[alg["name"]] = {
        "points": [],
        "results": []
    }
    metrics[alg["name"]] = {
        "acc": [],
        "precision": [],
        "recall": [],
        "f_score": [],
    }
    ponct[alg["name"]] = []

for i in range(attempts):
    print(colored("Attempt: " + str(i + 1) + "\n", "red", attrs=["reverse", "bold"]))
    save_data(requirements, 1000)
    for alg in algorithms:
        name = alg["name"]

        all_points = []
        all_results = []
        for req in requirements:
            X, y = get_data(req)

            print("")
            print(colored("Requirement: " + str(req) + "\n", "cyan", attrs=["reverse", "bold"]))
            print(colored("Training: " + name, "green"))
            pipeline = train(alg["algorithm"])
            fitted = fit(pipeline, X, y)
            decide = model_decision(fitted)
            print(colored("Testing: " + name, "green"))
            points = test(decide=decide, games=100, visualize=False)
            print(colored(name + ": " + str(points) + " points", "blue"))

            results = validate.k_fold(10, X, y, fitted)

            print("Accuracy: %.2f" % results[0])
            print("Precision: %.2f" % results[1])
            print("Recall: %.2f" % results[2])
            print("F-Score: %.2f" % results[3])

            all_points.append(points)
            if req == 40:
                ponct[name] = ponct[name] + all_points
            all_results.append(results)

        info[alg["name"]]["points"].append(all_points)
        info[alg["name"]]["results"].append(all_results)

lines = []
plt.figure()
for alg in algorithms:
    all_points = []
    for idx in range(len(requirements)):
        total = 0
        for occ in info[alg["name"]]["points"]:
            total += occ[idx]
        avg = total / attempts
        all_points.append(avg)

    line, = plt.plot(requirements, all_points, label=str(alg["name"]))
    lines.append(line)
plt.ylabel("Pontos")
plt.xlabel("Pontuação mínima para treinamento")
plt.legend(handles=lines)

for alg in algorithms:
    name = alg["name"]
    for idx in range(len(requirements)):
        acc = 0
        precision = 0
        recall = 0
        f_score = 0
        for occ in info[name]["results"]:
            acc += occ[idx][0]
            precision += occ[idx][1]
            recall += occ[idx][2]
            f_score += occ[idx][3]
        acc = acc/attempts
        precision = precision/attempts
        recall = recall/attempts
        f_score = f_score/attempts

        metrics[name]["acc"].append(acc)
        metrics[name]["precision"].append(precision)
        metrics[name]["recall"].append(recall)
        metrics[name]["f_score"].append(f_score)


metric_d = {
    "acc": "Acurácia",
    "precision": "Precisão",
    "recall": "Revocação",
    "f_score": "F1-Score"
}
for metric in ["acc", "precision", "recall", "f_score"]:
    metric_lines = []
    plt.figure()
    for alg in algorithms:
        name = alg["name"]
        line, = plt.plot(requirements, metrics[name][metric], label=str(name))
        metric_lines.append(line)
    plt.legend(handles=metric_lines)
    plt.xlabel("Pontuação mínima para treinamento")
    plt.ylabel(metric_d[metric])

plt.figure()
axes = []
for alg in algorithms:
    name = alg["name"]
    axes.append(sns.distplot(ponct[name], bins=5, label=name))
plt.legend(handles=axes)
plt.xlabel("Pontuação mínima para treinamento")
plt.ylabel("Pontuação")
plt.show()


