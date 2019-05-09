from src.main.data import get_data
from src.main.population import populate, test, model_decision
from src.main.train import algorithms, train
from termcolor import colored

requirements = [25, 50, 100]

for req in requirements:
    X, y = get_data(req)
    print("")
    print(colored("Requirement: " + str(req) + "\n", "cyan", attrs=["reverse", "bold"]))
    for alg in algorithms:
        name = alg["name"]
        print(colored("Training: " + name, "green"))
        model = train(alg["algorithm"], X, y)
        decide = model_decision(model)
        print(colored("Testing: " + name, "green"))
        points = test(decide=decide, games=100, visualize=False)
        print(colored(name + ": " + str(points), "blue"))