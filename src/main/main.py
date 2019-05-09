import time

from src.main.population import populate, test, model_decision


training, average = populate(games=166)
model = train_model(training, "a")

iterations = 2
for i in range(iterations):
    print("New training")
    time.sleep(1)
    decide = model_decision(model)
    training, average = populate(games=167, requirement=average)
    model = train_model(training, str(i), model)

decide = model_decision(model)
test(decide=decide, games=100, visualize=True)