import time

from population import populate, test, model_decision
from train import train_model


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

# model = train_model(child_population(model, 200), "b", model)
# model = train_model(child_population(model, 200))