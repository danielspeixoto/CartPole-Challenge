from populate import initial_population, child_population
from test import test
from train import train_model

model = train_model(initial_population(), "a")
print("part 2")
model = train_model(child_population(model, 200), "b", model)
# model = train_model(child_population(model, 200))
test(model)