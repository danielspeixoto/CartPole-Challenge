import pickle

from src.main.population import populate

def save_data(requirements, max_results):
    for req in requirements:
        print("Populating, accepting scores >= " + str(req))
        X, y, average = populate(max_results=max_results, requirement=req)
        with open(str(req) + "x.pkl", "wb") as f:
            pickle.dump(X, f)
        with open(str(req) + "y.pkl", "wb") as f:
            pickle.dump(y, f)

def get_data(req):
    with open(str(req) + "x.pkl", "rb") as f:
        x = pickle.load(f)
        with open(str(req) + "y.pkl", "rb") as fy:
            y = pickle.load(fy)
            return x, y