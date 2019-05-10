import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.main.population import populate
from scipy.stats import zscore


def save_data(requirements, max_results):
    for req in requirements:
        print("Populating, accepting scores >= " + str(req))
        x, y, scores = populate(max_results=max_results, requirement=req)

        position = []
        velocity = []
        angle = []
        pole_velocity = []
        for occ in x:
            position.append(occ[0])
            velocity.append(occ[1])
            angle.append(occ[2])
            pole_velocity.append(occ[3])

        df = pd.DataFrame.from_dict({
            "position": position,
            "velocity": velocity,
            "angle": angle,
            "pole_velocity": pole_velocity,
            "y": y
        })

        columns = ["position", "velocity", "angle", "pole_velocity"]
        for column in columns:
            z_column = column + "_zscore"
            df[z_column] = zscore(df[column])
            df = df[df.apply(lambda x: -2.2 <= x[z_column] <= 2.2, axis=1)]


        new_x = []
        new_y = []

        for index, row in df.iterrows():
            occ = []
            for i in range(len(columns)):
                occ.append(row[columns[i]])
            new_x.append(occ)
            new_y.append(row["y"])



        # column = 2
        # for a in x:
        #     column_data.append(a[column])
        #
        # sns.distplot(column_data)
        # plt.show()

        with open(str(req) + "x.pkl", "wb") as f:
            pickle.dump(new_x, f)
        with open(str(req) + "y.pkl", "wb") as f:
            pickle.dump(new_y, f)


def get_data(req):
    with open(str(req) + "x.pkl", "rb") as f:
        x = pickle.load(f)
        with open(str(req) + "y.pkl", "rb") as fy:
            y = pickle.load(fy)
            return x, y
