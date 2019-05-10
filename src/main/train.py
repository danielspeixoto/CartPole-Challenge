from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import validation

algorithms = [
    {
        "name": "Naive Bayes",
        "parameters": {},
        "algorithm": GaussianNB()
    },
    # {
    #     "name": "KNN",
    #     "parameters": {
    #         "clf__n_neighbors": [3],
    #     },
    #     "algorithm": KNeighborsClassifier(n_jobs=-1)
    # },
    {
        "name": "Árvore de Decisão",
        "parameters": {
            'clf__criterion': ['entropy'],
        },
        "algorithm": DecisionTreeClassifier()
    },
    {
        "name": "Regressão Linear",
        "algorithm": LinearRegression()
    },
]

validator = lambda df, y, clf: validation.k_fold(10, df, y, clf),


def train(algorithm):
    run = Pipeline([
        ('clf', algorithm)
    ])
    return run

def fit(pipeline, df, y):
    pipeline.fit(df, y)
    return pipeline