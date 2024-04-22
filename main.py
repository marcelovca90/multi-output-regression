import numpy as numpy
import pandas as pd
from pprint import pprint

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from autosklearn.regression import AutoSklearnRegressor

df = pd.read_parquet('synthetic.parquet')

X, y = df.filter(like='x'), df.filter(like='y')

print('\n\nX shape and columns', X.shape, X.columns.tolist())
print('\n\ny shape and columns', y.shape, y.columns.tolist())

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

automl = AutoSklearnRegressor(time_left_for_this_task=300,
                              per_run_time_limit=60,
                              memory_limit=8*1024,
                              seed=42)

automl.fit(X_train, y_train, dataset_name='synthetic')

print('\n\nLeaderboard\n\n')
print(automl.leaderboard())

print('\n\nModels\n\n')
pprint(automl.show_models(), indent=4)

predictions = automl.predict(X_test)
print("'\n\nR2 score:", r2_score(y_test, predictions))
