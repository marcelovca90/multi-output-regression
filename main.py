import numpy as np
from pprint import pprint

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from autosklearn.regression import AutoSklearnRegressor

data = np.load('fas_reduced_dataset.npz')
X, y = data['X'], data['y']

print('\nX shape', X.shape)
print('\ny shape', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

automl = AutoSklearnRegressor(time_left_for_this_task=300,
                              per_run_time_limit=60,
                              memory_limit=8*1024,
                              n_jobs=1,
                              seed=42)

automl.fit(X_train, y_train, dataset_name='fas_reduced')

print('\nLeaderboard\n')
print(automl.leaderboard())

print('\nModels\n')
pprint(automl.show_models(), indent=4)

predictions = automl.predict(X_test)
print("'\nR2 score:", r2_score(y_test, predictions))
