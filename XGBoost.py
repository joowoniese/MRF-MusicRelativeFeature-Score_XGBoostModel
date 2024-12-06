import xgboost as xgb
import numpy as np

np.set_printoptions(threshold=np.inf)

x1_train = np.load('dataNumpy/FrequentNote_Train.npy')
x2_train = np.load('dataNumpy/MFCC_Train.npy')
x3_train = np.load('dataNumpy/Chromagram_Train.npy')
y_train = np.load('dataNumpy/NoteLabel_Train.npy')
label_train = np.load('dataNumpy/filePath_Train.npy')

x1_test = np.load('dataNumpy/FrequentNote_Test.npy')
x2_test = np.load('dataNumpy/MFCC_Test.npy')
x3_test = np.load('dataNumpy/Chromagram_Test.npy')
y_test = np.load('dataNumpy/NoteLabel_Test.npy')
label_test = np.load('dataNumpy/filePath_Test.npy')

x1_valid = np.load('dataNumpy/FrequentNote_Valid.npy')
x2_valid = np.load('dataNumpy/MFCC_Valid.npy')
x3_valid = np.load('dataNumpy/Chromagram_Valid.npy')
y_valid = np.load('dataNumpy/NoteLabel_Valid.npy')
label_valid = np.load('dataNumpy/filePath_Valid.npy')

x1_train = x1_train[:, np.newaxis]
x2_train = x2_train[:, np.newaxis]
x3_train = x3_train[:, np.newaxis]

x1_test = x1_test[:, np.newaxis]
x2_test = x2_test[:, np.newaxis]
x3_test = x3_test[:, np.newaxis]

x1_valid = x1_valid[:, np.newaxis]
x2_valid = x2_valid[:, np.newaxis]
x3_valid = x3_valid[:, np.newaxis]


min_val = np.amin(y_train)
max_val = np.amax(y_train)
y_train = [(val - min_val) / (max_val - min_val) for val in y_train]

min_val = np.amin(y_test)
max_val = np.amax(y_test)
y_test = [(val - min_val) / (max_val - min_val) for val in y_test]

min_val = np.amin(y_valid)
max_val = np.amax(y_valid)
y_valid = [(val - min_val) / (max_val - min_val) for val in y_valid]

x_train = np.concatenate([x1_train, x2_train, x3_train], axis=1)
x_test = np.concatenate([x1_test, x2_test, x3_test], axis=1)
x_valid = np.concatenate([x1_valid, x2_valid, x3_valid], axis=1)


def evaluate(individual):
    n_estimators = int(individual[0])
    learning_rate = max(individual[1], 0)
    alpha = max(individual[2], 0)
    lambda_ = max(individual[3], 0)

    model = xgb.XGBRegressor(n_estimators=n_estimators,
                             learning_rate=learning_rate,
                             reg_alpha=alpha,
                             reg_lambda=lambda_)

    model.fit(x_train, y_train)
    preds = model.predict(x_valid)
    loss = ((y_valid - preds) ** 2).mean()

    print(f"MSE for individual {individual}: {loss}")

    return loss,

best_model = xgb.XGBRegressor(n_estimators=410, learning_rate=0.11526, reg_alpha=0.317, reg_lambda=2.90999)

evals = [(x_train, y_train), (x_valid, y_valid)]
best_model.fit(x_train, y_train, eval_metric="rmse", eval_set=evals, verbose=True)
results = best_model.evals_result()