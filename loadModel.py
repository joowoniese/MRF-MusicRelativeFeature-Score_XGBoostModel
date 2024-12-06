import xgboost as xgb
import numpy as np
from sklearn.metrics import mean_squared_error
import time

print("실행 명령어 : python ./loadModel")
loaded_model = xgb.Booster()
loaded_model.load_model('./MFRmodel.model')

x1_test = np.load('dataNumpy/FrequentNote_Test.npy')
x2_test = np.load('dataNumpy/MFCC_Test.npy')
x3_test = np.load('dataNumpy/Chromagram_Test.npy')
y_test = np.load('dataNumpy/NoteLabel_Test.npy')
label_test = np.load('dataNumpy/filePath_Test.npy')

x1_test = x1_test[:, np.newaxis]
x2_test = x2_test[:, np.newaxis]
x3_test = x3_test[:, np.newaxis]

min_val = np.amin(y_test)
max_val = np.amax(y_test)
y_test = [(val - min_val) / (max_val - min_val) for val in y_test]

x_test = np.concatenate([x1_test, x2_test, x3_test], axis=1)
x_test = xgb.DMatrix(x_test)

final_preds = loaded_model.predict(x_test)

for i in range(len(y_test)):
    test_mse = np.mean((y_test[i] - final_preds[i]) ** 2)
    print(time.strftime('%Y.%m.%d - %H:%M:%S'), end=" ")

    loops = label_test[i].split('/')
    filenum = "Track No." + loops[-1]
    print(filenum, end=" ")

    print("mse = ", test_mse)

mse = mean_squared_error(y_test, final_preds)
print("Total MSE : ", mse)
