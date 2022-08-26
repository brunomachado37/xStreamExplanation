import pandas as pd
import numpy as np
from statistics import mean, stdev


def convertMap(featureMap):
    ret = []

    for map in featureMap:
        temp = []
        for idx, feature in enumerate(map):
            if feature:
                temp.append(idx)
 
        ret.append(temp)

    return ret



def add_noise(X, irrelevant_features_num, standarize_X = True):
    X_cp = X.copy()

    if standarize_X:
        X_cp = (X_cp - X_cp.mean()) / X_cp.std()
        noise = np.random.normal(0, 1, (X_cp.shape[0], irrelevant_features_num))
    else:
        noise = np.random.normal(mean([y for x in X_cp.values.tolist() for y in x]),  stdev([y for x in X_cp.values.tolist() for y in x]), (X_cp.shape[0], irrelevant_features_num))
        # noise = np.random.normal(0, 1, (X_cp.shape[0], irrelevant_features_num))

    fnames = ['irr' + str(i) for i in range(irrelevant_features_num)]
    noise_df = pd.DataFrame(noise, columns=fnames)

    return pd.concat([X_cp, noise_df], axis=1)



def load_data(path, numberOfNoiseFeatures):
    df = pd.read_csv(path)
    y = df['is_anomaly'].values.tolist()
    X = df.drop(columns=['is_anomaly', 'subspaces'])
    dataFrame = add_noise(X, numberOfNoiseFeatures, True) 
    
    return y, dataFrame.values.tolist(), dataFrame.shape[1],  X.shape[1]