import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve, precision_score
from scipy.stats import ks_2samp, ttest_ind


def detection_performance(groundTruth, scores):
    anomalyscores = -1.0 * np.array(scores)
    ap = average_precision_score(groundTruth, anomalyscores) 
    auc = roc_auc_score(groundTruth, anomalyscores)
    print(f"Detection AP: {ap*100:.1f}%")
    print(f"Detection AUC: {auc*100:.1f}%\n")

    # Find the threshold to transform the scores into labels
    precisions, recalls, thresholds = precision_recall_curve(groundTruth, anomalyscores)
    f1 = 2 * ((np.array(precisions) * np.array(recalls)) / (np.array(precisions) + np.array(recalls)))

    return thresholds[f1.tolist().index(max(f1))]
    


def MDN_analysis(minimumDensityNodesInfo, featureMap, mode):
    results = []

    for sample in minimumDensityNodesInfo:
        featureFrequency = {}
        featureCounts = {}

        for chain in sample:
            depth, subspace, score = chain
            features = featureMap[subspace]

            for feature in features:
                if feature in featureFrequency.keys():
                    featureFrequency[feature] += score
                    featureCounts[feature] += 1
                else:
                    featureFrequency[feature] = score
                    featureCounts[feature] = 1

        if mode == "feature_count":
            result = featureCounts
        elif mode == "average_score":
            result = {k: featureFrequency[k]/featureCounts[k] for k in featureFrequency.keys() & featureCounts}

        result = dict(sorted(result.items(), key = lambda item: item[0]))
        results.append(result)

    return results



def MDN_statistical_analysis(minimumDensityNodesInfo, y, featureMap, scores, threshold, dimensionality, pThreshold, test, useTruePositive):
    havesFeat = {}
    dontHavesFeat = {}

    anomalyscores = -1.0 * np.array(scores)

    for id, sample in enumerate(minimumDensityNodesInfo):

        if useTruePositive:
            condition = anomalyscores[id] > threshold and y[id] == 1
        else:
            condition = anomalyscores[id] > threshold

        if condition:
            haveFeat = {}
            dontHaveFeat = {}

            for Feat in range(dimensionality):
                for chain in sample:
                    depth, subspace, score = chain
                    features = featureMap[subspace]

                    if Feat in features:
                        if Feat in haveFeat.keys():
                            haveFeat[Feat].append(score)
                        else:
                            haveFeat[Feat] = [score]

                    else:
                        if Feat in dontHaveFeat.keys():
                            dontHaveFeat[Feat].append(score)
                        else:
                            dontHaveFeat[Feat] = [score]

            havesFeat[id] = haveFeat
            dontHavesFeat[id] = dontHaveFeat


    pValues = {}
    tStats = {}

    for id in havesFeat.keys():
        pValue = {}
        tStat = {}

        for Feat in range(dimensionality):
            if Feat in havesFeat[id].keys() and Feat in dontHavesFeat[id].keys():
                if test == "ks":
                    tStat[Feat], pValue[Feat] = ks_2samp(havesFeat[id][Feat], dontHavesFeat[id][Feat], alternative = 'greater')

                elif test == "t":
                    tStat[Feat], pValue[Feat] = ttest_ind(havesFeat[id][Feat], dontHavesFeat[id][Feat], equal_var = False, alternative = 'greater')

        pValues[id] = pValue
        tStats[id] = tStat


    dependent = {}

    for P in pValues.keys():
        for k, p in pValues[P].items():
            if p < pThreshold:
                if P in dependent.keys():
                    dependent[P].append(k)
                else:
                    dependent[P] = [k]

    return dependent



def explanation_performance(featureFrequencies, y, scores, threshold, mode, dimensionality, originalFeatures, useTruePositive):

    anomalyscores = -1.0 * np.array(scores)

    relevantFeatures = [i for i in range(originalFeatures)]
    ranking = [0] * dimensionality
    
    mAP = 0
    num = 0
    ground = [0] * dimensionality

    for r in range(originalFeatures):
        ground[r] = 1

    TP = 0

    for id, freq in enumerate(featureFrequencies):

        if useTruePositive:
            condition = anomalyscores[id] > threshold and y[id] == 1
        else:
            condition = anomalyscores[id] > threshold

        if condition:

            for f in relevantFeatures:
                if f in freq.keys():
                    if mode == "feature_count":
                        ranking[list(dict(sorted(freq.items(), key = lambda item: item[1], reverse=True)).keys()).index(f)] += 1 
                    elif mode == "average_score":
                        ranking[list(dict(sorted(freq.items(), key = lambda item: item[1])).keys()).index(f)] += 1 

            sample = [0] * dimensionality

            for f in list(dict(sorted(freq.items(), key = lambda item: item[1], reverse=True)).keys())[:originalFeatures]:
                sample[f] = 1

            mAP += average_precision_score(ground, sample)
            num += 1

            if y[id] == 1:
                TP += 1


    print(f"{num} identified anomalies | {TP} true positives | {y.count(1) - TP} false negatives\n")
    print(f"Final feature ranking: {ranking}\n")
    print(f"Explanation mAP: {mAP*100/num:.1f}%")

    # ap = average_precision_score(ground, ranking) 
    # auc = roc_auc_score(ground, ranking)
    # print(f"Explanation ranking AP: {ap*100:.1f}%")
    # print(f"Explanation ranking AUC: {auc*100:.1f}%")



def explanation_performance_stat(relevantFeatures, y, dimensionality, originalFeatures):   

    mP = 0
    mAP = 0
    coun = 0
    TP = 0
    gt = [0] * dimensionality

    for i in range(originalFeatures):
        gt[i] = 1

    for id, features in relevantFeatures.items():
        vec = [0] * dimensionality

        for f in features:
            vec[f] = 1

        mP += precision_score(gt, vec)
        mAP += average_precision_score(gt, vec)
        coun += 1

        if y[id] == 1:
            TP += 1


    print(f"{coun} identified anomalies | {TP} true positives | {y.count(1) - TP} false negatives\n")

    print(f"Explanation mP: {mP*100/coun:.1f}%")
    print(f"Explanation mAP: {mAP*100/coun:.1f}%")