# import xstream
import cppimport
xstream = cppimport.imp_from_filepath("src/xstream.cpp")

import argparse

from explanation import *
from util import load_data, convertMap


def main(args):
    alg = xstream.xStream(args.K, args.C, args.D, args.W)

    # Load the data
    y, X, dimensionality, originalFeatures = load_data(args.data, args.noise)

    # Fit sample per sample to xStream
    for x in X:
        sample = [f"{i}:{s}" for i, s in enumerate(x)]
        alg.fit(sample)

    # Get xStream results
    minDen = alg.getMinDensity()
    featureMap = alg.getFeatureProjectionMap()
    decFeatureMap = convertMap(featureMap)
    scores = alg.getScores()

    # Evaluate the detection performance
    threshold = detection_performance(y, scores)

    # Perform the Feature Count or Average Score technique
    if args.mode == "feature_count" or args.mode == "average_score":
        featureFrequencies = MDN_analysis(minDen, decFeatureMap, args.mode)
        explanation_performance(featureFrequencies, y, scores, threshold, args.mode, dimensionality, originalFeatures, args.true_positive)

    # Perform the Statistical Test technique
    elif args.mode == "statistical_test":
        relevantFeatures = MDN_statistical_analysis(minDen, y, decFeatureMap, scores, threshold, dimensionality, args.pValue, args.test, args.true_positive)
        explanation_performance_stat(relevantFeatures, y, dimensionality, originalFeatures)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--K", type = int, default = 100, help = "Number of projection subspaces to be used by xStream")
    parser.add_argument("--C", type = int, default = 25, help = "Number of half-space chains to be used by xStream")
    parser.add_argument("--D", type = int, default = 10, help = "Depth of each half-space chains to be used by xStream")
    parser.add_argument("--W", type = int, default = 128, help = "Window size to be used by xStream")

    parser.add_argument("-d", "--data", type = str, default = "datasets/breast_cancer_lof.csv", help = "Path of the dataset on which xStream will be used")
    parser.add_argument("-m", "--mode", type = str, choices = ["feature_count", "average_score", "statistical_test"], default = "feature_count", help = "Anomaly explanation technique to be applied")
    parser.add_argument("-n", "--noise", type = int, default = 15, help = "Number of random noise features to be added to the original data")
    parser.add_argument("-tp", "--true_positive", action = "store_true", help = "Will use only true positives to evaluate the explanations performance, instead of all detected anomalies")

    parser.add_argument("-t", "--test", type = str, choices = ["t", "ks"], default = "ks", help = "Choose the statistical test to be performed")
    parser.add_argument("-p", "--pValue", type = float, default = 0.05, help = "Define the limit pValue on the statistical test")

    args = parser.parse_args()

    main(args)