# mira.py
# -------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Mira implementation
import util
PRINT = True

class MiraClassifier:
    """
    Mira classifier.

    Note that the variable 'datum' in this code refers to a counter of features
    (not to a raw samples.Datum).
    """
    def __init__( self, legalLabels, max_iterations):
        self.legalLabels = legalLabels
        self.type = "mira"
        self.automaticTuning = False
        self.C = 0.001
        self.legalLabels = legalLabels
        self.max_iterations = max_iterations
        self.initializeWeightsToZero()

    def initializeWeightsToZero(self):
        "Resets the weights of each label to zero vectors"
        self.weights = {}
        for label in self.legalLabels:
            self.weights[label] = util.Counter() # this is the data-structure you should use

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        "Outside shell to call your method. Do not modify this method."

        self.features = trainingData[0].keys() # this could be useful for your code later...

        if (self.automaticTuning):
            Cgrid = [0.002, 0.004, 0.008]
        else:
            Cgrid = [self.C]

        return self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, Cgrid)

    def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, Cgrid):
        """
        This method sets self.weights using MIRA.  Train the classifier for each value of C in Cgrid,
        then store the weights that give the best accuracy on the validationData.

        Use the provided self.weights[label] data structure so that
        the classify method works correctly. Also, recall that a
        datum is a counter from features to values for those features
        representing a vector of values.
        """
        accsAndWeights = []
        for const in Cgrid:
            tempWeights = self.weights.copy()
            for iteration in range(self.max_iterations):
                print "Starting iteration ", iteration, "..."
                for i in range(len(trainingData)):
                    datum = trainingData[i]
                    
                    # get predicted label for data instance
                    predLabel = self.classify([datum], weights=tempWeights)[0]
                    actualLabel = trainingLabels[i]
                    if predLabel != actualLabel:
                        # predicted and actual labels differ
                        # so we need to update the weights
                        stepSize = updateStepSize(tempWeights, datum, actualLabel, predLabel, const)
                        # we don't want to mutate the original data instance
                        feature = util.Counter({key: val*stepSize for (key, val) in datum.items()})
                        # update the current weights vector
                        tempWeights[actualLabel] += feature
                        tempWeights[predLabel] -= feature

            # evalute accuracy using current weights vector
            accuracy = self.getAccuracy(validationData, validationLabels, tempWeights)
            accsAndWeights.append((accuracy, tempWeights))

        _, self.weights = max(accsAndWeights)


    def classify(self, data, weights=None):
        """
        Classifies each datum as the label that most closely matches the prototype vector
        for that label.  See the project description for details.

        Recall that a datum is a util.counter...
        """
        # if no default weights passed, use class instance of weights
        if not weights:
            weights = self.weights
        guesses = []
        for datum in data:
            vectors = util.Counter()
            for l in self.legalLabels:
                vectors[l] = weights[l] * datum
            guesses.append(vectors.argMax())
        return guesses


    def getAccuracy(self, validationData, validationLabels, w):
        '''
        Returns the validation accuracy for
        the given validation data and its associated
        labels using the given weights w
        '''
        # number of matches found & number of data instances
        count, n = 0, len(validationData)
        # classifiy data using the current weight vector
        predictions = self.classify(validationData, weights=w)
        for actual, prediction in zip(validationLabels, predictions):
            if prediction == actual:
                count += 1
        return count / float(n)


def updateStepSize(weights, feature, alabel, plabel, constant):
    """
    Returns the updated stepsize (t) given the weights and features
    vectores (util.Counter objects) and the actual label and predicted
    label (for a data instance). Formula can be seen on slide 26 of
    berkeley perceptron lecture (Lecture 22)
    """
    numerator = (weights[plabel] - weights[alabel]) * feature + 1.0
    denominator = feature * feature * 2.0
    return min(constant, numerator / denominator)
