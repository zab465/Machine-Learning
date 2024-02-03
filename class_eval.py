#import random module
import random

def main():
    pass

#calculate the number of test data points
#randomly shuffle the data in list
#split the data into training and testing (lists)
def split_training_testing(data, test_percent):
    """
    Splits input (list) data into training and testing sets. 
    Test_percent determines how much data is allocated to testing
    (takes for input a number between 1-100).
    Returns two lists.
    """

    # Check input types and values
    assert isinstance(data, list), "Input should be a list."
    assert isinstance(test_percent, (int, float)), "Input should be a number (int or float)."
    assert len(data) > 0, "Input list cannot be empty."
    assert 0 <= test_percent <= 100, "Input number must be between 0 and 100"

    # Calculate the size of the test set
    test_size = int(len(data) * test_percent / 100)

    # Shuffle the data
    random.shuffle(data)

    # Split the data into training and testing sets
    training = data[test_size:]
    testing = data[:test_size]

    return training, testing


#defining a function that estimates values from a confusion matrix
def confusion_matrix(predicted, actual, positive_class):
    """
    Returns the number of true positives, false positives, 
    true negatives, and false negatives.
    """
    #ensure that the input is a list
    #ensure that the input lists are of the same length
    #ensure that the input lists are not empty
    #i use assertions because these are unlikely to be user errors (based on function description)
    assert isinstance(predicted, list), "Input 'predicted' must be a list"
    assert isinstance(actual, list), "Input 'actual' must be a list"
    assert len(predicted) > 0, "Input 'predicted' list is empty"
    assert len(actual) > 0, "Input 'actual' list is empty"
    assert len(predicted) == len(actual), "Input lists must be of the same length"
    
    #initialize TP, FP, TN, FN to 0
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    #iterate over the predicted and actual values
    #increment the counter for each of the four cases
    for pred, act in zip(predicted, actual):
        if pred == act == positive_class:
            tp += 1
        elif pred == positive_class and act != positive_class:
            fp += 1
        elif pred != positive_class and act == positive_class:
            fn += 1
        else:
            tn += 1

    return tp, fp, tn, fn

#function 1: accuracy
def accuracy(tp, fp, tn, fn):
    """
    Returns the accuracy of the model.
    Uses formula: (TP + TN) / (TP + TN + FP + FN)
    """
    #In case the user tries to hardcode negative values into the function 
    if not all(isinstance(i, int) for i in [tp, fp, tn, fn]):
        raise TypeError("All inputs must be integers")
    if not all(i >= 0 for i in [tp, fp, tn, fn]):
        raise ValueError("All inputs must be non-negative integers")
    try:
        return (tp + tn) / (tp + tn + fp + fn)
    except ZeroDivisionError:
        return float('nan')
    
#function 2: sensitivity
def sensitivity(TP, FN):
    """
    Returns the sensitivity of the model.
    Uses formula: TP / (TP + FN).
    """
    if not all(isinstance(i, int) for i in [TP, FN]):
        raise TypeError("All inputs must be integers")
    if not all(i >= 0 for i in [TP, FN]):
        raise ValueError("All inputs must be non-negative integers")
    try:
        return TP / (TP + FN)
    except ZeroDivisionError:
        return float('nan')
    
#function 3 for specificity
def specificity(tn, fp):
    """
    Returns the specificity of the model.
    Uses formula: TN / (TN + FP)
    """
    if not all(isinstance(i, int) for i in [tn, fp]):
        raise TypeError("All inputs must be integers")
    if not all(i >= 0 for i in [tn, fp]):
        raise ValueError("All inputs must be non-negative integers")
    try:
        return tn / (tn + fp)
    except ZeroDivisionError:
        return float('nan')

#function 4 for positive_predictive_value
def pos_pred_value(tp, fp):
    """
    Returns the positive predictive value of the model.
    Uses formula: TP / (TP + FP)
    """
    if not all(isinstance(i, int) for i in [tp, fp]):
        raise TypeError("All inputs must be integers")
    if not all(i >= 0 for i in [tp, fp]):
        raise ValueError("All inputs must be non-negative integers")
    #check for zero division error
    try:
        return tp / (tp + fp)
    except ZeroDivisionError:
        return float('nan')
    
#function 5 for negative_predictive_value
def neg_pred_value(tn, fn):
    """
    Returns the negative predictive value of the model.
    Uses formula: TN / (TN + FN).
    """
    if not all(isinstance(i, int) for i in [tn, fn]):
        raise TypeError("All inputs must be integers")
    if not all(i >= 0 for i in [tn, fn]):
        raise ValueError("All inputs must be non-negative integers")
    #check for zero division error
    try:
        return tn / (tn + fn)
    except ZeroDivisionError:
        return float('nan')


def print_eval_metrics(predicted, actual, positive_class):
    """
    Prints the accuracy, sensitivity, 
    specificity, positive predictive value, 
    and negative predictive value of the model.
    """
    #no need for assertions, since this calls the confusion_matrix function which already checks
    #the length of inputs, the types of inputs
    
    tp, fp, tn, fn = confusion_matrix(predicted, actual, positive_class)
    
    acc = accuracy(tp, tn, fp, fn) 
    sens = sensitivity(tp, fn)
    spec = specificity(tn, fp)
    ppv = pos_pred_value(tp, fp)
    npv = neg_pred_value(tn, fn)
    
    print(f"Accuracy: {acc}")
    print(f"Sensitivity: {sens}")
    print(f"Specificity: {spec}")
    print(f"Positive Predictive Value: {ppv}")
    print(f"Negative Predictive Value: {npv}")

if __name__ == '__main__': 
    main()