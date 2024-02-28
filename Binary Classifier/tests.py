import unittest
import math
import unittest
import class_eval

#separate class for each function
#i don't test the print function, because it relies on the other functions, and is a simple print command
class TestSplitTrainingTesting(unittest.TestCase):
    def setUp(self):
        """
        Create a list of 100 integers from 0 to 99.
        """
        #a set of numbers has to be initialized 
        #so that the other tests can run 
        self.data = list(range(100))
    
    def test_split_training_empty_list(self):
        """
        Test that an AssertionError is raised when an empty list is provided.
        """
        with self.assertRaises(AssertionError, msg = "Input list cannot be empty."):
            class_eval.split_training_testing([], 30)

    def test_split_training_testing_50_50(self):
        """
        Test a 50/50 split.
        """
        training, testing = class_eval.split_training_testing(self.data, 50)
        self.assertEqual(len(training), 50, "Training set size should be 50 for 50% test split")
        self.assertEqual(len(testing), 50, "Testing set size should be 50 for 50% test split")

    def test_split_training_testing_90_10(self):
        """
        Test a 90/10 split.
        This is an imbalanced split
        """
        #this tests random splits of data
        training, testing = class_eval.split_training_testing(self.data, 10)
        self.assertEqual(len(training), 90, "Training set size should be 90 for 10% test split")
        self.assertEqual(len(testing), 10, "Testing set size should be 10 for 10% test split")
    
    def test_split_training_testing_less_than_zero(self):
        """
        Test that an AssertionError is raised when test_percent is less than 0.
        """
        #check that itraises an error if an invalid number is inputted (less than 0)
        with self.assertRaises(AssertionError, msg = "Input number must be between 0 and 100"):
            class_eval.split_training_testing(self.data, -1)

    def test_split_training_testing_greater_than_100(self):
        """
        Test that an AssertionError is raised when test_percent is greater than 100.
        """
        #check that itraises an error if an invalid number (100+) is inputted
        with self.assertRaises(AssertionError, msg = "Input number must be between 0 and 100"):
            class_eval.split_training_testing(self.data, 101)
    
    def test_split_training_testing_100(self):
        """
        Test that when test_percent is 100, all data is allocated to the testing set.
        """
        training, testing = class_eval.split_training_testing(self.data, 100)
        self.assertEqual(len(testing), len(self.data), "Training set size should be equal to the size of the input data")
        self.assertEqual(len(training), 0, "Testing set size should be 0 for 100% test split")
    
    def test_split_training_testing_non_number(self):
        """
        Test that a TypeError is raised when test_percent is not a float or an integer.
        """
        with self.assertRaises(AssertionError, msg="Input 'test_percent' must be a float or an integer"):
            class_eval.split_training_testing(self.data, 'a')


class TestConfusionMatrix(unittest.TestCase):
    #i make tests for different natural partitions (all true positives, all true negatives, etc.)
    def test_confusion_matrix_true_positives(self):
        """
        Test a list with all true positives.
        """
        predicted = [1, 1, 1, 1]
        actual = [1, 1, 1, 1]
        positive_class = 1
        output = class_eval.confusion_matrix(predicted, actual, positive_class)
        output_expected = (4, 0, 0, 0)
        self.assertEqual(output_expected, output, "The confusion matrix for all true positives should be (4, 0, 0, 0)")

    def test_confusion_matrix_true_negatives(self):
        """
        Test a list with all true negatives.
        """
        predicted = [0, 0, 0, 0]
        actual = [0, 0, 0, 0]
        positive_class = 1
        output = class_eval.confusion_matrix(predicted, actual, positive_class)
        output_expected = (0, 0, 4, 0)
        self.assertEqual(output_expected, output, "The confusion matrix for all true negatives should be (0, 0, 4, 0)")

    def test_confusion_matrix_false_positives(self):
        """
        Test a list with all false positives.
        """
        predicted = [1, 1, 1, 1]
        actual = [0, 0, 0, 0]
        positive_class = 1
        output = class_eval.confusion_matrix(predicted, actual, positive_class)
        output_expected = (0, 4, 0, 0)
        self.assertEqual(output_expected, output, "The confusion matrix for all false positives should be (0, 4, 0, 0)")

    def test_confusion_matrix_false_negatives(self):
        """
        Test a list with all false negatives.
        """
        predicted = [0, 0, 0, 0]
        actual = [1, 1, 1, 1]
        positive_class = 1
        output = class_eval.confusion_matrix(predicted, actual, positive_class)
        output_expected = (0, 0, 0, 4)
        self.assertEqual(output_expected, output, "The confusion matrix for all false negatives should be (0, 0, 0, 4)")
    
    
    def test_confusion_matrix_empty_list(self):
        """
        Test that an AssertionError is raised when an empty list is provided.
        """
        #check that it raises an error if an empty list is inputted
        with self.assertRaises(AssertionError, msg = "Input lists cannot be empty"):
            class_eval.confusion_matrix([], [], 1)
    
    def test_confusion_matrix_uneven_lists(self):
        """
        Test that an AssertionError is raised when the predicted and actual lists have different lengths.
        """
        with self.assertRaises(AssertionError, msg="Predicted and actual lists must have the same length"):
            class_eval.confusion_matrix([1, 1, 1], [1, 1], 1)
    
    def test_confusion_matrix_non_list_input(self):
        """
        Test that a TypeError is raised when either the 'predicted' or 'actual' inputs are not lists.
        """
        with self.assertRaises(AssertionError, msg="Input 'predicted' must be a list"):
            class_eval.confusion_matrix('a', [1, 1, 1], 1)

        with self.assertRaises(AssertionError, msg="Input 'actual' must be a list"):
            class_eval.confusion_matrix([1, 1, 1], 'b', 1)



class TestAccuracy(unittest.TestCase):
    #i begin by testing the natural partitions
    def test_accuracy_all_correct(self):
        """
        Test a list with all correct predictions.
        """
        inputted = [50, 50, 0, 0]
        output = class_eval.accuracy(*inputted)
        output_expected = 0.5
        self.assertEqual(output_expected, output, "The accuracy for TP=50, TN=50, FP=0, FN=0 should be 1.0")

    def test_accuracy_half_correct(self):
        """
        Test a list with half correct predictions.
        """
        inputted = [50, 50, 50, 50]
        output = class_eval.accuracy(*inputted)
        output_expected = 0.5
        self.assertEqual(output_expected, output, "The accuracy for TP=50, TN=50, FP=50, FN=50 should be 0.5")

    def test_accuracy_no_positives(self):
        """
        Test a list with no positive predictions.
        """
        inputted = [0, 50, 0, 50]
        output = class_eval.accuracy(*inputted)
        self.assertEqual(output, 0.0, "The accuracy for TP=0, TN=50, FP=0, FN=50 should be 0.0")

    def test_accuracy_no_negatives(self):
        """
        Test a list with no negative predictions.
        """
        inputted = [50, 0, 50, 0]
        output = class_eval.accuracy(*inputted)
        self.assertEqual(output, 1.0, "The accuracy for TP=50, TN=0, FP=50, FN=0 should be 1.0")

    def test_accuracy_no_correct_predictions(self):
        """
        Test a list with no correct predictions.
        """
        inputted = [0, 0, 50, 50]
        output = class_eval.accuracy(*inputted)
        self.assertEqual(output, 0.5, "The accuracy for TP=0, TN=0, FP=50, FN=50 should be 0.5")
    
    def test_accuracy_mixed(self):
        """
        Test a list with a random mix of correct and incorrect predictions.
        """
        inputted = [40, 30, 10, 20]
        output = class_eval.accuracy(*inputted)
        output_expected = 0.5
        self.assertEqual(output, output_expected, "The accuracy for TP=40, FP = 20, TN=10,FN=20 should be 0.5")
    
    def test_accuracy_edge_cases(self):
        """
        Test a list with edge case (extreme) values.
        """
        inputted = [10**6, 10**6, 10**6, 10**6]
        output = class_eval.accuracy(*inputted)
        self.assertEqual(output, 0.5, "The accuracy for very large values should be 0.5")

    #ensuring the function responds to errors correctly
    def test_accuracy_division_by_zero(self):
        """
        Test a list with division by zero.
        Ensure that the function returns NaN.
        """
        tp = tn = fp = fn = 0
        output = class_eval.accuracy(tp, tn, fp, fn)
        self.assertTrue(math.isnan(output), "The accuracy for TP=0, TN=0, FP=0, FN=0 should be NaN")
    
    def test_accuracy_non_integer(self):
        """
        Test that a TypeError is raised when a non-integer is provided.
        """
        with self.assertRaises(TypeError, msg = "All inputs must be integers"):
            class_eval.accuracy('a', 1, 1, 1)

    def test_accuracy_negative_integer(self):
        """
        Test that a ValueError is raised when a negative integer is provided.
        """
        with self.assertRaises(ValueError):
            class_eval.accuracy(-1, 1, 1, 1)

class TestSensitivity(unittest.TestCase):
    #start with natural partitions
    def test_sensitivity_true_positives(self):
        """
        Test a list with all true positives 
        and no false negatives.
        """
        tp = 50
        fn = 0
        output = class_eval.sensitivity(tp, fn)
        output_expected = 1.0
        self.assertEqual(output_expected, output, "The sensitivity for TP=50, FN=0 should be 1.0")

    def test_sensitivity_half_true_positives(self):
        """
        Test a list with half correct predictions.
        """
        tp = 50
        fn = 50
        output = class_eval.sensitivity(tp, fn)
        output_expected = 0.5
        self.assertEqual(output_expected, output, "The sensitivity for TP=50, FN=50 should be 0.5")

    def test_sensitivity_no_true_positives(self):
        """
        Test a list with no true positives
        and all false negatives.
        """
        tp = 0
        fn = 50
        output = class_eval.sensitivity(tp, fn)
        output_expected = 0.0
        self.assertEqual(output_expected, output, "The sensitivity for TP=0, FN=50 should be 0.0")

    def test_sensitivity_no_positives(self):
        """
        Test a list with no values.
        """
        tp = 0
        fn = 0
        output = class_eval.sensitivity(tp, fn)
        self.assertTrue(math.isnan(output), "Sensitivity cannot be calculated when TP=0 and FN=0")
    
    def test_sensitivity_boundary_cases(self):
        """
        Test a list with edge case (extreme) values.
        """
        #boundary case with possibly different behavior (extreme values)
        tp = 10**6
        fn = 10**6
        output = class_eval.sensitivity(tp, fn)
        self.assertEqual(output, 0.5, "The sensitivity for extreme values should be 0.5")


    def test_sensitivity_random_cases(self):
        """
        Test a list with random values.
        """
        tp = 30
        fn = 20
        output = class_eval.sensitivity(tp, fn)
        output_expected = 0.6
        self.assertEqual(output, output_expected, "The sensitivity for TP=30, FN=20 should be 0.6")
    
    #check that they respond to errors correctly
    def test_sensitivity_division_by_zero(self):
        """
        Test a list with division by zero.
        Ensure that the function returns NaN.
        """
        tp = fn = 0
        output = class_eval.sensitivity(tp, fn)
        self.assertTrue(math.isnan(output), "The sensitivity for TP=0, FN=0 should be NaN")
    
    def test_sensitivity_non_integer(self):
        """
        Test that a TypeError is raised when a non-integer is provided.
        """
        with self.assertRaises(TypeError):
            class_eval.sensitivity('a', 1)

    def test_sensitivity_negative_integer(self):
        """
        Test that a ValueError is raised when a negative integer is provided.
        """
        with self.assertRaises(ValueError):
            class_eval.sensitivity(-1, 1)

class TestSpecificity(unittest.TestCase):
    
    def test_specificity_all_true_negatives(self):
        """
        Test a list with all true negatives
        and no false positives.
        """
        tn = 50
        fp = 0
        output = class_eval.specificity(tn, fp)
        output_expected = 1.0
        self.assertEqual(output_expected, output, "The specificity for TN=50, FP=0 should be 1.0")

    def test_specificity_half_true_negatives(self):
        """
        Test a list with half correct predictions.
        """
        tn = 50
        fp = 50
        output = class_eval.specificity(tn, fp)
        output_expected = 0.5
        self.assertEqual(output_expected, output, "The specificity for TN=50, FP=50 should be 0.5")

    def test_specificity_no_true_negatives(self):
        """
        Test a list with no true negatives
        and all false positives.
        """
        tn = 0
        fp = 50
        output = class_eval.specificity(tn, fp)
        output_expected = 0.0
        self.assertEqual(output_expected, output, "The specificity for TN=0, FP=50 should be 0.0")

    def test_specificity_no_negatives(self):
        """
        Test a list with no negative predictions.
        """
        tn = 0
        fp = 0
        output = class_eval.specificity(tn, fp)
        self.assertTrue(math.isnan(output), "The specificity for TN=0, FP=0 should be NaN")
    
    def test_specificity_edge_cases(self):
        """
        Test a list with edge case (extreme) values.
        """
        tn = 10**6
        fp = 10**6
        output = class_eval.specificity(tn, fp)
        self.assertEqual(output, 0.5, "The specificity for very large values should be 0.5")

    def test_specificity_boundary_cases(self):
        """
        Test a list with boundary case values.
        """
        tn = 500
        fp = 500
        output = class_eval.specificity(tn, fp)
        self.assertAlmostEqual(output, 0.5, 2, "The specificity for TN=500, FP=500 should be approximately 0.5")

    def test_specificity_random_cases(self):
        """
        Test random values.
        """
        # Randomized test case
        tn = 30
        fp = 20
        output = class_eval.specificity(tn, fp)
        output_expected = 0.6
        self.assertEqual(output, output_expected, "The specificity for TN=30, FP=20 should be 0.6")
    
    def test_specificity_division_by_zero(self):
        """
        Test a list with division by zero.
        Ensure that the function returns NaN.
        """
        tn = fp = 0
        output = class_eval.specificity(tn, fp)
        self.assertTrue(math.isnan(output), "The specificity for TN=0, FP=0, should be NaN")
    
    def test_specificity_non_integer(self):
        """
        Test that a TypeError is raised when a non-integer is provided.
        """
        with self.assertRaises(TypeError):
            class_eval.specificity('a', 1)

    def test_specificity_negative_integer(self):
        """
        Test that a TypeError is raised when a negative integer is provided.
        """
        with self.assertRaises(ValueError):
            class_eval.specificity(-1, 1)


class TestPos_pred_value(unittest.TestCase):
    def test_pos_pred_true_pos(self):
        """
        Test a list with all true positives
        and no false positives.
        """
        tp = 50
        fp = 0
        output = class_eval.pos_pred_value(tp, fp)
        output_expected = 1.0
        self.assertEqual(output_expected, output, "The positive predictive value for TP=50, FP=0 should be 1.0")

    def test_pos_pred_half_pos(self):
        """
        Test a list with half correct predictions.
        """
        tp = 50
        fp = 50
        output = class_eval.pos_pred_value(tp, fp)
        output_expected = 0.5
        self.assertEqual(output_expected, output, "The positive predictive value for TP=50, FP=50 should be 0.5")

    def test_pos_pred_no_pos(self):
        """
        Test a list with no true positives
        and all false positives.
        """
        tp = 0
        fp = 50
        output = class_eval.pos_pred_value(tp, fp)
        output_expected = 0.0
        self.assertEqual(output_expected, output, "The positive predictive value for TP=0, FP=50 should be 0.0")

    def test_pos_pred_no_value(self):
        """
        Test a list with no positive predictions.
        """
        tp = 0
        fp = 0
        output = class_eval.pos_pred_value(tp, fp)
        self.assertTrue(math.isnan(output), "The positive predictive value for TP=0, FP=0 should be NaN")
    
    def test_pos_pred_edge(self):
        """
        Test a list with edge case (extreme) values.
        """
        tp = 10**6
        fp = 10**6
        output = class_eval.pos_pred_value(tp, fp)
        self.assertEqual(output, 0.5, "The positive predictive value for very large values should be 0.5")
    
    def test_pos_pred_value_random(self):
        """
        Test random values.
        """
        tp = 30
        fp = 20
        output = class_eval.pos_pred_value(tp, fp)
        output_expected = 0.6
        self.assertEqual(output, output_expected, "The positive predictive value for TP=30, FP=20 should be 0.6")
    
    def test_pos_pred_division_by_zero(self):
        """
        Test a list with division by zero.
        Ensure that the function returns NaN.
        """
        tp = fp = 0
        output = class_eval.pos_pred_value(tp, fp)
        self.assertTrue(math.isnan(output), "The positive predicted for TP=0, FP=0, should be NaN")
    
    def test_neg_pred_value_non_integer(self):
        """
        Test that a TypeError is raised when a non-integer is provided.
        """
        with self.assertRaises(TypeError):
            class_eval.neg_pred_value('a', 1)

    def test_neg_pred_value_negative_integer(self):
        """
        Test that a ValueError is raised when a negative integer is provided.
        """
        with self.assertRaises(ValueError):
            class_eval.neg_pred_value(-1, 1)


class TestNeg_pred_value(unittest.TestCase):
    #starting with natural partitions
    def test_neg_pred_true_neg(self):
        """
        Test a list with all true negatives
        and no false negatives.
        """
        tn = 50
        fn = 0
        output = class_eval.neg_pred_value(tn, fn)
        output_expected = 1.0
        self.assertEqual(output_expected, output, "The negative predictive value for TN=50, FN=0 should be 1.0")

    def test_neg_pred_value_half_true_neg(self):
        """
        Test a list with half correct predictions.
        """
        tn = 50
        fn = 50
        output = class_eval.neg_pred_value(tn, fn)
        output_expected = 0.5
        self.assertEqual(output_expected, output, "The negative predictive value for TN=50, FN=50 should be 0.5")

    def test_neg_pred_no_true_neg(self):
        """
        Test a list with no true negatives
        and all false negatives.
        """
        tn = 0
        fn = 50
        output = class_eval.neg_pred_value(tn, fn)
        output_expected = 0.0
        self.assertEqual(output_expected, output, "The negative predictive value for TN=0, FN=50 should be 0.0")

    def test_neg_pred_no_neg(self):
        """
        Test a list with no negative predictions.
        """
        tn = 0
        fn = 0
        output = class_eval.neg_pred_value(tn, fn)
        self.assertTrue(math.isnan(output), "The negative predictive value for TN=0, FN=0 should be NaN")

    def test_neg_pred_boundary(self):
        """
        Test a list with boundary case values.
        """
        tn = 500
        fn = 500
        output = class_eval.neg_pred_value(tn, fn)
        self.assertAlmostEqual(output, 0.5, 2, "The negative predictive value for TN=500, FN=500 should be approximately 0.5")

    def test_neg_pred_random(self):
        """
        Test a list with random values.
        """
        tn = 30
        fn = 20
        output = class_eval.neg_pred_value(tn, fn)
        output_expected = 0.6
        self.assertEqual(output, output_expected, "The negative predictive value for TN=30, FN=20 should be 0.6")

    def test_neg_pred_edge(self):
        """
        Test a list with edge case (extreme) values.
        """
        tn = 10**6
        fn = 10**6
        output = class_eval.neg_pred_value(tn, fn)
        self.assertEqual(output, 0.5, "The negative predictive value for very large values should be 0.5")

    #testing for error responses
    def test_neg_pred_division_by_zero(self):
        """
        Test a list with division by zero.
        Ensure that the function returns NaN.
        """
        tn = fn = 0
        output = class_eval.neg_pred_value(tn, fn)
        self.assertTrue(math.isnan(output), "The negative predicted value for TN=0, FN=0, should be NaN")
    
    def test_neg_pred_value_non_integer(self):
        """
        Test that a TypeError is raised when a non-integer is provided.
        """
        with self.assertRaises(TypeError, msg="All inputs must be integers."):
            class_eval.neg_pred_value('a', 1)

    def test_neg_pred_value_negative_integer(self):
        """
        Test that a ValueError is raised when a negative integer is provided.
        """
        with self.assertRaises(ValueError, msg="All inputs must be non-negative integers."):
            class_eval.neg_pred_value(-1, 1)
        
if __name__ == '__main__':
    unittest.main()