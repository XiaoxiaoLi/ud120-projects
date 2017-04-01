#!/usr/bin/python
import numpy as np

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []
    errors = np.array(predictions[:,0]) - np.array(net_worths[:,0])
    sorted_errors_indices = np.argsort(errors)
    bottom_error_indices = sorted_errors_indices[:int(len(sorted_errors_indices)*0.9)]
    cleaned_data = [(ages[i], net_worths[i], errors[i]) for i in bottom_error_indices]
    
    return cleaned_data
