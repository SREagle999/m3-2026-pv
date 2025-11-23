import math

def mean(x):
    # Find the average value of a list x

    # Initialize sum and number of NaN values to 0
    sum = 0
    num_nan = 0
    # Iterate over x
    for num in x:
        # If there is a NaN value, skip it
        if num != num:
            num_nan += 1    # Increment the variable that tracks the number of NaN values
            continue    
        sum += num  # Add current value to running total
    
    return sum / (len(x) - num_nan) # return the sum divided by the number of non-NaN values

def median(x):
    # Find the median value of a list x

    x = sorted([i for i in x if i == i]) # Sort the list in ascending order, excluding NaN values
    # If the list is empty, inform the user and do not return anything
    if len(x) == 0:
        print('List is empty!')
        return None
    # If the list has an even number of elements average the two middle values
    elif len(x) % 2 == 0:
        return (x[len(x) // 2] + x[(len(x) // 2) - 1]) / 2
    # If the list has an odd number of elements take the middle value
    else:
        return x[len(x) // 2]

def variance(x, sample=True):
    # Find the variance of a list x where x can be a sample or a population

    ss_diffs = 0    # Initialize the Sum of Squared Differences
    avg = mean(x)  # Determine the average of the list x
    num_nan = 0 # Initialize the number of NaN values to 0
    # Iterate over x
    for num in x:
        # If there is a NaN value skip it
        if num != num:
            num_nan += 1    # Increment variable that tracks the number of NaN values
            continue
        ss_diffs += (num - avg) ** (2)  # Add the squared error to the sum of squared errors

    n = len(x) - num_nan    # Count only the non-NaN values
    return (ss_diffs / (n - 1)) if sample else (ss_diffs / n)   # Divide the sum of squared differences by n - 1 for samples and n otherwise

def std_dev(x, sample=True):
    # Find the standard deviation of a list x where x can be a sample or a population

    return variance(x, sample) ** (0.5)    # Variance is the square of standard deviation

def mode(x):
    # Find the mode(s) of a list x

    vals = {}   # Initialize an empty dictionary
    # Iterate over x
    for num in x:
        # If there is a NaN value skip it
        if num != num:
            continue
        # If the current number is a key in vals, increment its value
        if num in vals:
            vals[num] += 1
        # Otherwise, set its value to 1
        else:
            vals[num] = 1

    maxval = max(vals.values()) # Establish max value in vals
    modes = [k for k, v in vals.items() if v == maxval] # Modes are any key with value equal to the max value

    # If modes has length greater than 1, it is multimodal and return the entire list
    if len(modes) > 1:
        print("Data is multimodal") # Explicitly inform the user that their data is multimodal
        return modes
    # Otherwise just return the first element, the only mode
    else:
        return modes[0]

def quantile(x, p, method="type 7"):
    # PLANNING TO ADD SUPPORT FOR OTHER QUANTILE METHODS
    # Determines the (100 * p)% quantile of the list x

    # Check that the specified method is supported. If not, let the user know and return nothing
    if not method in ["type 7"]:
        print("Please enter a valid method")
        return None
    
    x = sorted([i for i in x if not math.isnan(i)]) # Sort the given list in ascending order
    # If the list is empty, let the user know and return nothing
    if len(x) == 0:
        print("List is empty!")
        return None
    # If specified method is "type 7" linearly interpolate the quantile value
    elif method == "type 7":
        h = ((len(x) - 1) * p)  # Hypothetical index of the desired quantile

        # If the hypothetical index is a valid index, simply return the value at this index
        if h.is_integer():
            return x[int(h)]
        # Otherwise use linear interpolation between the closest integers above and below h to find the desired quantile
        else:
            return x[math.floor(h)] + ((h - math.floor(h)) * (x[math.ceil(h)] - x[math.floor(h)]))

def q1(x, method="tukey"):
    # PLANNING TO ADD SUPPORT FOR OTHER QUANTILE METHODS
    # Determines the first quartile of x using a specified method (Tukey "median of halves" by default)

    x = sorted([i for i in x if not math.isnan(i)]) # Sort the given list
    # If the list is empty, let the user know and return nothing
    if len(x) == 0:
        print("List is empty!")
        return None
    # If specified method is "tukey" find the median of the lower half of the list
    elif method == "tukey":
        return median(x[:(len(x) // 2)])
    # Otherwise use whatever quantile method to compute the 25th quantile
    else:
        return quantile(x, 0.25, method)

def q3(x, method="tukey"):
    # PLANNING TO ADD SUPPORT FOR OTHER QUANTILE METHODS
    # Determines the first quartile of x using a specified method (Tukey "median of halves" by default)

    x = sorted([i for i in x if not math.isnan(i)]) # Sort the given list
    # If the list is empty, let the user know and return nothing
    if len(x) == 0:
        print("List is empty!")
        return None
    # If specified method is "tukey" find the median of the upper half of the list
    elif method == "tukey":
        # If list length is even split with floor division
        if len(x) % 2 == 0:
            return median(x[((len(x) // 2)):])
        # If list length is odd split with ceiling of division
        else:
            return median(x[((len(x) // 2) + 1):])
    # Otherwise use whatever quantile method to compute the 25th quantile
    else:
        return quantile(x, 0.75, method)

def iqr(x, method="tukey"):
    # Determine the interquartile range of the list x

    # Return difference of 3rd and 1st quantiles computed with the specified method
    return q3(x, method) - q1(x, method)

def z_score(x, mean, std_dev):
    # Compute the z-score of a number given a mean and standard deviation

    # Return the number of standard deviations above or below the mean x is
    return (x - mean) / std_dev

def standardize(x):
    # Convert all values in a list to z-scores

    # Return a list of the z-scores computed for each element in x
    mu = mean(x)
    sigma = std_dev(x)
    return [z_score(num, mu, sigma) for num in x]

def find_outliers(x, criteria="1.5iqr", quant_method="tukey", z_crit=3):
    # Identify the outliers in a list

    # Check that the specified riteria is supported. If not, let the user know and return nothing
    if not criteria in ["1.5iqr", "z-score"]:
        print("Please enter a valid criteria")
        return None

    # If criteria is "1.5iqr" use 1.5IQR rule to identify outliers
    if criteria == "1.5iqr":
        inter_quart_range = iqr(x, quant_method)   # Find interquartile range
        upper_bound = q3(x, quant_method) + (1.5 * inter_quart_range)  # Compute upper bound
        lower_bound = q1(x, quant_method) - (1.5 * inter_quart_range)  # Compute lower bound

        return [i for i in x if (i > upper_bound) or (i < lower_bound)]
    # If criteria is "z-score" identify outliers as any value with a z-score with greater magnitude than z_crit
    elif criteria == "z-score":
        return [i for i in x if abs(z_score(i, mean(x), std_dev(x))) > z_crit]