import descriptive_stats as ds
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist
from scipy.stats import chi2

def z_test(x, y=None, mu=0, alternative="two-tailed", paired=False, alpha = 0.05):
    # Perform a z test with the specified arguments

    # Ensure valid alternative hypothesis
    if not alternative in ["two-tailed", "left", "right"]:
        print("Please select a valid alternative hypothesis")
        return None

    # Match each tail to their corresponding interpretation
    alt_hypoth = {"two-tailed" : "not equal to",
                  "left" : "less than",
                  "right" : "greater than"}
    
    # If two sample run two sample test
    if (y is not None) and (not paired):
        # Establish sample means and "population" standard deviations
        x1 = ds.mean(x)
        x2 = ds.mean(y)
        sigma1 = ds.std_dev(x)
        sigma2 = ds.std_dev(y)
        n1 = len(x)
        n2 = len(y)

        z = (x1 - x2) / ((((sigma1 ** 2) / n1) + ((sigma2 ** 2) / n2)) ** 0.5)  # Compute the test statistic
    # Otherwise use the procedure common to one sample and paired difference tests
    else:
        # If this is a paired test, we just set x to be the list of differences between x and y
        if paired:
            x = [i - j for i, j in zip(x, y)]   # Create list of paired differences
        
        # Establish sample mean and "population" standard deviation
        x_bar = ds.mean(x)
        sigma = ds.std_dev(x)
        n = len(x)

        z = (x_bar - mu) / (sigma / (n ** 0.5))    # Compute test statistic

        # Compute confidence interval
        z_crit = norm.ppf(alpha / 2)    # Find z-critical
        lower_bound = x_bar - (z_crit * (sigma / (n ** 0.5)))   # Calculate lower bound
        upper_bound = x_bar + (z_crit * (sigma / (n ** 0.5)))   # Calculate upper bound
    
    # Compute p value based on tail
    if alternative == "two-tailed":
        p = 2 * (1 - norm.cdf(abs(z)))   # probability of x greater than absolute value of z
    elif alternative == "left":
        p = norm.cdf(z) # probability of x less than z
    elif alternative == "right":
        p = 1 - norm.cdf(z) # probability of x greater than z

    # Print appropriate output
    if (y is not None) and (not paired):
        print("Two Sample Z-Test")
        print(f"z = {z} p = {p}")
        print(f"Alternative Hypothesis: Mean of sample 1 is {alt_hypoth[alternative]} the mean of sample 2")
        print("Sample Estimates:")
        print(f"Sample 1: mean = {x1} standard deviation = {sigma1} sample size = {n1}")
        print(f"Sample 2: mean = {x2} standard deviation = {sigma2} sample size = {n2}")
    elif (y is not None) and paired:
        print("Paired Difference Z-Test")
        print(f"z = {z} p = {p}")
        print(f"Alternative Hypothesis: True mean difference is {alt_hypoth[alternative]} 0")
        print("Sample Estimates:")
        print(f"mean = {x_bar} standard deviation = {sigma} sample size = {n}")
        print(f"{100 * (1 - alpha)}% Confidence Interval for true mean difference: [{lower_bound}, {upper_bound}]")
    else:
        print("One Sample Z-Test")
        print(f"z = {z} p = {p}")
        print(f"Alternative Hypothesis: True mean is {alt_hypoth[alternative]} {mu}")
        print("Sample Estimates:")
        print(f"mean = {x_bar} standard deviation = {sigma} sample size = {n}")
        print(f"{100 * (1 - alpha)}% Confidence Interval for true mean: [{lower_bound}, {upper_bound}]")

def t_test(x, y=None, mu=0, alternative="two-tailed", paired=False, alpha=0.05):
    # Perform a t test with the specified arguments

    # Ensure valid alternative hypothesis
    if not alternative in ["two-tailed", "left", "right"]:
        print("Please select a valid alternative hypothesis")
        return None

    # Match each tail to their corresponding interpretation
    alt_hypoth = {"two-tailed" : "not equal to",
                  "left" : "less than",
                  "right" : "greater than"}
    
    # If two sample run two sample test
    if (y is not None) and (not paired):
        # Establish sample means and standard deviations
        x1 = ds.mean(x)
        x2 = ds.mean(y)
        n1 = len(x)
        n2 = len(y)
        s_pooled = ((((n1 - 1) * (ds.std_dev(x) ** 2)) + ((n2 - 1) * (ds.std_dev(y) ** 2))) / (n1 + n2 - 2)) ** 0.5
        dof = n1 + n2 - 2

        t = (x1 - x2) / (s_pooled * (((1 / n1) + (1 / n2)) ** 0.5))  # Compute the test statistic
    # Otherwise use the procedure common to one sample and paired difference tests
    else:
        # If this is a paired test, we just set x to be the list of differences between x and y
        if paired:
            x = [i - j for i, j in zip(x, y)]   # Create list of paired differences
        
        # Establish sample mean and "population" standard deviation
        x_bar = ds.mean(x)
        s = ds.std_dev(x)
        n = len(x)
        dof = n - 1

        t = (x_bar - mu) / (s / (n ** 0.5))    # Compute test statistic

        # Compute confidence interval
        t_crit = norm.ppf(alpha / 2)    # Find t-critical
        lower_bound = x_bar - (t_crit * (s / (n ** 0.5)))   # Calculate lower bound
        upper_bound = x_bar + (t_crit * (s / (n ** 0.5)))   # Calculate upper bound
        
    # Compute p value based on tail
    if alternative == "two-tailed":
        p = 2 * (1 - t_dist.cdf(abs(t), dof))   # probability of x greater than absolute value of t
    elif alternative == "left":
        p = t_dist.cdf(t, dof) # probability of x less than t
    elif alternative == "right":
        p = 1 - t_dist.cdf(t, dof) # probability of x greater than t

    # Print appropriate output
    if (y is not None) and (not paired):
        print("Two Sample t-Test")
        print(f"t = {t} p = {p} dof = {dof}")
        print(f"Alternative Hypothesis: Mean of sample 1 is {alt_hypoth[alternative]} the mean of sample 2")
        print("Sample Estimates:")
        print(f"Sample 1: mean = {x1} sample size = {n1}")
        print(f"Sample 2: mean = {x2} sample size = {n2}")
        print(f"pooled standard deviation = {s_pooled}")

        # Return results dict
        return {"t" : t,
                "p" : p,
                "dof" : dof,
                "sample 1 mean" : x1,
                "sample 1 size" : n1,
                "sample 2 mean" : x2,
                "sample 2 size" : n2,
                "pooled s": s_pooled}
    elif (y is not None) and paired:
        print("Paired Difference t-Test")
        print(f"t = {t} p = {p} dof = {dof}")
        print(f"Alternative Hypothesis: True mean difference is {alt_hypoth[alternative]} 0")
        print("Sample Estimates:")
        print(f"mean = {x_bar} standard deviation = {s} sample size = {n}")
        print(f"{100 * (1 - alpha)}% Confidence Interval for true mean difference: [{lower_bound}, {upper_bound}]")

        # Return results dict
        return {"t" : t,
                "p" : p,
                "dof" : dof,
                "mean diff" : x_bar,
                "standard deviation" : s,
                "sample size" : n,
                "interval" : (lower_bound, upper_bound)}
    else:
        print("One Sample t-Test")
        print(f"t = {t} p = {p} dof = {dof}")
        print(f"Alternative Hypothesis: True mean is {alt_hypoth[alternative]} {mu}")
        print("Sample Estimates:")
        print(f"mean = {x_bar} standard deviation = {s} sample size = {n}")
        print(f"{100 * (1 - alpha)}% Confidence Interval for true mean: [{lower_bound}, {upper_bound}]")
          
        # Return results dict
        return {"t" : t,
                "p" : p,
                "dof" : dof,
                "mean diff" : x_bar,
                "standard deviation" : s,
                "sample size" : n,
                "interval" : (lower_bound, upper_bound)}
    
def chi_square(df, alpha=0.05):
    # Conduct a chi square goodness of fit or independence test

    dof = (len(df) - 1) * (len(df.columns) - 1) # Calculate degrees of freedom

    # Temporarily annotate original dataframe with row and column sums
    df.loc['Column Sum'] = df.apply(sum, axis=0)
    df['Row Sum'] = df.apply(sum, axis=1)

    n = sum(df['Row Sum'])  # Determine total number of units

    # Create the dataframe of expected values
    expected = pd.DataFrame()   # Initialize an empty dataframe
    # Iterate over each row of the dataframe
    for index, row in df.iterrows():
        if index == "Column Sum": continue  # Skip the "Column Sum" row
        # Iterate over every column of the dataframe
        for col in df.columns:
            if col == "Row Sum": continue   # Skip the "Row Sum" column
            expected.loc[index, col] = (row["Row Sum"] * df.loc['Column Sum', col]) / df.loc["Column Sum", "Row Sum"]   # Definition of expected value
    
    # Remove temporary annotations
    df = df.drop("Column Sum")
    df = df.drop("Row Sum", axis=1)

    # Conduct test and store results
    chi_square = sum((((df - expected) ** 2) / expected).apply(sum))    # Compute the test statistic
    p = 1 - chi2.cdf(chi_square, dof)   # chi2.cdf is left tailed
    crit = chi2.ppf(1 - alpha, dof) # chi2.ppf is left tailed

    # Print results
    print("Chi-Square Test for Independence")
    print(f"chi-square = {chi_square} p = {p} dof = {dof}")
    print("Expected:")
    print(expected)
    print("Observed:")
    print(df)
    print(f"Rejection Region for alpha = {alpha}: Chi-Square > {crit}")

    # Return results dict
    return {"chi-square" : chi_square,
            "p" : p,
            "dof" : dof,
            "expected" : expected,
            "crit" : crit}