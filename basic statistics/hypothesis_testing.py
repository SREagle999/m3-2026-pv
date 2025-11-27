import descriptive_stats as ds
import pandas as pd
from scipy.stats import norm
from scipy.stats import t as t_dist
from scipy.stats import chi2
from scipy.stats import f as f_dist

class HypothesisTest:
    def __init__(self, alternative="two-tailed", alpha=0.05):
        # Constructor method for the HypothesisTest class

        # Ensure valid alternative hypothesis
        if not alternative in ["two-tailed", "left", "right"]:
            print("Please select a valid alternative hypothesis")
            return None
        
        self.alternative = alternative
        self.alpha = alpha

        # Filled in by subclasses
        self.statistic = None
        self.p_value = None
        self.ci = None
        self.df = None
        self.summary_text = ""

    def compute(self):
        # Implemented by subclass

        raise NotImplementedError
    
    def summary(self):
        # Print stored summary text

        print(self.summary_text)

    def to_dict(self):
        # Convert test results to a dict

        return {"statistic" : self.statistic,
                "p" : self.p_value,
                "ci" : self.ci,
                "df" : self.df}
    
class ZTest(HypothesisTest):
    def __init__(self, x, y=None, mu=0, paired=False, alternative="two-tailed", alpha=0.05):
        # Constructor for z-test class

        super().__init__(alternative=alternative, alpha=alpha)  # Call constructor of superclass

        # Initialize instance variables
        self.x = x
        self.y = y
        self.mu = mu
        self.paired = paired

        self.compute()

    def compute(self):
        # Perform a z test with the specified arguments

        # Match each tail to their corresponding interpretation
        alt_hypoth = {"two-tailed" : "!=",
                    "left" : "<",
                    "right" : ">"}
        
        # If two sample run two sample test
        if (self.y is not None) and (not self.paired):
            # Establish sample means and "population" standard deviations
            x1 = ds.mean(self.x)
            x2 = ds.mean(self.y)
            sigma1 = ds.std_dev(self.x)
            sigma2 = ds.std_dev(self.y)
            n1 = len(self.x)
            n2 = len(self.y)
            se = ((((sigma1 ** 2) / n1) + ((sigma2 ** 2) / n2)) ** 0.5)

            self.statistic = (x1 - x2) / se # Compute the test statistic

            # Compute confidence interval
            z_crit = abs(norm.ppf(self.alpha / 2))   # Find z-critical
            lower_bound = (x1 - x2) - (z_crit * se)   # Calculate lower bound
            upper_bound = (x1 - x2) + (z_crit * se)   # Calculate upper bound
            self.ci = (lower_bound, upper_bound)
        # Otherwise use the procedure common to one sample and paired difference tests
        else:
            # If this is a paired test, we just set x to be the list of differences between x and y
            if self.paired:
                assert n1 == n2, "For paired test, both samples must be of same size"

                data = [i - j for i, j in zip(self.x, self.y)]  # Create list of paired differences
            else:
                data = self.x
            
            # Establish sample mean and "population" standard deviation
            x_bar = ds.mean(data)
            sigma = ds.std_dev(data)
            n = len(data)
            se = (sigma / (n ** 0.5))

            self.statistic = (x_bar - self.mu) / se   # Compute test statistic

            # Compute confidence interval
            z_crit = abs(norm.ppf(self.alpha / 2))   # Find z-critical
            lower_bound = x_bar - (z_crit * se)   # Calculate lower bound
            upper_bound = x_bar + (z_crit * se)   # Calculate upper bound
            self.ci = (lower_bound, upper_bound)
        
        # Compute p value based on tail
        if self.alternative == "two-tailed":
            self.p_value = 2 * (1 - norm.cdf(abs(self.statistic)))    # probability of x greater than absolute value of z
        elif self.alternative == "left":
            self.p_value = norm.cdf(self.statistic)   # probability of x less than z
        elif self.alternative == "right":
            self.p_value = 1 - norm.cdf(self.statistic)   # probability of x greater than z

        if (self.y is not None) and (not self.paired):
            self.summary_text = (
                f"Two Sample z-Test\n"
                f"Alternative Hypothesis: True mean of population 1 is {alt_hypoth[self.alternative]} true mean of population 2\n"
                f"z = {self.statistic:.4f}, p = {self.p_value:.4f}\n"
                f"mean 1 = {x1:.4f}, sd 1 = {sigma1:.4f}, n 1 = {n1}\n"
                f"mean 2 = {x2:.4f}, sd 2 = {sigma2:.4f}, n 2 = {n2}\n"
                f"{100*(1-self.alpha)}% CI for true difference in means: {self.ci}"
            )
        elif (self.y is not None) and (self.paired):
            self.summary_text = (
                f"Paired Difference z-Test\n"
                f"Alternative Hypothesis: True mean difference is {alt_hypoth[self.alternative]} {self.mu}\n"
                f"z = {self.statistic:.4f}, p = {self.p_value:.4f}\n"
                f"mean = {x_bar:.4f}, sd = {sigma:.4f}, n = {n}\n"
                f"{100*(1-self.alpha)}% CI for true mean difference: {self.ci}"
            )
        else:
            self.summary_text = (
                f"One Sample z-Test\n"
                f"Alternative Hypothesis: True mean is {alt_hypoth[self.alternative]} {self.mu}\n"
                f"z = {self.statistic:.4f}, p = {self.p_value:.4f}\n"
                f"mean = {x_bar:.4f}, sd = {sigma:.4f}, n = {n}\n"
                f"{100*(1-self.alpha)}% CI for true mean: {self.ci}"
            )
    
class TTest(HypothesisTest):
    def __init__(self, x, y=None, mu=0, paired=False, pooled=True, regression=True, alternative="two-tailed", alpha=0.05):
        # Constructor for t-test class

        super().__init__(alternative=alternative, alpha=alpha)  # Call constructor of superclass

        # If this is a t-Test for regression, we set the instance variables manually
        self.regression = regression
        if self.regression: return

        # Initialize instance variables
        self.x = x
        self.y = y
        self.mu = mu
        self.paired = paired
        self.pooled = pooled

        self.compute()

    def compute(self):
        # Perform a t test with the specified arguments

        # Match each tail to their corresponding interpretation
        alt_hypoth = {"two-tailed" : "!=",
                    "left" : "<",
                    "right" : ">"}
        
        # If two sample run two sample test
        if (self.y is not None) and (not self.paired):
            # Establish sample means and standard deviations
            x1 = ds.mean(self.x)
            x2 = ds.mean(self.y)
            n1 = len(self.x)
            n2 = len(self.y)

            # When running two sample t-tests with equal variance you use pooled standard deviation
            if self.pooled:
                s_pooled = ((((n1 - 1) * ds.variance(self.x)) + ((n2 - 1) * ds.variance(self.y))) / (n1 + n2 - 2)) ** 0.5
                self.df = n1 + n2 - 2
                se = (s_pooled * (((1 / n1) + (1 / n2)) ** 0.5))
            # Otherwise we use the Welch method
            else:
                se = ((ds.variance(self.x) / n1) + (ds.variance(self.y) / n2)) ** 0.5
                self.df = (se ** 4) / (((ds.variance(self.x) ** 2) / ((n1 ** 2) * (n1 - 1))) + ((ds.variance(self.y) ** 2) / ((n2 ** 2) * (n2 - 1))))

            self.statistic = (x1 - x2) / se    # Compute the test statistic

            # Compute confidence interval
            t_crit = abs(t_dist.ppf(self.alpha / 2, self.df))    # Find t-critical
            lower_bound = (x1 - x2) - (t_crit * se)   # Calculate lower bound
            upper_bound = (x1 - x2) + (t_crit * se)   # Calculate upper bound
            self.ci = (lower_bound, upper_bound)    # Store confidence interval
        # Otherwise use the procedure common to one sample and paired difference tests
        else:
            # If this is a paired test, we just set x to be the list of differences between x and y
            if self.paired:
                assert n1 == n2, "For paired test, both samples must be of same size"

                data = [i - j for i, j in zip(self.x, self.y)]  # Create list of paired differences
            else:
                data = self.x
            
            # Establish sample mean and "population" standard deviation
            x_bar = ds.mean(data)
            s = ds.std_dev(data)
            n = len(data)
            self.df = n - 1
            se = (s / (n ** 0.5))

            self.statistic = (x_bar - self.mu) / se   # Compute test statistic

            # Compute confidence interval
            t_crit = abs(t_dist.ppf(self.alpha / 2, self.df))    # Find t-critical
            lower_bound = x_bar - (t_crit * se)   # Calculate lower bound
            upper_bound = x_bar + (t_crit * se)   # Calculate upper bound
            self.ci = (lower_bound, upper_bound)    # Store confidence interval
            
        # Compute p value based on tail
        if self.alternative == "two-tailed":
            self.p_value = 2 * (1 - t_dist.cdf(abs(self.statistic), self.df))   # probability of x greater than absolute value of t
        elif self.alternative == "left":
            self.p_value = t_dist.cdf(self.statistic, self.df)  # probability of x less than t
        elif self.alternative == "right":
            self.p_value = 1 - t_dist.cdf(self.statistic, self.df)  # probability of x greater than t

        if (self.y is not None) and (not self.paired) and (self.pooled):
            self.summary_text = (
                f"Two Sample t-Test\n"
                f"Alternative Hypothesis: True mean of population 1 is {alt_hypoth[self.alternative]} true mean of population 2\n"
                f"t = {self.statistic:.4f}, p = {self.p_value:.4f}, df = {self.df}\n"
                f"mean 1 = {x1:.4f}, n 1 = {n1}\n"
                f"mean 2 = {x2:.4f}, n 2 = {n2}\n"
                f"pooled sd = {s_pooled:.4f}\n"
                f"{100*(1-self.alpha)}% CI for true difference in means: {self.ci}"
            )
        elif (self.y is not None) and (not self.paired) and (not self.pooled):
            self.summary_text = (
                f"Welch Two Sample t-Test\n"
                f"Alternative Hypothesis: True mean of population 1 is {alt_hypoth[self.alternative]} true mean of population 2\n"
                f"t = {self.statistic:.4f}, p = {self.p_value:.4f}, df = {self.df:.4f}\n"
                f"mean 1 = {x1:.4f}, sd 1 = {ds.std_dev(self.x)}, n 1 = {n1}\n"
                f"mean 2 = {x2:.4f}, sd 2 = {ds.std_dev(self.y)}, n 2 = {n2}\n"
                f"{100*(1-self.alpha)}% CI for true difference in means: {self.ci}"
            )
        elif (self.y is not None) and (self.paired):
            self.summary_text = (
                f"Paired Difference t-Test\n"
                f"Alternative Hypothesis: True mean difference is {alt_hypoth[self.alternative]} {self.mu}\n"
                f"t = {self.statistic:.4f}, p = {self.p_value:.4f}, df = {self.df}\n"
                f"mean = {x_bar:.4f}, sd = {s:.4f}, n = {n}\n"
                f"{100*(1-self.alpha)}% CI for true mean difference: {self.ci}"
            )
        else:
            self.summary_text = (
                f"One Sample t-Test\n"
                f"Alternative Hypothesis: True mean is {alt_hypoth[self.alternative]} {self.mu}\n"
                f"t = {self.statistic:.4f}, p = {self.p_value:.4f}, df = {self.df}\n"
                f"mean = {x_bar:.4f}, sd = {s:.4f}, n = {n}\n"
                f"{100*(1-self.alpha)}% CI for true mean: {self.ci}"
            )

class ChiSquareTest(HypothesisTest):
    def __init__(self, data, alpha=0.05):
        # Constructor for chi-square test class

        super().__init__(alpha=alpha)  # Call constructor of superclass

        # Initialize instance variables
        self.x = data

        self.compute()

    def compute(self):
        # Conduct a chi square goodness of fit or independence test

        self.df = (len(self.x) - 1) * (len(self.x.columns) - 1) # Calculate degrees of freedom

        # Temporarily annotate original dataframe with row and column sums
        self.x.loc['Column Sum'] = self.x.apply(sum, axis=0)
        self.x['Row Sum'] = self.x.apply(sum, axis=1)

        self.n = sum(self.x['Row Sum'])  # Determine total number of units

        # Create the dataframe of expected values
        self.expected = pd.DataFrame()   # Initialize an empty dataframe
        # Iterate over each row of the dataframe
        for index, row in self.x.iterrows():
            if index == "Column Sum": continue  # Skip the "Column Sum" row
            # Iterate over every column of the dataframe
            for col in self.x.columns:
                if col == "Row Sum": continue   # Skip the "Row Sum" column
                self.expected.loc[index, col] = (row["Row Sum"] * self.x.loc['Column Sum', col]) / self.n   # Definition of expected value
        
        # Remove temporary annotations
        self.x = self.x.drop("Column Sum")
        self.x = self.x.drop("Row Sum", axis=1)

        # Conduct test and store results
        self.statistic = sum((((self.x - self.expected) ** 2) / self.expected).apply(sum))    # Compute the test statistic
        self.p = 1 - chi2.cdf(self.statistic, self.df)   # chi2.cdf is left tailed
        self.crit = chi2.ppf(1 - self.alpha, self.df) # chi2.ppf is left tailed

        self.summary_text = (
                f"CHi-Square Test for Independence\n"
                f"Alternative Hypothesis: Factors are dependent\n"
                f"chi-square = {self.statistic:.4f}, p = {self.p_value:.4f}, df = {self.df}\n"
                f"expected values:\n"
                f"{self.expected}"
                f"Total units: {self.n}"
                f"Rejection region for alpha = {self.alpha} significance level: Chi-Square > {self.crit}"
            )
        
class VarianceTest(HypothesisTest):
    def __init__(self, x, y=None, sigma2=1, alternative="two-tailed", alpha=0.05):
        # Constructor for variance test class

        super().__init__(alternative=alternative, alpha=alpha)  # Call constructor of superclass

        self.x = x
        self.y = y
        self.sigma2 = sigma2

        self.compute()

    def compute(self):
        # Perform an inference test on variance with specified arguments

        # Match each tail to their corresponding interpretation
        alt_hypoth = {"two-tailed" : "!=",
                    "left" : "<",
                    "right" : ">"}

        # If two samples run two sample procedure
        if self.y is not None:
            # Establish constants
            var1 = ds.variance(self.x)
            var2 = ds.variance(self.y)
            n1 = len(self.x)
            n2 = len(self.y)
            nu1 = n1 - 1
            nu2 = n2 - 1
            self.df = (nu1, nu2)

            self.statistic = var1 / var2    # Calculate F

            # Determine p based off of tail
            if self.alternative == "two-tailed":
                self.p_value = 2 * min(1 - f_dist.cdf(self.statistic, nu1, nu2), f_dist.cdf(self.statistic, nu1, nu2))  # 2 * probability of x more extreme than F
            elif self.alternative == "left":
                self.p_value = f_dist.cdf(self.statistic, nu1, nu2) # Probability of x < F 
            elif self.alternative == "right":
                self.p_value = 1 - f_dist.cdf(self.statistic, nu1, nu2) # Probability of x > F
            
            # Compute confidence interval
            lower_bound = self.statistic * (1 / f_dist.ppf(1 - (self.alpha / 2), nu1, nu2)) # Calculate lower bound
            upper_bound = self.statistic * f_dist.ppf(1 - (self.alpha / 2), nu2, nu1)   # Calculate upper bound
            self.ci = (lower_bound, upper_bound)    # Store confidence interval

            self.summary_text = (
                f"Two Sample F-Test\n"
                f"Alternative Hypothesis: True ratio of population variances {alt_hypoth[self.alternative]} {self.sigma2}\n"
                f"F = {self.statistic:.4f}, p = {self.p_value:.4f}, df = {self.df}\n"
                f"variance 1 = {var1:.4f}, n 1 = {n1}\n"
                f"variance 2 = {var2:.4f}, n 2 = {n2}\n"
                f"{100*(1-self.alpha)}% CI for true ratio of variances: {self.ci}"
            )
        else:
            # Establish constants
            s = ds.variance(self.x)
            n = len(self.x)
            self.df = n - 1

            self.statistic = (self.df * s) / self.sigma2    # Calculate chi-square

            chi_crit_upper = chi2.ppf(1 - (self.alpha / 2), self.df)
            chi_crit_lower = chi2.ppf(self.alpha / 2, self.df)

            lower_bound = (self.df * s) / (chi_crit_upper)  # Calculate lower bound
            upper_bound = (self.df * s) / (chi_crit_lower)  # Calculate upper bound
            self.ci = (lower_bound, upper_bound)    # Store confidence interval

            # Compute p value based on tail
            if self.alternative == "two-tailed":
                self.p_value = 2 * min(chi2.cdf(self.statistic, self.df), 1 - chi2.cdf(self.statistic, self.df))    # probability of x greater than chi-square upper or less than chi-square lower
            elif self.alternative == "left":
                self.p_value = chi2.cdf(self.statistic, self.df)    # probability of x less than chi-square
            elif self.alternative == "right":
                self.p_value = 1 - chi2.cdf(self.p_value, self.df)  # probability of x greater than chi-square

            self.summary_text = (
                f"One Sample Chi-Square Test for Variance\n"
                f"Alternative Hypothesis: True population variance {alt_hypoth[self.alternative]} {self.sigma2}\n"
                f"Chi-Square = {self.statistic:.4f}, p = {self.p_value:.4f}, df = {self.df}\n"
                f"variance = {s:.4f}, n 1 = {n}\n"
                f"{100*(1-self.alpha)}% CI for true variance: {self.ci}"
            )