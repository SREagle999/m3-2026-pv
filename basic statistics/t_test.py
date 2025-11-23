import descriptive_stats as ds
from hypothesis_testing import HypothesisTest
from scipy.stats import t as t_dist

class TTest(HypothesisTest):
    def __init__(self, x, y=None, mu=0, paired=False, pooled=True, alternative="two-tailed", alpha=0.05):
        # Constructor for t-test class

        super().__init__(alternative=alternative, alpha=alpha)  # Call constructor of superclass

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