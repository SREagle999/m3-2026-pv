import descriptive_stats as ds
from hypothesis_testing import HypothesisTest
from scipy.stats import norm

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
            