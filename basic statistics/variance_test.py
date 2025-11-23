import math
import descriptive_stats as ds
from hypothesis_testing import HypothesisTest
from scipy.stats import chi2
from scipy.stats import f as f_dist

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