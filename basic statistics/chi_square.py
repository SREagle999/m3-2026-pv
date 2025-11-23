import pandas as pd
from hypothesis_testing import HypothesisTest
from scipy.stats import chi2

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