import descriptive_stats as ds
import pandas as pd
import math
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