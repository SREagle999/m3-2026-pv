import descriptive_stats as ds
from anova import *
from hypothesis_testing import *
from scipy.stats import t as t_dist
from scipy.stats import f as f_dist

class RegressionModel:
    def __init__(self):
        # Constructor for the regression model class

        # Instance variables will be declared by subclasses
        self.predictors = {}
        self.response= {}
        self.model_params = {}
        self.metrics = {}
        self.aov = None
        self.t_test = None

    def fit(self, x, y):
        # Will be implemented by subclass

        raise NotImplementedError("fit() must be implemented by subclass")
    
    def predict(self, *args):
        # Will be implemented by subclass

        raise NotImplementedError("predict() must be implemented by subclass")
    
class ModelANOVA(ANOVA):
    def __init__(self, model):
        # Constructor for Model ANOVA class

        super().__init__(None)  # Call constructor for superclass

        # Initialize instance variables
        self.sources = ["Model", "Residuals", "Total"]
        self.model = model

        self.compute()  # Run the ANOVA test

    def compute(self):
        # Compute the ANOVA test on a regression model

        # Establish constants
        n = len(list(self.model.predictors.values())[0])    # Number of data points
        p = len(self.model.predictors)  # Number of predictors

        df_m = p    # Degrees of freedom of model
        df_res = n - p - 1  # Degrees of freedom of residuals

        self.dfs["Model"] = df_m
        self.dfs["Residuals"] = df_res
        self.dfs["Total"] = n - 1

        # Extract the predicted and observed responses
        predicted = self.model.predict(*list(self.model.predictors.values())[0])
        observed = list(self.model.response.values())[0]

        global_mean = sum([(i + j) for i, j in zip(predicted, observed)]) / (len(predicted) + len(observed))    # Compute global mean

        # Calculate sum of squares of the model
        ssm = 0
        for pred in predicted:
            ssm += (pred - global_mean) ** 2

        # Calculate sum of squares of the residuals
        ssres = 0
        for i in range(len(predicted)):
            ssres += (observed[i] - predicted[i]) ** 2

        self.sum_of_squares["Model"] = ssm
        self.sum_of_squares["Residuals"] = ssres
        self.sum_of_squares["Total"] = ssm + ssres

        # Find mean of squares
        msm = ssm / df_m
        msres = ssres / df_res

        self.mean_of_squares["Model"] = msm
        self.mean_of_squares["Residuals"] = msres

        # Conduct the F-Test
        f = msm / msres
        self.f_vals["Model"] = f
        p = 1 - f_dist.cdf(f, df_m, df_res)
        self.p_values["Model"] = p
    
class LinearRegression(RegressionModel):
    def __init__(self, x, y, predictor_name="predictor", response_name="response"):
        # Constructor for Linear Regression class

        super().__init__()  # Call constructor of superclass

        #   Initialize variables
        self.predictors[predictor_name] = x
        self.response[response_name] = y
        self.fit(x, y)
        self.aov = ModelANOVA(self)

    def fit(self, x, y):
        # Compute model parameters for the given datapoints

        assert len(x) == len(y), "Must be a y for every x"  # Ensure there is a y for every x

        # Compute sum of squares
        ssxy = sum([(i * j) for i, j in zip(x, y)]) - ((sum(x) * sum(y)) / len(x))
        ssxx = sum([(i ** 2) for i in x]) - ((sum(x) ** 2) / len(x))

        # Compute slope coefficient and add it to the dict of parameters
        b1 = ssxy / ssxx
        self.model_params["Beta " + list(self.predictors.keys())[0]] = b1

        # Compute intercept coefficient and add it to the dict of parameters
        b0 = ds.mean(y) - (b1 * ds.mean(x))
        self.model_params["Beta 0"] = b0

        # Begin evaluation of the model by calculating sum of squares across y and error terms
        ssyy = sum([(i ** 2) for i in y]) - ((sum(y) ** 2) / len(y))
        sse = ssyy - (b1 * ssxy)
        mse = sse / (len(x) - 2)

        self.metrics["SSE"] = sse
        self.metrics["MSE"] = mse

        # Conduct t-Test
        t = (b1) / ((mse / ssxx) ** 0.5)
        p = 2 * (1 - t_dist.cdf(abs(t), len(x) - 2))

        # Create a t-Test object for the regression object
        self.t_test = TTest([], regression=True)    # Use an empty list to fool the t-Test constructor
        self.t_test.statistic = t
        self.t_test.p_value = p
        self.t_test.df = len(x) - 2
        self.t_test.ci = (b1 - (t_dist.ppf(1 - (self.t_test.alpha / 2), len(x) - 2) * ((mse / ssxx) ** 0.5)), b1 + (t_dist.ppf(1 - (self.t_test.alpha / 2), len(x) - 2) * ((mse / ssxx) ** 0.5)))
        self.t_test.summary_text = (
                f"t-Test for Beta 1\n"
                f"Alternative Hypothesis: True Beta 1 is != 0\n"
                f"t = {self.t_test.statistic:.4f}, p = {self.t_test.p_value:.4f}, df = {self.t_test.df}\n"
                f"Sample Estimates: Slope Coefficient = {b1} Standard Error = {((mse / ssxx) ** 0.5)}\n"
                f"{100 * (1 - self.t_test.alpha)}% CI for true slope coefficient: {self.t_test.ci}"
            )

        # Compute coefficients of correlation and determination
        r = ssxy / ((ssxx * ssyy) ** 0.5)
        r2 = r ** 2

        self.metrics["r"] = r
        self.metrics["R-Squared"] = r2

    def predict(self, *args):
        # Find the predicted response for a given predictor

        result = [((self.model_params["Beta " + list(self.predictors.keys())[0]] * i) + self.model_params["Beta 0"]) for i in list(args)]   # Compute predicted response

        # If we have just one argument return simply the predicted value
        if len(result) == 1:
            return result[0]
        # If we have multiple arguments, return a list of the predicted values
        else:
            return result