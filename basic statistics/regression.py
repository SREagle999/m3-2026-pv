import descriptive_stats as ds
from scipy.stats import t as t_dist

class RegressionModel:
    def __init__(self):
        # Constructor for the regression model class

        self.model_params = {}
        self.metrics = {}
        self.aov = None

    def fit(self, x, y):
        raise NotImplementedError("fit() must be implemented by subclass")
    
    def predict(self, *args):
        raise NotImplementedError("fit() must be implemented by subclass")
    
class LinearRegression(RegressionModel):
    def __init__(self, x, y):
        # Constructor for Linear Regression class

        super().__init__()  # Call constructor of superclass

        self.fit(x, y)

    def fit(self, x, y):
        # Compute model parameters for the given datapoints

        assert len(x) == len(y), "Must be a y for every x"

        ssxy = sum([(i * j) for i, j in zip(x, y)]) - ((sum(x) * sum(y)) / len(x))
        ssxx = sum([(i ** 2) for i in x]) - ((sum(x) ** 2) / len(x))

        b1 = ssxy / ssxx
        self.model_params["Beta 1"] = b1

        b0 = ds.mean(y) - (b1 * ds.mean(x))
        self.model_params["Beta 0"] = b0

        ssyy = sum([(i ** 2) for i in y]) - ((sum(y) ** 2) / len(y))
        sse = ssyy - (b1 * ssxy)
        mse = sse / (len(x) - 2)

        self.metrics["SSE"] = sse
        self.metrics["MSE"] = mse

        t = (b1) / ((mse / ssxx) ** 0.5)
        p = 1 - t_dist.cdf(abs(t), len(x) - 2)

        self.metrics["t"] = t
        self.metrics["p-value"] = p

        r = ssxy / ((ssxx * ssyy) ** 0.5)
        r2 = r ** 2

        self.metrics["r"] = r
        self.metrics["R-Squared"] = r2

    def predict(self, *args):
        if len(list(args)) == 1:
            return (self.model_params["Beta 1"] * list(args)[0]) + self.model_params["Beta 0"]
        else:
            return[((self.model_params["Beta 1"] * i) + self.model_params["Beta 0"]) for i in list(args)]