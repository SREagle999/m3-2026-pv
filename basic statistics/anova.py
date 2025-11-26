import pandas as pd
import descriptive_stats as ds
from scipy.stats import f as f_dist

class ANOVA:
    def __init__(self, data):
        # Constructor for the general ANOVA class

        self.data = data
        
        # To be filled in by subclass
        self.sources = None
        self.dfs = {}
        self.sum_of_squares = {}
        self.mean_of_squares = {}
        self.f_vals = {}
        self.p_values = {}

    def compute(self):
        raise NotImplementedError("Subclass must implement compute()")
    
    def summary(self):
        print(f"{"Source":^20}{"df":^10}{"SS":^10}{"MS":^10}{"F":^10}{"p-value":^10}")
        for i, source in enumerate(self.sources):
            if source == "Factor Combinations":
                continue
            print(f"{self.sources[i]:20}{self.dfs[source] if source in self.dfs else "":^10}", end="")

            if source in self.sum_of_squares:
                print(f"{self.sum_of_squares[source]:^10.1f}", end="")
            else:
                print(f"{"":^10}", end="")

            if source in self.mean_of_squares:
                print(f"{self.mean_of_squares[source]:^10.1f}", end="")
            else:
                print(f"{"":^10}", end="")

            if source in self.f_vals:
                print(f"{self.f_vals[source]:^10.4f}", end="")
            else:
                print(f"{"":^10}", end="")

            if source in self.p_values:
                print(f"{self.p_values[source]:^10.4f}")
            else:
                print(f"{"":^10}")

class OneWayANOVA(ANOVA):
    def __init__(self, data):
        # Constructor for One Way ANOVA class

        super().__init__(data)  # Call constructor of superclass

        self.sources = ["Treatment", "Error", "Total"]

        self.compute()
    
    def compute(self):
        # Carryout a one-way analysis of variance

        # Establish constants
        k = len(self.data.columns)   # Number of columns
        n = [0] * k # Initialize an empty list with an element for each column
        # Iterate over the columns of data
        for i, col in enumerate(self.data.columns):
            n[i] = len(self.data[col])   # For each column populate the corresponding list element with the column length
        df_t = k - 1
        df_e = sum(n) - k

        self.dfs["Treatment"] = df_t
        self.dfs["Error"] = df_e
        self.dfs["Total"] = sum(n) - 1
        
        global_mean = sum(self.data.apply(sum)) / sum(n) # Compute global mean
        col_avgs = self.data.apply(ds.mean) # Calculate column averages

        # Initialize sum of squares for treatment and error to 0
        sst = 0
        sse = 0
        # Iterate over the columns of data
        for i, col in enumerate(self.data.columns):
            sst += n[i] * ((col_avgs[col] - global_mean) ** 2)    # Definition of sum of squares of treatment

            # Iterate over the observations in each column
            for num in self.data[col]:
                sse += (num - col_avgs[col]) ** 2 # Definition of the sum of squares of errors

        self.sum_of_squares["Treatment"] = sst
        self.sum_of_squares["Error"] = sse
        self.sum_of_squares["Total"] = sse + sst

        # Compute the mean of squares for treatment and errors
        mst = sst / df_t
        mse = sse / df_e

        self.mean_of_squares["Treatment"] = mst
        self.mean_of_squares["Error"] = mse

        # Perform an F-test
        f = mst / mse   # Compute the f-value for the ANOVA test
        p = 1 - f_dist.cdf(f, df_t, df_e)   # ANOVA test is right tailed

        self.f_vals["Treatment"] = f
        self.p_values["Treatment"] = p

class TwoWayANOVA(ANOVA):
    def __init__(self, *data):
        # Constructor for One Way ANOVA class

        super().__init__(list(data)[0])  # Call constructor of superclass

        self.data = list(data)

        self.replication = len(self.data) > 1
        self.sources = ["Treatment A", "Treatment B", "Interaction", "Error", "Total"] if self.replication else ["Treatment", "Blocks", "Error", "Total"]

        self.data = self.data if self.replication else self.data[0]

        self.compute()
    
    def compute(self):
        # Carryout a two-way analysis of variance

        # If replication (factorial)
        if self.replication:
            # Establish constants
            a = len(self.data[0].columns)   # Levels of first factor
            b = len(self.data)  # Levels of second factor
            r = len(self.data[0])   # Number of replications
            n = a * b * r   # Total units

            # Set degrees of freedom
            df_a = a - 1
            df_b = b - 1
            df_ab = df_a * df_b
            df_btw = ((a * b) - 1)
            df_e = a * b * (r - 1)

            self.dfs["Treatment A"] = df_a
            self.dfs["Treatment B"] = df_b
            self.dfs["Interaction"] = df_ab
            self.dfs["Factor Combinations"] = df_btw
            self.dfs["Error"] = df_e
            self.dfs["Total"] = n - 1

            # Compute global mean
            global_mean = 0
            for tab in self.data:
                global_mean += sum(tab.apply(sum))
            global_mean /= n

            # I don't know if anyone reads these comments, but this method for calculating ssa
            # came to me through brute force so feel free to try to clean it up if you want
            ssa = 0 # Initialize ssa
            col_avgs = []   # Initialize list for the averages of each factor level
            # Iterate over levels of second factor
            for tab in self.data:
                col_avgs.append(tab.apply(ds.mean)) # Store the list of averages of the levels of the first factor
            factor_avgs = [0] * len(col_avgs[0])    # Initialize the final resting place of the averages of the levels of the first factor
            # Average the averages of each level of factor 1 across the levels of factor 2
            for i in range(len(col_avgs[0])):
                for avgs in col_avgs:
                    factor_avgs[i] += avgs.iloc[i] / len(col_avgs)
            # Iterate over the averages for each level of factor 1
            for avg in factor_avgs:
                ssa += ((avg - global_mean) ** 2)   # Definition of sum of squares
            ssa *= r * b    # Definition of sum of squares

            # Calculate sum of squares of factor level 2
            ssb = 0
            for tab in self.data:
                ssb += (((sum(tab.apply(sum)) / (a * r)) - global_mean) ** 2)   # Definition of sum of squares
            ssb *= r * a    # Definition of sum of squares

            # Calculate sum of squares between factors (SST)
            ssbtw = 0
            for tab in self.data:
                col_avgs = tab.apply(ds.mean)
                for avg in col_avgs:
                    ssbtw += ((avg - global_mean) ** 2)
            ssbtw *= r

            ssab = ssbtw - ssa - ssb    # Calculate the sum of squares for the interaction between the two factors

            # Calculate sum of squares of total
            ss_total = 0
            # Iterate over the levels of factor 2
            for tab in self.data:
                # Iterate over the levels of factor 1
                for col in tab.columns:
                    # Iterate over the replications of each treatment
                    for val in tab[col]:
                        ss_total += ((val - global_mean) ** 2)  # Definition of sum of squares
            
            sse = ss_total - ssa -ssb - ssab # We can calculate the sse by using the difference of ss_total and the rest because it is harder to use the actual formula

            self.sum_of_squares["Treatment A"] = ssa
            self.sum_of_squares["Treatment B"] = ssb
            self.sum_of_squares["Interaction"] = ssab
            self.sum_of_squares["Factor Combinations"] = ssbtw
            self.sum_of_squares["Error"] = sse
            self.sum_of_squares["Total"] = ss_total

            # Calculate the mean of squares
            msa = ssa / df_a
            msb = ssb / df_b
            msab = ssab / df_ab
            msbtw = ssbtw / df_btw
            mse = sse / df_e

            self.mean_of_squares["Treatment A"] = msa
            self.mean_of_squares["Treatment B"] = msb
            self.mean_of_squares["Interaction"] = msab
            self.mean_of_squares["Factor Combinations"] = msbtw
            self.mean_of_squares["Error"] = mse

            f_a = msa / mse
            f_b = msb / mse
            f_ab = msab / mse
            f_btw = msbtw / mse
            p_a = 1 - f_dist.cdf(f_a, df_a, df_e)
            p_b = 1 - f_dist.cdf(f_b, df_b, df_e)
            p_ab = 1 - f_dist.cdf(f_ab, df_ab, df_e)
            p_btw = 1 - f_dist.cdf(f_btw, df_btw, df_e)

            self.f_vals["Treatment A"] = f_a
            self.f_vals["Treatment B"] = f_b
            self.f_vals["Interaction"] = f_ab
            self.f_vals["Factor Combinations"] = f_btw
            self.p_values["Treatment A"] = p_a
            self.p_values["Treatment B"] = p_b
            self.p_values["Interaction"] = p_ab
            self.p_values["Factor Combinations"] = p_btw
        else:
            # Establish constants
            k = len(self.data.columns)   # Number of columns
            b = len(self.data)  # Number of rows
            n = [0] * k # Initialize an empty list with an element for each column
            # Iterate over the columns of data
            for i, col in enumerate(self.data.columns):
                n[i] = len(self.data[col])   # For each column populate the corresponding list element with the column length
            df_t = k - 1
            df_b = b - 1
            df_e = df_t * df_b

            self.dfs["Treatment"] = df_t
            self.dfs["Blocks"] = df_b
            self.dfs["Error"] = df_e
            self.dfs["Total"] = sum(n) - 1
            
            global_mean = sum(self.data.apply(sum)) / sum(n) # Compute global mean
            col_avgs = self.data.apply(ds.mean) # Calculate column averages
            row_avgs = self.data.apply(ds.mean, axis=1) # Calculate row averages

            # Initialize sum of squares for treatment blocks and error to 0
            sst = 0
            ssb = 0
            sse = 0
            # Iterate over the columns of data
            for col in self.data.columns:
                sst += ((col_avgs[col] - global_mean) ** 2)    # Definition of sum of squares of treatment
            # Iterate over each row of data
            for row in self.data.index:
                ssb += ((row_avgs[row] - global_mean) ** 2)    # Definition of sum of squares of blocks

            # Iterate over the columns of data
            for col in self.data.columns:
                # Iterate over the rows of data
                for row in self.data.index:
                    sse += (self.data[col][row] - row_avgs[row] - col_avgs[col] + global_mean) ** 2 # Definition of the sum of squares of errors

            sst *= len(self.data)   # Multiply by number of rows (consult the formula)
            ssb *= len(self.data.columns)   # Multiply by number of columns (consult the formula)

            self.sum_of_squares["Treatment"] = sst
            self.sum_of_squares["Blocks"] = ssb
            self.sum_of_squares["Error"] = sse
            self.sum_of_squares["Total"] = sse + ssb + sst

            # Compute the mean of squares for treatment, blocks, and errors
            mst = sst / df_t
            msb = ssb / df_b
            mse = sse / df_e

            self.mean_of_squares["Treatment"] = mst
            self.mean_of_squares["Blocks"] = msb
            self.mean_of_squares["Error"] = mse

            # Perform an F-test
            f_t = mst / mse
            f_b = msb / mse
            p_t = 1 - f_dist.cdf(f_t, df_t, df_e)
            p_b = 1 - f_dist.cdf(f_b, df_b, df_e)

            self.f_vals["Treatment"] = f_t
            self.f_vals["Blocks"] = f_b
            self.p_values["Treatment"] = p_t
            self.p_values["Blocks"] = p_b
    
    def summary(self):
        super().summary()   # Call summary of super class
        
        if self.replication:
            print("Main Effects Test")
            print(f"{"Treatment":20}{self.dfs["Factor Combinations"]:^10}{self.sum_of_squares["Factor Combinations"]:^10.1f}{self.mean_of_squares["Factor Combinations"]:^10.1f}{self.f_vals["Factor Combinations"]:^10.4f}{self.p_values["Factor Combinations"]:^10.4f}")