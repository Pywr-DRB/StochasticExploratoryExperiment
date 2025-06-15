"""
Defines multiple objective functions used in the MOEA+Kirsch experiment.

Each function should accept historic flows (Qh) and synthetic flows (Qs) as input,
and return a single float value representing the objective.
"""

import pandas as pd
import numpy as np

from sglib.droughts.ssi import SSI, SSIDroughtMetrics
from sglib import KirschGenerator, NowakDisaggregator


x_star = np.array([0.0, 
                   0.0, 
                   0.0])  # nadir point for the objectives

obj_maxs = np.array([-4.0,
                    -100.0, 
                    -100.0])  # maximum values for the objectives, used for scaling

class Objectives:
    def __init__(self,
                 Qh=None,
                 x_star=x_star):
        """Initialize the Objectives class."""
        
        self.nobjs = 4  # Number of objectives: severity, duration, magnitude, and Mahalanobis distance
        self.ssi_calculator = SSI()
        self.drought_calculator = SSIDroughtMetrics()
        
        if Qh is not None:        
            self.Qh = Qh.squeeze()
            self.ssi_calculator.fit(Qh)
            self.ssi_calculator_is_fitted = True
        else:
            self.ssi_calculator_is_fitted = False

        # The final objective is the Mahalanobis distance from all other objectives and 
        # the x* which is opposite the ideal point.
        # The objectives are all min, so these should be large positive
        self.x_star = x_star
        self.obj_maxs = obj_maxs


    def get_drought_metrics(self, 
                            Qh, Qs):
        """
        Calculate drought metrics based on historic and synthetic flows.
        Parameters:
        -----------
        Qh : array-like
            Historic flow data.
        Qs : array-like
            Synthetic flow data.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing drought metrics such as start date, end date, severity, and duration.
        """

        if not self.ssi_calculator_is_fitted:
            self.ssi_calculator.fit(Qh)
        
        
        try:
            # get the ssi from the Qs
            Qs_ssi = self.ssi_calculator.transform(Qs)
        
            # calculate the drought metrics
            droughts = self.drought_calculator.calculate_drought_metrics(Qs_ssi)

            return droughts

        
        except Exception as e:
            print(f"Error calculating drought metrics: {e}\n\n")
            raise e
        
    def get_most_extreme_drought(self,
                                 droughts,
                                 weights=[1/5, 1/100, 1/100]):
        """
        Given N drought events (N>0), return the index for the event with:
        - max weighted avg of severity, duration, and magnitude.
        """
        
        # Normalize weights so they sum to 1
        weights = np.array(weights)
        weights /= np.sum(weights)
        
        # get weighted avg of each row
        droughts = droughts.loc[:, ['severity', 'duration', 'magnitude']]
        droughts['weighted_avg'] = (droughts['severity'] * weights[0] +
                                    droughts['duration'] * weights[1] +
                                    droughts['magnitude'] * weights[2])
        
        # Get the index of the row with the max weighted avg
        max_index = droughts['weighted_avg'].idxmax()
        most_extreme_drought = droughts.loc[max_index]

        return most_extreme_drought

    def manhattan_distance(self, objs):
        """
        Calculate the Mahalanobis distance from the objectives to the ideal point.
        
        Parameters:
        -----------
        objs : list
            List of objective values.
        x_star : array-like
            Ideal point for the objectives.
        
        Returns:
        --------
        float
            Mahalanobis distance from the objectives to the ideal point.
        """
        objs = np.array(objs)
        scaled_objs = objs / np.array(self.obj_maxs)
        diff = scaled_objs - np.zeros(len(x_star))
        return np.sum(np.abs(diff))


    def value(self, Qh, Qs):
        """
        Calculate the value of the objectives.
        
        Parameters
        ----------
        Qh : array
            Array of historical flow values.
        Qs : array
            Array of simulated flow values.

        Returns
        -------
        array
            Array of objective values.
        """
        
        # Get the most extreme drought event
        droughts = self.get_drought_metrics(Qh, Qs)

        assert droughts is not None, "Droughts should not be None"
        
        if droughts.empty:
            # If no droughts, return zeros for all objectives
            return [0.0] * self.nobjs

        
        most_extreme_drought = self.get_most_extreme_drought(droughts)

        # Objs are the severity, duration, and magnitude 
        # of the most extreme drought
        objs = []
        objs.append(most_extreme_drought['severity'])
        objs.append(most_extreme_drought['duration'])
        objs.append(most_extreme_drought['magnitude'])
        
        # Make sure they are all floats
        objs = [-abs(float(obj)) for obj in objs]
        
        # Append the Mahalanobis distance as the last objective
        objs.append(self.manhattan_distance(objs))
        
        return objs
    
    
class Constraints:
    def __init__(self, functions):
        """Initialize the Constraints class.
        
        Parameters
        ----------
        name : str
            Name of the constraints.
        functions : list
            List of functions that define the constraints.
            These functions should take two parameters: Qh (historical flow) and Qs (simulated flow).
            The functions should return a 1.0 when the constraint is violated, else 0.0 when satisfied.
        """            
        # list of functions
        self.functions = functions
        
    def value(self, Qh, Qs):
        """
        Calculate the value of the constraints.
        
        Parameters
        ----------
        Qh : array
            Array of historical flow values.
        Qs : array
            Array of simulated flow values.

        Returns
        -------
        array
            Array of constraint values.
        """
        cons = []
        for f in self.functions:
            try:
                cons.append(f(Qh, Qs))
            except Exception as e:
                raise ValueError(f"Error calculating constraint '{f.__name__}': {e}")

        return cons
    


class MOEAKirschNowakGenerator(KirschGenerator):
    """Multi-objective evolutionary algorithm generator for Kirsch streamflows.
    
    This class extends the KirschGenerator to support multi-objective optimization
    using Borg or other MOEA frameworks.
    """
    
    def __init__(self, 
                 Q, 
                 use_mpi=True,
                 **kwargs):
        """Initialize the MOEAKirschGenerator.
        
        Parameters
        ----------
        Q : pd.DataFrame
            DataFrame of historical monthly flows with shape (n_years, 12).
        objectives : list of functions, optional
            List of objective functions to be optimized.
        constraints : list of functions, optional
            List of constraint functions to be satisfied.
        """
        super().__init__(Q, **kwargs)
        self.nowak_disaggregator = NowakDisaggregator(Qh_daily=Q)        
        self.borg_output = None
        
        if use_mpi:
            from mpi4py import MPI
            self.mpi_comm = MPI.COMM_WORLD
            self.rank = self.mpi_comm.Get_rank()
            self.size = self.mpi_comm.Get_size()

        self.nobjs = 4
        self.nconstr = 1        

    def preprocessing(self, **kwargs):
        """Preprocessing step for the MOEA generator.
        
        This method can be used to set up any necessary preprocessing steps
        before running the MOEA.
        
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for preprocessing.
        """
        # No specific preprocessing needed for now
        super().preprocessing(**kwargs)

        # Preprocess the multisite NowakDisaggregator
        self.nowak_disaggregator.preprocessing()
        
    
    def fit(self):
        """Fit the MOEA generator.
        
        This method can be used to set up any necessary fitting steps
        before running the MOEA.
        
        Parameters
        ----------
        **kwargs : dict
            Additional keyword arguments for fitting.
        """
        # No specific fitting needed for now
        super().fit()
        
        # Fit the multisite NowakDisaggregator
        print("Fitting multisite NowakDisaggregator")
        self.nowak_disaggregator.fit()
        

    def load_borg_output(self, output_fname, nobjs=4, nconstr=1):
        """Load Borg output and convert to ensemble realizations.
        
        Parameters
        ----------
        output_fname : str
            Path to the Borg output CSV file.
        nobjs : int, optional
            Number of objective columns in the Borg output (default is 4).
        nconstr : int, optional
            Number of constraint columns in the Borg output (default is 1).
        
        Returns
        -------
        df : pd.DataFrame
            DataFrame containing the Borg output with columns for variables, objectives, and constraints.
        """

        df = pd.read_csv(output_fname)
    
        # first (12 * n_years) columns are var{i}
        # remaing columns are obj{i} and const{i} 
        # rows are Borg solutions (realizations)
        n_months = (df.shape[1] - nobjs - nconstr)
        n_years = n_months // 12
        if n_months % 12 != 0:
            raise ValueError("Number of months is not a multiple of 12.")
        
        return df
    
    def convert_borg_output_to_M_array(self):
        """Given dataframe with Borg output, convert to array [n_realizations, n_years, 12].
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing the Borg output with columns for variables, objectives, and constraints.
        
        Returns
        -------"
        M : np.ndarray
            3D array of shape (n_realizations, n_years, 12) containing the sample indices for streamflow realizations.
        """
        if self.borg_output is None:
            raise ValueError("Borg output not loaded. Call load_borg_output() first.")

        df = self.borg_output.copy()
        
        var_cols = [col for col in df.columns if col.startswith('var')]
        df_vars = df[var_cols]
        n_realizations = df_vars.shape[0]
        n_years = df_vars.shape[1] // 12
        if df_vars.shape[1] % 12 != 0:
            raise ValueError("Number of months is not a multiple of 12.")
        
        M = df_vars.values.reshape((n_realizations, n_years, 12))
        
        return M
    
    
    def generate(self, output_fname, nobjs=4, nconstr=1):
        """Generate streamflow ensemble from Borg output.
        
        Parameters
        ----------
        output_fname : str
            Path to the Borg output CSV file.
        nobjs : int, optional
            Number of objective columns in the Borg output (default is 4).
        nconstr : int, optional
            Number of constraint columns in the Borg output (default is 1).
        
        Returns
        -------
        Qs : dict
            Dictionary of ensemble realizations {int : pd.DataFrame}.
        """
        
        self.borg_output = self.load_borg_output(output_fname, nobjs, nconstr)
        
        M_matrix = self.convert_borg_output_to_M_array()
        
        n_realizations = M_matrix.shape[0]
        n_years = M_matrix.shape[1] - 1         # -1 to account for shift during Kirsch
        
        # Generate ensemble realizations
        Qse_monthly = {}
        for i in range(n_realizations):
            Qse_monthly[i] = self.generate_single_series(n_years=n_years, 
                                                M=M_matrix[i], 
                                                as_array=False)
        
       # Dict of daily flows, matching Qse_monthly format
        Qse_daily = {}
        
        # For each realization, disaggregate all sites simultaneously
        for real_id in Qse_monthly.keys():
            print(f"Disaggregating realization {real_id + 1}/{n_realizations}")
            
            # Get the multisite monthly flows for this realization
            # This should be a pd.DataFrame with sites as columns
            Qs_monthly_multisite = Qse_monthly[real_id]
            
            # Verify the input format
            if not isinstance(Qs_monthly_multisite, pd.DataFrame):
                raise ValueError(f"Expected DataFrame for realization {real_id}, got {type(Qs_monthly_multisite)}")
            
            if not all(col in self.site_names for col in Qs_monthly_multisite.columns):
                raise ValueError(f"Monthly data columns {Qs_monthly_multisite.columns.tolist()} "
                               f"do not match expected sites {self.site_names}")
            
            # Disaggregate all sites simultaneously using multisite disaggregator
            # Output will be pd.DataFrame of daily flows with sites as columns
            Qs_daily_multisite = self.nowak_disaggregator.disaggregate_monthly_flows(
                Qs_monthly=Qs_monthly_multisite
            )
            
            # Verify the output format
            if not isinstance(Qs_daily_multisite, pd.DataFrame):
                raise ValueError(f"Expected DataFrame output from disaggregator, got {type(Qs_daily_multisite)}")
            
            if not all(col in self.site_names for col in Qs_daily_multisite.columns):
                raise ValueError(f"Daily data columns {Qs_daily_multisite.columns.tolist()} "
                               f"do not match expected sites {self.site_names}")
            
            # Verify temporal consistency
            expected_start = Qs_monthly_multisite.index[0]
            expected_end = Qs_monthly_multisite.index[-1] + pd.offsets.MonthEnd(0)
            
            if (Qs_daily_multisite.index[0] != expected_start or 
                Qs_daily_multisite.index[-1] != expected_end):
                raise ValueError(f"Disaggregated daily flows temporal range does not match expected range.\n"
                               f"Expected: {expected_start} to {expected_end}\n"
                               f"Got: {Qs_daily_multisite.index[0]} to {Qs_daily_multisite.index[-1]}")
            
            # Store in the results dictionary
            Qse_daily[real_id] = Qs_daily_multisite
            

        print(f"Successfully disaggregated realization {real_id + 1}: "
                f"Shape {Qs_daily_multisite.shape}, "
                f"Date range {Qs_daily_multisite.index[0]} to {Qs_daily_multisite.index[-1]}")
        return Qse_daily