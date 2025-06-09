from sglib import KirschNowakGenerator

class Objectives:
    def __init__(self, name, functions):
        self.name = name
        
        # list of functions
        self.functions = functions
        

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
        objs = []
        for f in self.functions:
            try:
                objs.append(f(Qh, Qs))
            except Exception as e:
                raise ValueError(f"Error calculating objective '{f.__name__}': {e}")
        
        return objs
    
    
class Constraints:
    def __init__(self, name, functions):
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
    
    
    
# class MOEAKirschNowakGenerator(KirschNowakGenerator):
#     """
#     Multi-objective evolutionary algorithm generator based on the Kirsch-Nowak method.
    
#     Parameters
#     ----------
#     objectives : Objectives
#         An instance of the Objectives class containing the objective functions.
#     """
    
#     def __init__(self, objectives):
#         super().__init__()
#         self.objectives = objectives
        
#     def generate(self, Qh, Qs):
#         """
#         Generate the objectives values for the given historical and simulated flows.
        
#         Parameters
#         ----------
#         Qh : array
#             Historical flow values.
#         Qs : array
#             Simulated flow values.

#         Returns
#         -------
#         list
#             List of objective values.
#         """
#         return self.objectives.value(Qh, Qs)
    
        