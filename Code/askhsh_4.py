import pulp
import numpy as np

def solver_primal():
    """Solve the linear programming problem using the PuLP library. The problem is defined as:
    maximize x1 + 2*x2 + x3 - 3*x4 x5 + x6 - x7
    subject to:
        x1 + x2 - x4 + 2*x6 - 2*x7 <= 6
        x2 - x4 + x5 - 2*x6 + 2*x7 <= 4
        x2 + x3 + x6 - x7 <= 2
        x2 - x4 - x6 + x7 <= 1 
        x1, x2, x3, x4, x5, x6, x7 >= 0
        
    The function also prints the status of the solved LP, the value of the objective, the value of the variables at the optimum,
    and the shadow price and slack of the constraints.
    """

    # Initialize a maximization problem
    prob = pulp.LpProblem("ask4", pulp.LpMaximize)

    # Basic Variables
    x1 = pulp.LpVariable("x1", lowBound=0, cat=pulp.const.LpContinuous)
    x2 = pulp.LpVariable("x2", lowBound=0, cat=pulp.const.LpContinuous)
    x3 = pulp.LpVariable("x3", lowBound=0, cat=pulp.const.LpContinuous)
    x4 = pulp.LpVariable("x4", lowBound=0, cat=pulp.const.LpContinuous)
    x5 = pulp.LpVariable("x5", lowBound=0, cat=pulp.const.LpContinuous)
    x6 = pulp.LpVariable("x6", lowBound=0, cat=pulp.const.LpContinuous)
    x7 = pulp.LpVariable("x7", lowBound=0, cat=pulp.const.LpContinuous)

    # Objective function
    prob += x1 + 2*x2 + x3 - 3*x4 + x5 + x6 - x7, "obj"

    # Constraints
    prob += x1 + x2 - x4 + 2*x6 - 2*x7 <= 6, "c1"
    prob += x2 - x4 + x5 - 2*x6 + 2*x7 <= 4, "c2"
    prob += x2 + x3 + x6 - x7 <= 2, "c3"
    prob += x2 - x4 - x6 + x7 <= 1, "c4"


    

    
    # Solve the problem using the default solver
    prob.solve()

    # Print the status of the solved LP
    print("Status:", pulp.LpStatus[prob.status])

    # Print the value of the objective
    print("objective =", pulp.value(prob.objective))

    # Print the value of the variables at the optimum
    for v in prob.variables():
        print(f'{v.name} = {v.varValue}')

    # Print the shadow price and slack of the constraints
    print("\nSensitivity Analysis\nConstraint\t\t\t\tShadow Price\t\tSlack")
    for name, c in prob.constraints.items():
        print(f'{name} : {c}\t\t{c.pi}\t\t{c.slack}')

def solver_dual():
    """
    Solve the dual problem using the PuLP library.
    The problem is defined as:
    minimize 6*y1 + 4*y2 + 2*y3 + y4
    subject to:
        y1 >= 1
        y1 + y2 + y3 + y4 >= 2
        y3 >= 1
        -y1 - y2 - y4 >= -3
        y2 >= 1
        2*y1 - 2*y2 + y3 - y4 >= 1
        -2*y1 + 2*y2 - y3 + y4 >= -1
        y1, y2, y3, y4 >= 0 
    """

    # Initialize a minimization problem
    prob = pulp.LpProblem("Dual", pulp.LpMinimize)

    # Basic Variables
    y1 = pulp.LpVariable("y1", lowBound=1, cat=pulp.const.LpContinuous)
    y2 = pulp.LpVariable("y2", lowBound=1, cat=pulp.const.LpContinuous)
    y3 = pulp.LpVariable("y3", lowBound=1, cat=pulp.const.LpContinuous)
    y4 = pulp.LpVariable("y4", lowBound=0, cat=pulp.const.LpContinuous)

    # Objective function
    prob += 6*y1 + 4*y2 + 2*y3 + y4, "obj"

    # Constraints
    prob += y1 + y2 + y3 + y4 >= 2, "c1"
    prob += -y1 - y2 - y4 >= -3, "c2"
    prob += 2*y1 - 2*y2 + y3 - y4 >= 1, "c3"
    prob += -2*y1 + 2*y2 - y3 + y4 >= -1, "c4"

    # Solve the problem using the default solver
    prob.solve()

    # Print the status of the solved LP
    print("Status:", pulp.LpStatus[prob.status])

    # Print the value of the objective
    print("objective =", pulp.value(prob.objective))

    # Print the value of the variables at the optimum
    for v in prob.variables():
        print(f'{v.name} = {v.varValue}')

    # Print the shadow price and slack of the constraints
    print("\nSensitivity Analysis\nConstraint\t\t\t\tShadow Price\t\tSlack")
    for name, c in prob.constraints.items():
        print(f'{name} : {c}\t\t{c.pi}\t\t{c.slack}')

def helper():

    # Primal problem
    b_primal = np.array([6, 4, 2, 1])
    c_primal = np.array([1, 2, 1, -3, 1, 1, -1])

    A_primal = np.array([[1, 1, 0, -1, 0, 2, -2],
                        [0, 1, 0, -1, 1, -2, 2],
                        [0, 1, 1, 0, 0, 1, -1],
                        [0, 1, 0, -1, 0, -1, 1]])
    
    # Dual problem
    b_dual = c_primal
    c_dual = b_primal

    A_dual = A_primal.T

    # Given solution
    # x1, x3, x5, x7 are the basic variables
    x = np.array([15/2, 0, 11/4, 0, 5/2, 0, 3/4])

    # Possible solution of the dual problem when c1, c3, c5, c7 are set to equals
    y = np.array([1, 1, 1, 0])

    objective_primal = c_primal @ x
    objective_dual = c_dual @ y
    print(f"Primal objective: {objective_primal}")
    print(f"Dual objective: {objective_dual}")


    

if __name__ == "__main__":
    # solver_primal()
    # solver_dual()
    helper()