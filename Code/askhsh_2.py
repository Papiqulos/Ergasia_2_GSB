import numpy as np
import pulp

def solver_primal():
    """
    Solve the primal problem using the PuLP library.
    The problem is defined as:
    maximize 3*x1 - 2*x2 - 5*x3 + 7*x4 + 8*x5
    subject to:
        x2 - x3 + 3*x4 - 4*x5 = 6
        2*x1 - 3*x2 + 3*x3 + x4 <= -2
        x1 + 2*x3 - 2*x4 <= -5
        -x1 <= 2
         x1 <= 10
        -x2 <= -5
        x2 <= 25

    The function also prints the status of the solved LP, the value of the objective, the value of the variables at the optimum,
    and the shadow price and slack of the constraints.
    """
    
    # Initialize a maximization problem
    prob = pulp.LpProblem("Primal", pulp.LpMaximize)

    # Basic Variables
    x1 = pulp.LpVariable("x1", cat=pulp.const.LpContinuous)
    x2 = pulp.LpVariable("x2", lowBound=0, cat=pulp.const.LpContinuous)
    x3 = pulp.LpVariable("x3", lowBound=0, cat=pulp.const.LpContinuous)
    x4 = pulp.LpVariable("x4", lowBound=0, cat=pulp.const.LpContinuous)
    x5 = pulp.LpVariable("x5", cat=pulp.const.LpContinuous)

    # Objective function
    prob += 3*x1 - 2*x2 - 5*x3 + 7*x4 + 8*x5, "obj"

    # Constraints
    prob +=  x2 - x3 + 3*x4 - 4*x5 == -6, "c1"
    prob += -2*x1 - 3*x2 + 3*x3 + x4 <= -2, "c2"
    prob += x1 + 2*x3 - 2*x4 <= -5, "c3"
    prob += -x1 <= 2, "c4"
    prob += x1 <= 10, "c5"
    prob += -x2 <= -5, "c6"
    prob += x2 <= 25, "c7"

    
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
    minimize -6*y1 - 2*y2 - 5*y3 + 2*y4 + 10*y5 - 5*y6 + 25*y7
    subject to:
        2*y2 + y3 + y4 + y5 = 3
        y1 + 3*y2 + y6 - y7 >= -2
        -y1 - 3*y2 + 2*y3 >= -5
        3*y1 - y2 - 2*y3 >= 7
        -4*y1 == 8
        
    The function also prints the status of the solved LP, the value of the objective, the value of the variables at the optimum,
    and the shadow price and slack of the constraints.
    """
    
    # Initialize a minimization problem
    prob = pulp.LpProblem("Dual", pulp.LpMinimize)

    # Basic Variables
    y1 = pulp.LpVariable("y1", cat=pulp.const.LpContinuous)
    y2 = pulp.LpVariable("y2", lowBound=0, cat=pulp.const.LpContinuous)
    y3 = pulp.LpVariable("y3", lowBound=0, cat=pulp.const.LpContinuous)
    y4 = pulp.LpVariable("y4", lowBound=0, cat=pulp.const.LpContinuous)
    y5 = pulp.LpVariable("y5", lowBound=0, cat=pulp.const.LpContinuous)
    y6 = pulp.LpVariable("y6", lowBound=0, cat=pulp.const.LpContinuous)
    y7 = pulp.LpVariable("y7", lowBound=0, cat=pulp.const.LpContinuous)

    # Objective function
    prob += -6*y1 - 2*y2 - 5*y3 + 2*y4 + 10*y5 - 5*y6 + 25*y7, "obj"

    # Constraints
    prob += -2*y2 + y3 - y4 + y5 == 3, "c1"
    prob += y1 - 3*y2 - y6 + y7 >= -2, "c2"
    prob += -y1 + 3*y2 + 2*y3 >= -5, "c3"
    prob += 3*y1 + y2 - 2*y3 >= 7, "c4"
    prob += -4*y1 == 8, "c5"


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
    A = np.array([  [0, 1, -1, 3, -4],
                    [-2, -3, 3, 1, 0],
                    [1, 0, 2, -2, 0],
                    [-1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [0, -1, 0, 0, 0],
                    [0, 1, 0, 0, 0]])

    # Right-hand side of the constraints
    b = np.array([-6, -2, -5, 2, 10, -5, 25])
    # Coefficients of the corresponding variables in the objective function
    c = np.array([-2, 7, 8, 0, 0, 0, 0])

    # I = 2,4,5 J = 3,4,5,6
    # x2, x4, x5, x8, x9, x10, x11 are the basic variables
    B = np.array([[1, 3, -4, 0, 0, 0, 0],
                 [-3, 1, 0, 0, 0, 0, 0],
                 [0, -2, 0, 1, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0],
                 [-1, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0]])

    # Dual problem
    A_dual = A.T

    # Right-hand side of the constraints
    b_dual = np.array([3, -2, -5, 7, 8])
    # Coefficients of the corresponding variables in the objective function
    c_dual = np.array([-6, -2, 25, 0, 0])

    # I = 1,3 J = 1,2,7
    # y1, y2, y7, y8, y10 are the basic variables
    Bc = np.array([
                [0, -2, 0, -1, 0],
                [1, -3, 1, 0, 0],
                [-1, 3, 0, 0, -1],
                [3, 1, 0, 0, 0],
                [-4, 0, 0, 0, 0]])

    primal_solution = np.linalg.inv(B) @ b 
    dual_solution = np.linalg.inv(Bc) @ b_dual

    objective_primal = c @ primal_solution
    objective_dual = c_dual @ dual_solution

    print(f"Objective of Primal: {objective_primal}")
    print(f"Objective of Dual: {objective_dual}")
    print(f"Basic solution of Primal: {primal_solution}")
    print(f"Basic solution of Dual: {dual_solution}")
    print(f"Determinant of Primal B: {np.linalg.det(B):.2f}")
    print(f"Determinant of Dual B: {np.linalg.det(Bc):.2f}")



if __name__ == "__main__":
    solver_primal()
    print("-----------------------------------------")
    solver_dual()
    print("-----------------------------------------")
    helper()