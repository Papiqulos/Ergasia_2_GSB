import numpy as np
import pulp



def solver(g1=0, g2=0):
    """Solve the linear programming problem using the PuLP library. The problem is defined as:
    maximize 5*x1 + 3*x2 + x3 + 4*x4
    subject to:
        x1 - 2*x2 + 2*x3 + 3*x4 <= 10
        2*x1 + 2*x2 + 2*x3 - x4 <= 6
        3*x1 + x2 - x3 + x4 <= 10
        -x2 + 2*x3 + 2*x4 <= 7
        x1, x2, x3, x4 >= 0
        
    The function also prints the status of the solved LP, the value of the objective, the value of the variables at the optimum,
    and the shadow price and slack of the constraints.
    
    Args:
        g1: The value to add to the coefficient of x1 in the objective function.
        g2: The value to add to the coefficient of x2 in the objective function.
    """
    # Initialize a maximization problem
    prob = pulp.LpProblem("ask1", pulp.LpMaximize)

    # Basic Variables
    x1 = pulp.LpVariable("x1", lowBound=0, cat=pulp.const.LpContinuous)
    x2 = pulp.LpVariable("x2", lowBound=0, cat=pulp.const.LpContinuous)
    x3 = pulp.LpVariable("x3", lowBound=0, cat=pulp.const.LpContinuous)
    x4 = pulp.LpVariable("x4", lowBound=0, cat=pulp.const.LpContinuous)

    # Objective function
    prob += (5+g1)*x1 + (3+g2)*x2 + x3 + 4*x4, "obj"

    # Constraints
    prob += x1 - 2*x2 + 2*x3 + 3*x4 <= 10, "c1"
    prob += 2*x1 + 2*x2 + 2*x3 - x4 <= 6, "c2"
    prob += 3*x1 + x2 - x3 + x4 <= 10, "c3"
    prob += -x2 + 2*x3 + 2*x4 <= 7, "c4"

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


def tolerance_range():
    """Various operations with numpy arrays necessary to find the tolerance ranges of the above linear programming problem."""
    B = np.array([[-2., 2., 3., 1],
                  [2., 2., -1., 0],
                  [1., -1., 1., 0],
                  [-1., 2., 2., 0]])
    
    N = np.array([[1., 0., 0., 0.],
                  [2., 1., 0., 0.],
                  [3., 0., 1., 0.],
                  [0., 0., 0., 1.]])
    
    b = np.array([10., 6., 10., 7.])
    e1 = np.array([1., 0., 0., 0.])
    
    cB = np.array([3., 1., 4., 0.])
    cN = np.array([5., 0., 0., 0.])

    B_inv = np.linalg.inv(B)
    dot = B_inv @ e1
    return dot
    
    



if __name__ == '__main__':
    solver(g1=-1)
    # print(tolerance_range())