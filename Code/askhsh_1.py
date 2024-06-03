import pulp



def solver(g1, g2):

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
    



if __name__ == '__main__':
    solver(34, 6)