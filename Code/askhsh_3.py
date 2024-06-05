import pulp

def solver():

    # Initialize a minimization problem
    prob = pulp.LpProblem("ask3", pulp.LpMinimize)

    # Variables
    x1 = pulp.LpVariable("x1", upBound=2, cat=pulp.const.LpContinuous)
    x2 = pulp.LpVariable("x2", upBound=5, cat=pulp.const.LpContinuous)
    x3 = pulp.LpVariable("x3", upBound=2, cat=pulp.const.LpContinuous)
    x4 = pulp.LpVariable("x4", upBound=2, cat=pulp.const.LpContinuous)
    x5 = pulp.LpVariable("x5", upBound=3, cat=pulp.const.LpContinuous)


    # Objective function
    prob += 6*x1 + 10*x2 +8*x3 + 8*x4 + 3*x5, "obj"

    # Constraints
    prob += x1 + x2 + x3 + x4 + x5 == 4, "c1"


    # Solve the problem using the default solver
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    # Print the status of the solved LP
    print("--------------------------------------")
    print("Status:", pulp.LpStatus[prob.status])

    # Print the value of the objective
    print("objective =", pulp.value(prob.objective))

    # Print the value of the variables at the optimum
    for v in prob.variables():
        print(f'{v.name} = {v.varValue:5.2f}')

    # Print the shadow price and slack of the constraints
    print("\nSensitivity Analysis\nConstraint\t\t\t\tShadow Price\t\tSlack")
    for name, c in prob.constraints.items():
        print(f'{name} : {c}\t\t{c.pi:.2f}\t\t{c.slack:.2f}')

if __name__ == '__main__':
    solver()