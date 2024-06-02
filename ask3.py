import pulp



def solver():

    # LpProblem
    prob = pulp.LpProblem("ask3", pulp.LpMinimize)

    # Variables
    x1 = pulp.LpVariable("x1", upBound=2, cat=pulp.const.LpContinuous)
    x2 = pulp.LpVariable("x2", upBound=5, cat=pulp.const.LpContinuous)
    x3 = pulp.LpVariable("x3", upBound=2, cat=pulp.const.LpContinuous)
    x4 = pulp.LpVariable("x4", upBound=2, cat=pulp.const.LpContinuous)
    x5 = pulp.LpVariable("x5", upBound=3, cat=pulp.const.LpContinuous)


    # Objective
    prob += 6*x1 + 10*x2 +8*x3 + 8*x4 + 3*x5, "obj"

    # Constraints
    prob += x1 + x2 + x3 + x4 + x5 == 4, "c1"


    # solve the problem using the default solver
    prob.solve()

    # print the status of the solved LP
    print("Status:", pulp.LpStatus[prob.status])

    # print the value of the objective
    print("objective =", pulp.value(prob.objective))

    # print the value of the variables at the optimum
    for v in prob.variables():
        print(f'{v.name} = {v.varValue:5.2f}')


if __name__=='__main__':
    solver()