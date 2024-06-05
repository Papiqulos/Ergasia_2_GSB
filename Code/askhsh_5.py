
def constraint_compression():
    # Coefficients of the constraint
    a = [6, 5, 10, 7, 5]
    # Right-hand side of the constraint
    b = 14
    S = sum(a)


    # Constraint compression
    aj_=  0
    b_ = 0

    for j in range(len(a)):
        if a[j] != 0 and S < b + abs(a[j]):
            if a[j] > 0:
                aj_ = S - b
                b_ = S - a[j]

                a[j] = aj_
                b = b_
            else:
                a[j] = b - S

    print(a)

def constraint(x1=0, x2=0, x3=0, x4=0, x5=0):
    C = []
    if x1 != 0:
        # print("x1", end=" ")    
        C.append("x1")
    if x2 != 0:
        # print("x2", end=" ")
        C.append("x2")
    if x3 != 0:
        # print("x3", end=" ")
        C.append("x3")
    if x4 != 0:
        # print("x4", end=" ")
        C.append("x4")
    if x5 != 0:
        # print("x5", end=" ")
        C.append("x5")
    return 6*x1 + 5*x2 + 10*x3 + 7*x4 + 5*x5, f"{C} <= {len(C) - 1}"

def generate_cutting_planes():
    print("Cutting planes:")
    _, c1 = constraint(x1=1, x3=1, x4=1)
    _, c2 = constraint(x1=1, x2=1, x3=1)
    _, c3 = constraint(x1=1, x2=1, x4=1)

    print(c1)
    print(c2)
    print(c3)


    



if __name__ == "__main__":
    constraint_compression()
    print("-----------------------------------------")
    generate_cutting_planes()






"""
MAX Z = 23x1 + 17x2 + 30x3 + 14x4 + 9x5
subject to
6x1 + 5x2 + 10x3 + 7x4 + 5x5 <= 14
x1 + x3 + x4 <=2
and x1,x2,x3,x4,x5 >= 0 
"""

"""
MAX Z = 23x1 + 17x2 + 30x3 + 14x4 + 9x5
subject to
6x1 + 5x2 + 10x3 + 7x4 + 5x5 <= 14
x1 + x2 + x3 <=2
and x1,x2,x3,x4,x5 >= 0 
"""

"""
MAX Z = 23x1 + 17x2 + 30x3 + 14x4 + 9x5
subject to
6x1 + 5x2 + 10x3 + 7x4 + 5x5 <= 14
x1 + x2 + x4 <=2
and x1,x2,x3,x4,x5 >= 0 
"""
