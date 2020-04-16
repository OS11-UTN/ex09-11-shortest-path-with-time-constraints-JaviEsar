import numpy as np
from scipy.optimize import linprog

def ctnn2ctna(cnn, tnn):
    # Find the existing arcs
    arcs = np.argwhere(cnn | tnn)

    # Create the node-arc matrix and cost matrix
    na = np.zeros([cnn.shape[0], arcs.shape[0]]).astype(int)
    c = np.zeros([arcs.shape[0]])
    t = np.zeros([1, arcs.shape[0]])

    # For each arc, update the two corresponding entries in the node-arc matrix
    for i in range(arcs.shape[0]):
        na[arcs[i, 0], i] = 1
        na[arcs[i, 1], i] = -1
        t[0, i] = tnn[arcs[i, 0], arcs[i, 1]]
        c[i] = cnn[arcs[i, 0], arcs[i, 1]]

    # Return
    return na, c, t, arcs



# Time required to go from node 'row' to node 'column'
tnn = np.array([
        [0, 3, 1, 0, 0, 0],
        [0, 0, 0, 3, 0, 1],
        [0, 0, 0, 0, 3, 0],
        [0, 0, 0, 0, 0, 3],
        [0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0]
    ])

# Cost to go from node 'row' to node 'column'
cnn = np.array([
        [0, 2, 1, 0, 0, 0],
        [0, 0, 0, 2, 0, 5],
        [0, 0, 0, 0, 2, 0],
        [0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 2],
        [0, 0, 0, 0, 0, 0]
    ])

# Node-Arc Matrix and Cost Matrix computation
Aeq, C, Aub, arcs = ctnn2ctna(cnn, tnn)

# Net flow for each node (>0 is a source / <0 is a sink)
Beq = np.array([1, 0, 0, 0, 0, -1])
Bub = 9

# Decision variables bounds
bounds = tuple([0, None] for arcs in range(C.shape[0]))


# Solve the linear program using interior-point
res = linprog(C, A_eq=Aeq, b_eq=Beq, A_ub=Aub, b_ub=Bub, bounds=bounds, method='simplex')

# Print Result
print("Max Time: 9")
print("Solver: Simplex")
print("Raw Solution: ", res.x)
print("Shortest Path:")
for i in range(res.x.shape[0]):
    if res.x[i] > 1e-3:
        print(arcs[i]+1, end=" -> ")
print("Objective Function Value: ", res.fun)




Bub = 8
# Solve the linear program using interior-point
res = linprog(C, A_eq=Aeq, b_eq=Beq, A_ub=Aub, b_ub=Bub, bounds=bounds, method='simplex')

# Print Result
print("\n\n\nMax Time: 8")
print("Solver: Simplex")
print("Raw Solution: ", res.x)
print("Shortest Path:")
for i in range(res.x.shape[0]):
    if res.x[i] > 1e-3:
        print(arcs[i]+1, end=" -> ")
print("Objective Function Value: ", res.fun)


print("\n\n\nThe flow is being divided between a lower cost path and a lower time path, not fullfilling the intended "
      "time constraint.\nIt is possible to prevent such outcome by using a solver that supports binary/integer "
      "variables, which simplex does not support")
