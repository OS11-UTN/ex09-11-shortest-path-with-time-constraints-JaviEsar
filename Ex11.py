import numpy as np
from scipy.optimize import linprog

def ctnn2ctna(cnn, tnn):
    # Find the existing arcs
    arcs = np.argwhere(cnn)

    # Create the node-arc matrix and cost matrix
    na = np.zeros([cnn.shape[0], arcs.shape[0]]).astype(int)
    c = np.zeros([arcs.shape[0]])
    t = np.zeros([arcs.shape[0]])

    # For each arc, update the two corresponding entries in the node-arc matrix
    for i in range(arcs.shape[0]):
        na[arcs[i, 0], i] = 1
        na[arcs[i, 1], i] = -1
        c[i] = cnn[arcs[i, 0], arcs[i, 1]]  # Cost
        t[i] = tnn[arcs[i, 0], arcs[i, 1]]  # Time Penalty

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

# Net flow for each node (>0 is a source / <0 is a sink)
Beq = np.array([1, 0, 0, 0, 0, -1])
T = 8

# Node-Arc Matrix and Cost Matrix computation
Aeq, C, t, arcs = ctnn2ctna(cnn, tnn)

# Decision variables bounds
bounds = tuple([0, None] for arcs in range(C.shape[0]))

#lambda values
lmbd = 0
dlmbd = 1
i = 0

while abs(dlmbd) > 0.001*abs(lmbd):
    i += 1

    # Solve the linear program using interior-point
    Ct = C + t * lmbd
    res = linprog(Ct, A_eq=Aeq, b_eq=Beq, bounds=bounds, method='simplex')

    # Compute gradient and update lambda
    dlmbd = (np.dot(t, res.x) - T)/i
    lmbd += dlmbd

    if (i%100) == 0:
        print("Iter ", i, ": ", lmbd - dlmbd, " + ", dlmbd)


print("\n\n\nThe optimum lambda is: ", lmbd - dlmbd)

# Solve the linear program using interior-point
Ct = C + t * (lmbd+0.001)
res = linprog(Ct, A_eq=Aeq, b_eq=Beq, bounds=bounds, method='simplex')

print("Max Time: ", T)
print("Solver: Simplex")
print("Raw Solution: ", res.x)
print("Shortest Path:")
for i in range(res.x.shape[0]):
    if res.x[i] > 1e-3:
        print(arcs[i]+1, end=" -> ")
print("Objective Function Value: ", res.fun)
