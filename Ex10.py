import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

def ctnn2ctna(cnn, tnn):
    # Find the existing arcs
    arcs = np.argwhere(cnn)

    # Create the node-arc matrix and cost matrix
    na = np.zeros([cnn.shape[0], arcs.shape[0]]).astype(int)
    c = np.zeros([arcs.shape[0]])

    # For each arc, update the two corresponding entries in the node-arc matrix
    for i in range(arcs.shape[0]):
        na[arcs[i, 0], i] = 1
        na[arcs[i, 1], i] = -1
        c[i] = cnn[arcs[i, 0], arcs[i, 1]] + tnn[arcs[i, 0], arcs[i, 1]]    # Cost + Time Penalty

    # Return
    return na, c, arcs



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

# Decision variables bounds
arcs = np.argwhere(cnn)
bounds = tuple([0, None] for arcs in range(arcs.shape[0]))

#lambda values
lmbds = np.linspace(0, 1, 101)
results = np.zeros(lmbds.shape[0])

for l in range(lmbds.shape[0]):
    # Node-Arc Matrix and Cost Matrix computation
    Aeq, C, arcs = ctnn2ctna(cnn, tnn*lmbds[l])

    # Solve the linear program using interior-point
    res = linprog(C, A_eq=Aeq, b_eq=Beq, bounds=bounds, method='simplex')

    # Store result
    results[l] = res.fun - lmbds[l]*T



# Print Result
plt.plot(lmbds, results, 'bo')
plt.ylabel('Results')
plt.xlabel('Lambda')
#plt.axis([xmin, xmax, ymin, ymax])
plt.show()


l = results.argmax()
print("The optimum lambda is: ", lmbds[l])

# Node-Arc Matrix and Cost Matrix computation
Aeq, C, arcs = ctnn2ctna(cnn, tnn*lmbds[l])

# Solve the linear program using interior-point
res = linprog(C, A_eq=Aeq, b_eq=Beq, bounds=bounds, method='simplex')

print("Max Time: ", T)
print("Solver: Simplex")
print("Raw Solution: ", res.x)
print("Shortest Path:")
for i in range(res.x.shape[0]):
    if res.x[i] > 1e-3:
        print(arcs[i]+1, end=" -> ")
print("Objective Function Value: ", res.fun)





print("\n\n\nThe value lambda =", lmbds[l],
      " is an inflexion point, a slightly smaller value would result in the cost of the route having\nmore weight in "
      "the objective function than the time, and a slightly higher value would have the opposite effect")




print("\n\n\nSmaller lambda:   (Higher Cost Weight)")

# Node-Arc Matrix and Cost Matrix computation
Aeq, C, arcs = ctnn2ctna(cnn, tnn*(lmbds[l]-0.001))

# Solve the linear program using interior-point
res = linprog(C, A_eq=Aeq, b_eq=Beq, bounds=bounds, method='simplex')

print("Raw Solution: ", res.x)
print("Shortest Path:")
for i in range(res.x.shape[0]):
    if res.x[i] > 1e-3:
        print(arcs[i]+1, end=" -> ")
print("Objective Function Value: ", res.fun)


print("\n\n\nHigher lambda:   (Higher Time Weight)")

# Node-Arc Matrix and Cost Matrix computation
Aeq, C, arcs = ctnn2ctna(cnn, tnn*(lmbds[l]+0.001))

# Solve the linear program using interior-point
res = linprog(C, A_eq=Aeq, b_eq=Beq, bounds=bounds, method='simplex')

print("Raw Solution: ", res.x)
print("Shortest Path:")
for i in range(res.x.shape[0]):
    if res.x[i] > 1e-3:
        print(arcs[i]+1, end=" -> ")
print("Objective Function Value: ", res.fun)
