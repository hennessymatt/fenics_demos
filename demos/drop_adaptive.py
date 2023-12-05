"""
Solves the equations for a liquid drop resting on a surface.

The mesh is adaptively refined
"""

from fenics import *
import matplotlib.pyplot as plt

B = Constant(1000)
p = Constant(1)

N = 10
mesh = RectangleMesh(Point(0, 0), Point(1, 1), N, N)

P1 = FiniteElement('CG', 'triangle', 1)
Vh = FunctionSpace(mesh, P1)

def boundaries(x):
    return near(x[0], 0) or near(x[0], 1) or near(x[1], 0) or near(x[1], 1)
bc = DirichletBC(Vh, 0.0, boundaries)

h = Function(Vh)
v = TestFunction(Vh)

E = (sqrt(1 + dot(grad(h), grad(h))) - p * h + B * h**2 / 2) * dx
F = derivative(E, h)
J = derivative(F, h)

h.interpolate(Expression("x[0] * (1-x[0]) * x[1] * (1-x[1])", degree=1))

problem = NonlinearVariationalProblem(F, h, bcs = bc, J = J)

"""
new code starts here
"""
tol = 1e-9
M = h * h * dx
solver = AdaptiveNonlinearVariationalSolver(problem, M)

solver.solve(tol)

"""
get solution on the coarse (initial) and refined mesh
"""
h_0 = h.root_node()
h_1 = h.leaf_node()

"""
the code below is for plotting
"""
# plotting
plt.figure()
c = plot(h_0)
plt.colorbar(c)
plt.savefig('original_soln.png')

plt.figure()
plot(h.root_node().function_space().mesh())
plt.savefig('original_mesh.png')

plt.figure()
c = plot(h_1)
plt.colorbar(c)
plt.savefig('refined_soln.png')

plt.figure()
plot(h.leaf_node().function_space().mesh())
plt.savefig('refined_mesh.png')
