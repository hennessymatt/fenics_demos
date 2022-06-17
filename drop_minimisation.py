"""
Solves the equations for a liquid drop resting on a surface.

The weak form is generated automatically from the energy functional
"""

from fenics import *
from timeit import default_timer as timer

B = Constant(1000)
p = Constant(1)

N = 500
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
solver = NonlinearVariationalSolver(problem)
solver.parameters['nonlinear_solver'] = 'snes'
solver.parameters['snes_solver']['absolute_tolerance'] = 1e-8
solver.parameters['snes_solver']['linear_solver'] = 'mumps'

start = timer()
solver.solve()
end = timer()
print("Problem assembled and solved in", (end - start) * 1000, 'ms')
