"""
solves Poisson's equation on the unit square
"""

from fenics import *

N = 20
mesh = RectangleMesh(Point(0, 0), Point(1, 1), N, N)

P1 = FiniteElement('CG', 'triangle', 1)
Vh = FunctionSpace(mesh, P1)

def boundaries(x):
    return near(x[0], 0) or near(x[0], 1) or near(x[1], 0) or near(x[1], 1)
bc = DirichletBC(Vh, 0.0, boundaries)

u = Function(Vh)
v = TestFunction(Vh)

f = 1
residual = dot(grad(u), grad(v)) * dx + f * v * dx

solve(residual == 0, u, bc)
