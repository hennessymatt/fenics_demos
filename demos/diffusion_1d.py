"""
This Python code solves the 1D diffusion equation

        u_t = D * u_{xx}

The boundary conditions are:

        u_x(0,t) = 0
        u(L, t) = u_right

The initial condition is

        u(x, 0) = u_0(x)

"""

from fenics import *
import numpy as np
import matplotlib.pyplot as plt

"""
Define model parameters
"""

# Diffusion constant
D = Constant(1)

# Value of solution at the right boundary
u_right = Constant(0)

# Size of the domain
L = Constant(3)


"""
Computational parameters
"""

# Size of time step
delta_t = 1e-2

# Number of time steps
N_time = 40

# Number of mesh points
N_x = 40

"""
Create the mesh
"""

mesh = IntervalMesh(N_x, 0, L)

# get the x coodinates
x = mesh.coordinates()[:]


"""
Define the elements used to represent the solution
"""

# use piecewise linear functions
P1 = FiniteElement("CG", mesh.ufl_cell(), 1)

# create a function space on the mesh
V = FunctionSpace(mesh, P1)

"""
Define the solution u and its test function v
"""

u = Function(V)
v = TestFunction(V)

# also define the solution at the previous time step
u_old = Function(V)

"""
Define the Dirichlet boundary conditions
"""

# Define a function for the right boundary; this function
# just needs to return the value true when x is close to
# the boundary L
def right(x):
    return near(x[0], L)

# Define the boundary condition at the right boundary
bc_right = DirichletBC(V, u_right, right)

# Create a list for all of the boundary conditions
all_bcs = [bc_right]

"""
Define the weak form
"""

# define the time derivative of u
dudt = (u - u_old) / delta_t

# Define the weak form of the diffusion equation
Fun = dudt * v * dx + D * u.dx(0) * v.dx(0) * dx

# Compute the Jacobian of the weak form (needed for the solver)
Jac = derivative(Fun, u)


"""
Define the problem and solver
"""
problem = NonlinearVariationalProblem(Fun, u, all_bcs, Jac)
solver = NonlinearVariationalSolver(problem)

"""
Define the initial condition
"""

# Expression for the initial condition
u_0 = Expression("1 - x[0]*x[0] / L / L + u_right", L = L, u_right = u_right, degree = 1)

# interpolate the initial condition to the mesh points and store in the solution u
u.interpolate(u_0)

# set u_old equal to the values in u
u_old.assign(u)



"""
Loop over time steps and solve
"""
for n in range(N_time):
    
    # solve the weak form
    solver.solve()

    # update u_old with the new solution
    u_old.assign(u)


"""
Plot the solution at the final time step
"""
plot(u)
plt.show()
