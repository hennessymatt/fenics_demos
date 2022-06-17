"""
Solves Stokes flow in a channel that contains a cylinder
"""

from fenics import *
import matplotlib.pyplot as plt

mu = Constant(1)

"""
problem geometry, mesh, and boundaries
"""

# import the mesh for the problem (created with gmsh)
mesh = Mesh('mesh/channel_sphere.xml')
subdomains = MeshFunction("size_t", mesh, 'mesh/channel_sphere_physical_region.xml')
bdry = MeshFunction("size_t", mesh, 'mesh/channel_sphere_facet_region.xml')

# define the boundaries (using the values from the .geo file)
cylinder = 1
axis = 2
inlet = 3
outlet = 4
wall = 5

# define the surface differential on the cylinder
ds_circ = Measure("ds", domain = mesh, subdomain_data = bdry, subdomain_id = cylinder)

# get normal vector to domain
n = FacetNormal(mesh)

# axial unit vector
e_z = as_vector([1, 0])

"""
elements, function spaces, functions, and test functions
"""

P2 = VectorElement("CG", 'triangle', 2)
P1 = FiniteElement("CG", 'triangle', 1)
TH = MixedElement([P2, P1])
Vh = FunctionSpace(mesh, TH)

X = Function(Vh)
u, p = split(X)
v, q = TestFunctions(Vh)

"""
boundary conditions
"""

# inlet and outlet
u_far_field = Expression(('1 - x[1] * x[1] / 0.25', '0'), degree=0)
bc_inlet = DirichletBC(Vh.sub(0), u_far_field, bdry, inlet)
bc_outlet = DirichletBC(Vh.sub(0), u_far_field, bdry, outlet)

# axis of symmetry (only u.n = u[1] = 0 here)
bc_axis = DirichletBC(Vh.sub(0).sub(1), Constant(0), bdry, axis)

# no-slip conditions on the circle
bc_circ = DirichletBC(Vh.sub(0), (0, 0), bdry, cylinder)

# no-slip conditions at the channel wall at z = 0.5
bc_wall = DirichletBC(Vh.sub(0), (0, 0), bdry, wall)

# collect all BCs into a Python list
bcs = [bc_inlet, bc_outlet, bc_axis, bc_circ, bc_wall]

"""
define stress tensor
"""

# create the identity tensor
I = Identity(2)

# create the stress tensor
T = mu * (grad(u) + grad(u).T) - p * I

"""
Define the weak form and compute the Jacobian
"""

FUN = -inner(T, grad(v)) * dx + div(u) * q * dx
JAC = derivative(FUN, X)

"""
Define the problem and the solver. We are going to assume that the
problem is nonlinear and use nonlinear solvers.
"""
problem = NonlinearVariationalProblem(FUN, X, bcs = bcs, J = JAC)
solver = NonlinearVariationalSolver(problem)

solver.parameters['newton_solver']['relative_tolerance'] = 1e-6
solver.parameters['newton_solver']['linear_solver'] = 'mumps'

# extract solution components
u, p = X.split()

"""
Define a function to compute the force on the cylinder and print
the result
"""

def compute_force(T):
    F = -2 * assemble(dot(e_z, T * n) * ds_circ)
    print('Force on cylinder = ', F)
    print('-----------------------------------------')
    return F

"""
solve
"""

solver.solve()
F = compute_force(T)

"""
Write the solution for the velocity and pressure to Paraview files
"""

file_u = File('output/u.pvd')
file_u << u
file_p = File('output/p.pvd')
file_p << p

plot(u[0])
plt.savefig('u.png')
