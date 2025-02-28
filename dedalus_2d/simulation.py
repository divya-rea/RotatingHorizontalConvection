
import pathlib
import numpy as np
import dedalus.public as d3
from scipy.special import erf
import logging
logger = logging.getLogger(__name__)

# Control parameters
H = 1 # vertical domain extent
Ly = 10*H # horizontal domain extent
b0 = 1 # buoyancy scale
Pr = 1 # Prandtl number
Ra = 1e10 # Rayleigh number
Ta = 1e11 # Taylor number

# Dimensional parameters (derived)
ν = (b0 * Ly**3 * Pr / Ra)**(1/2) # Laplacian viscosity
κ = ν / Pr # Laplacian diffusivity
f0 = Ta**(1/2) * ν / Ly**2 # Coriolis parameter

# Numerical parameters
Ny, Nz = 512, 64 # horizontal, vertical resolution
f_rolloff = 4 # gridpoints resolving rolloff of f (dont change)
dealias = 3/2
dtype = np.float64
stop_sim_time = 20000 / f0
max_timestep = 0.2 / f0 # need to explicitly resolve rotation
timestepper = d3.RK222
snapshot_dt = 100

# Bases
coords = d3.CartesianCoordinates('y', 'z')
dist = d3.Distributor(coords, dtype=dtype)
ybasis = d3.RealFourier(coords['y'], size=2*Ny, bounds=(-Ly, Ly), dealias=dealias) # doubled for imposing sine/cosine symmetry
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(-H, 0), dealias=dealias)

# Fields
def even_field(dist, *args, **kwargs):
    A = dist.Field(*args, **kwargs)
    A.valid_modes[1::2] = False
    return A
def odd_field(dist, *args, **kwargs):
    A = dist.Field(*args, **kwargs)
    A.valid_modes[0::2] = False
    return A

p = even_field(dist, name='p', bases=(ybasis, zbasis))
b = even_field(dist, name='b', bases=(ybasis, zbasis))
u = even_field(dist, name='u', bases=(ybasis, zbasis))
v = odd_field(dist, name='v', bases=(ybasis, zbasis))
w = even_field(dist, name='w', bases=(ybasis, zbasis))
tau_p = dist.Field(name='tau_p')
tau_b1 = even_field(dist, name='tau_b1', bases=ybasis)
tau_b2 = even_field(dist, name='tau_b2', bases=ybasis)
tau_u1 = even_field(dist, name='tau_u1', bases=ybasis)
tau_u2 = even_field(dist, name='tau_u2', bases=ybasis)
tau_v1 = odd_field(dist, name='tau_v1', bases=ybasis)
tau_v2 = odd_field(dist, name='tau_v2', bases=ybasis)
tau_w1 = even_field(dist, name='tau_w1', bases=ybasis)
tau_w2 = even_field(dist, name='tau_w2', bases=ybasis)
taus = [tau_p, tau_b1, tau_b2, tau_u1, tau_u2, tau_v1, tau_v2, tau_w1, tau_w2]

# Substitutions
y, z = dist.local_grids(ybasis, zbasis)
ey, ez = coords.unit_vector_fields(dist)
B = even_field(dist, name='B', bases=ybasis) # Surface buoyancy forcing
B['g'] = - b0 * np.cos(np.pi*y/Ly)
F = even_field(dist, name='F', bases=ybasis) # Mollified Coriolis parameter
Lf = f_rolloff * Ly / Ny # rolloff lengthscale of f
step = lambda x: erf(x*np.sqrt(np.pi)/2) # Smooth step function
F['g'] = f0 * step(np.sin(np.pi*y/Ly)*Ly/np.pi/Lf)

dx = lambda A: 0*A
dy = lambda A: d3.Differentiate(A, coords['y'])
dz = lambda A: d3.Differentiate(A, coords['z'])
lap = lambda A, Az: dx(dx(A)) + dy(dy(A)) + dz(Az)
adv = lambda A, Az: u*dx(A) + v*dy(A) + w*Az

# First-order reductions
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
bz = dz(b) + lift(tau_b1)
uz = dz(u) + lift(tau_u1)
vz = dz(v) + lift(tau_v1)
wz = dz(w) + lift(tau_w1)

# Problem
def add_even_equation(problem, *args, **kwargs):
    eq = problem.add_equation(*args, **kwargs)
    eq['valid_modes'][1::2] = False
def add_odd_equation(problem, *args, **kwargs):
    eq = problem.add_equation(*args, **kwargs)
    eq['valid_modes'][0::2] = False
problem = d3.IVP([p, b, u, v, w] + taus, namespace=locals())
add_even_equation(problem, "dx(u) + dy(v) + wz + tau_p = 0")
add_even_equation(problem, "dt(b) - κ*lap(b,bz) + lift(tau_b2) = - adv(b,bz)")
add_even_equation(problem, "dt(u) - ν*lap(u,uz) + dx(p) + lift(tau_u2) = - adv(u,uz) + F*v")
add_odd_equation(problem, "dt(v) - ν*lap(v,vz) + dy(p) + lift(tau_v2) = - adv(v,vz) - F*u")
add_even_equation(problem, "dt(w) - ν*lap(w,wz) + dz(p) - b + lift(tau_w2) = - adv(w,wz)")
add_even_equation(problem, "bz(z=-H) = 0")
add_even_equation(problem, "uz(z=-H) = 0")
add_odd_equation(problem, "vz(z=-H) = 0")
add_even_equation(problem, "w(z=-H) = 0")
add_even_equation(problem, "b(z=0) = B")
add_even_equation(problem, "uz(z=0) = 0")
add_odd_equation(problem, "vz(z=0) = 0")
add_even_equation(problem, "w(z=0) = 0")
problem.add_equation("integ(p) = 0")

# Solver
solver = problem.build_solver(timestepper)
solver.stop_sim_time = stop_sim_time

# Initial conditions
if pathlib.Path("restart.h5").exists():
    _, initial_timestep = solver.load_state("restart.h5")
    fh_mode = 'append'
else:
    b.fill_random('g', seed=42, distribution='normal', scale=1e-3) # Random noise
    initial_timestep = max_timestep / 10
    fh_mode = 'overwrite'

# Analysis
snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=snapshot_dt, max_writes=10, parallel='gather', mode=fh_mode)
snapshots.add_tasks(solver.state)
snapshots.add_task(dy(w) - dz(v), name='vorticity')

# CFL
CFL = d3.CFL(solver, initial_dt=initial_timestep, cadence=10, safety=0.5, threshold=0.05,
             max_change=1.1, min_change=0.5, max_dt=max_timestep)
CFL.add_velocity(v*ey + w*ez)

# Flow properties
flow = d3.GlobalFlowProperty(solver, cadence=10)
flow.add_property(np.sqrt(u*u + v*v + w*w), name='U')

# Main loop
try:
    logger.info('Starting main loop')
    while solver.proceed:
        timestep = CFL.compute_timestep()
        solver.step(timestep)
        if (solver.iteration-1) % 10 == 0:
            logger.info('Iteration=%i, Time=%e, dt=%e, max(U)=%f' %(solver.iteration, solver.sim_time, timestep, flow.max('U')))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    solver.log_stats()

