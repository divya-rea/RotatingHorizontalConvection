
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
lap = lambda A: dx(dx(A)) + dy(dy(A)) + dz(dz(A))
adv = lambda A: u*dx(A) + v*dy(A) + w*dz(A)

# Problem
def add_equation(problem, eqn, even=None, tau=0):
    eq = problem.add_equation(eqn)
    # Enforce parity in y
    if even is True:
        # Drop sin(k*y) modes
        eq['valid_modes'][1::2] = False
    elif even is False:
        # Drop cos(k*y) modes
        eq['valid_modes'][0::2] = False
    # Drop tau modes in z
    if tau:
        eq['valid_modes'][:, -tau:] = False
    return eq

problem = d3.IVP([p, b, u, v, w, tau_p], namespace=locals())
add_equation(problem, "dx(u) + dy(v) + dz(w) + tau_p = 0", even=True, tau=1)
add_equation(problem, "dt(b) - κ*lap(b) = - adv(b)", even=True, tau=2)
add_equation(problem, "dt(u) - ν*lap(u) + dx(p) = - adv(u) + F*v", even=True, tau=2)
add_equation(problem, "dt(v) - ν*lap(v) + dy(p) = - adv(v) - F*u", even=False, tau=2)
add_equation(problem, "dt(w) - ν*lap(w) + dz(p) - b = - adv(w)", even=True, tau=1)
add_equation(problem, "dz(b)(z=-H) = 0", even=True)
add_equation(problem, "dz(u)(z=-H) = 0", even=True)
add_equation(problem, "dz(v)(z=-H) = 0", even=False)
add_equation(problem, "w(z=-H) = 0", even=True)
add_equation(problem, "b(z=0) = B", even=True)
add_equation(problem, "dz(u)(z=0) = 0", even=True)
add_equation(problem, "dz(v)(z=0) = 0", even=False)
add_equation(problem, "w(z=0) = 0", even=True)
add_equation(problem, "integ(p) = 0")

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
snapshots.add_task(dy(w) - dz(v), name='ωx')

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

