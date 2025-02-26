using Pkg
using Oceananigans
using CairoMakie
using KernelAbstractions: @index, @kernel
using Oceananigans.Operators: Δzᶠᶜᶜ
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume
using Printf
using Oceananigans.Utils: launch!
using CUDA
using JLD2

## Grid

H = 1.0 # vertical domain extent
Ly = 10*H # horizontal domain extent
Ny, Nz = 640, 64 # horizontal, vertical resolution

CUDA.device!(0)
arch = GPU()

grid = RectilinearGrid(arch; size = (Ny, Nz),
y = (0, Ly),
z = (-H, 0),
topology = (Flat, Bounded, Bounded))

## Buoyancy

b★ = 1.0
@inline bˢ(y, t, p) = -p.b★ * cos(π * y / p.Ly)
b_bcs = FieldBoundaryConditions(top = ValueBoundaryCondition(bˢ, parameters=(; b★, Ly)))


## Parameters

Pr = 1.0 # Prandtl number
Ra = 1e10 # Rayleigh number
Ta = 1e11 # Taylor number

ν = sqrt(Pr * b★ * Ly^3 / Ra) # Laplacian viscosity
κ = ν / Pr # Laplacian diffusivity
f0 = Ta^(1/2) * ν /Ly^2 # Coriolis parameters


## Model instantiation

model = NonhydrostaticModel(; grid,
advection = Centered(),
timestepper = :RungeKutta3,
coriolis = FPlane(f=f0),
tracers = :b,
buoyancy = BuoyancyTracer(),
closure = ScalarDiffusivity(; ν, κ),
boundary_conditions = (; b=b_bcs))

# Load the checkpoint file
# checkpoint_file = "Rotating_hc_iteration6022195.jld2"

# restored_state = load(checkpoint_file, "checkpointed_properties")


## Simulation setup

simulation = Simulation(model, Δt=1e-4, stop_time=3200000.0)


## Timestep

wizard = TimeStepWizard(cfl=0.75, max_change=1.2, max_Δt=4e-3)

simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(500))


## Progress messanger

progress(sim) = @printf("i: % 6d, sim time: % 1.3f, wall time: % 10s, Δt: % 1.4f, advective CFL: %.2e, diffusive CFL: %.2e\n",
iteration(sim), time(sim), prettytime(sim.run_wall_time),
sim.Δt, AdvectiveCFL(sim.Δt)(sim.model), DiffusiveCFL(sim.Δt)(sim.model))

simulation.callbacks[:progress] = Callback(progress, IterationInterval(1000))


## Output

u, v, w = model.velocities # unpack velocity `Field`s
b = model.tracers.b # unpack buoyancy `Field`

# total flow speed
s = @at (Center, Center, Center) sqrt(u^2 + v^2 + w^2)

# x-component of vorticity
ζ = - ∂z(v) + ∂y(w)

@kernel function _streamfunction!(ψ, grid, v)
i, j = @index(Global, NTuple)

@inbounds ψ[i, j, 1] = 0
for k in 2:grid.Nz+1
@inbounds ψ[i, j, k] = ψ[i, j, k-1] - Δzᶠᶜᶜ(i, j, k-1, grid) * v[i, j, k-1]
end
end

ψ = Field{Face, Center, Face}(grid)

function compute_ψ!(simulation)
v = simulation.model.velocities.v
grid = simulation.model.grid
launch!(arch, grid, :xy, _streamfunction!, ψ, grid, v)
end

saved_output_filename = "horizontal_convection_Ta11(12).jld2"

simulation.output_writers[:fields] = JLD2OutputWriter(model, (; s, u, b, ζ, ψ),
schedule = TimeInterval(25),
filename = saved_output_filename,
with_halos = true,
overwrite_existing = true)


## Run simulation

simulation.callbacks[:compute_ψ] = Callback(compute_ψ!, IterationInterval(1))

simulation.output_writers[:checkpointer] = Checkpointer(model,
schedule = TimeInterval(10000),
prefix = "Rotating_11hc",
overwrite_existing = true)

run!(simulation, pickup=true)
