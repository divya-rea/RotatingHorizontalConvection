
using CairoMakie
using Oceananigans
using Oceananigans.Fields
using Oceananigans.AbstractOperations: volume
using Printf, Statistics, LinearAlgebra

saved_output_filename = "horizontal_convection_Ta11.jld2"

# Open the file with our data
ψ_timeseries = FieldTimeSeries(saved_output_filename, "ψ", backend=OnDisk())
s_timeseries = FieldTimeSeries(saved_output_filename, "s", backend=OnDisk())
u_timeseries = FieldTimeSeries(saved_output_filename, "u", backend=OnDisk())
b_timeseries = FieldTimeSeries(saved_output_filename, "b", backend=OnDisk())
ζ_timeseries = FieldTimeSeries(saved_output_filename, "ζ", backend=OnDisk())

times = b_timeseries.times

# Coordinate arrays
xu, yu, zu = nodes(u_timeseries[1])
xc, yc, zc = nodes(b_timeseries[1])
xζ, yζ, zζ = nodes(ζ_timeseries[1])


# Relevant parameters 

H = 1.0 # vertical domain extent
Ly = 10 * H # horizontal domain extent
Ny, Nz = 640, 64 # horizontal, vertical resolution
b★ = 1.0
Pr = 1.0 # Prandtl number
Ra = 1e10 # Rayleigh number
Ta = 1e11 # Taylor number
ν = sqrt(Pr * b★ * Ly^3 / Ra) # Laplacian viscosity
κ = ν / Pr # Laplacian diffusivity

@info "Making an animation from saved data..."

n = Observable(1)

title = @lift @sprintf("t=%1.2f", times[$n])

ψₙ = @lift interior(ψ_timeseries[$n], 1, :, :)
sₙ = @lift interior(s_timeseries[$n], 1, :, :)
uₙ = @lift interior(u_timeseries[$n], 1, :, :)
ζₙ = @lift interior(ζ_timeseries[$n], 1, :, :)
bₙ = @lift interior(b_timeseries[$n], 1, :, :)

ψlim = 0.0002
ulim = 0.6
blim = 0.6
ζlim = 9E-2

axis_kwargs = (xlabel = L"y / H",
ylabel = L"z / H",
limits = ((0, Ly), (-H, 0)),
aspect = Ly / H,
titlesize = 20)

fig = Figure(size = (1000, 1200))

ax_ψ = Axis(fig[2, 1];
title = L"streamfunction, $ψ / (L_y^3 b_*) ^{1/2}$", axis_kwargs...)

ax_b = Axis(fig[3, 1];
title = L"buoyancy, $b / b_*$", axis_kwargs...)

ax_ζ = Axis(fig[4, 1];
title = L"vorticity, $(∂u/∂z - ∂w/∂x) \, (L_y / b_*)^{1/2}$", axis_kwargs...)

ax_u = Axis(fig[5, 1];
title = L"in page velocity, $v / (L_y b_*)^{1/2}$", axis_kwargs...)

fig[1, :] = Label(fig, title, fontsize=24, tellwidth=false)

hm_ψ = heatmap!(ax_ψ, yζ, zζ, ψₙ;
colorrange = (-ψlim, ψlim),
colormap = :balance)
Colorbar(fig[2, 2], hm_ψ)

hm_b = contour!(ax_b, yc, zc, bₙ;
colorrange = (-blim, blim), levels=40,
colormap = :thermal)
# Colorbar(fig[3, 2], hm_b)

hm_ζ = heatmap!(ax_ζ, yζ, zζ, ζₙ;
colorrange = (-ζlim, ζlim),
colormap = :balance)
Colorbar(fig[4, 2], hm_ζ)

hm_u = heatmap!(ax_u, yu, zu, uₙ;
colorrange = (-ulim, ulim),
colormap = :balance)
Colorbar(fig[5, 2], hm_u)

frames = 1:length(times)

CairoMakie.record(fig, "horizontal_convection.mp4", frames, framerate=8) do i
msg = string("Plotting frame ", i, " of ", frames[end])
print(msg * " \r")
n[] = i
end


t = b_timeseries.times

# Plot vertical b profile
fig = Figure(size = (850, 450))
ax_b = Axis(fig[1, 1], xlabel = L"b $ / b_* $", ylabel = L"z $ / L_y $")
B = mean([mean(interior(b_timeseries[i]), dims = 2)[1, 1, 1:Nz] for i in length(t)-20:length(t)])
lines!(ax_b, B, collect(zc); linewidth = 3)
save("Bprofile.png", fig)


# Plot EKE time series
kinetic_energy = zeros(length(t))
for i = 1:length(t)
@info "iteration $i"
ke = 1/2 * sum(interior(s_timeseries[i])[1,:,:].^2) / (Ny*Nz)
kinetic_energy[i] = ke
end

fig = Figure(size = (850, 450))
ax_KE = Axis(fig[1, 1], xlabel = L"t \, (b_* / L_y)^{1/2}", ylabel = L"KE $ / (L_y b_*)$")
lines!(ax_KE, t, kinetic_energy; linewidth = 3)
save("KEtimeseries.png", fig)

# Plot psi 
fig = Figure(size = (850, 450))
ax_ψ = Axis(fig[1, 1], xlabel = L"streamfunction, $ψ / (L_y^3 b_*) ^{1/2}$")
PSI = mean([interior(ψ_timeseries[i])[1,:,:] for i in length(t)-100:length(t)])
hm_ψ = heatmap!(ax_ψ, yζ, zζ, PSI;
colorrange = (-ψlim, ψlim),
colormap = :balance)
Colorbar(fig[1,2], hm_ψ)
save("PSIprofile.png", fig)
