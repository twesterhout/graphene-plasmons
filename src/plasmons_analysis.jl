using Plasmons
import Plasmons.dispersion
using LaTeXStrings
using LinearAlgebra
using HDF5
using Plots


function all_matrices(filenames::AbstractVector{<:AbstractString}; group_name = "/χ")
    groups = [h5open(f, "r")[group_name] for f in filenames]
    datasets = []
    for g in groups
        for d in g
            ω = real(read(attributes(d), "ħω"))
            push!(datasets, (ω, g.file.filename, HDF5.name(d)))
        end
    end
    sort!(datasets; by = x -> x[1])
    return ((t[1], h5open(io -> read(io[t[3]]), t[2], "r")) for t in datasets)
end

function loss_function(ϵ::AbstractVector{<:Complex}; count::Integer = 1)
    loss = sort!(@. imag(1 / ϵ))
    loss[start:count]
end
loss_function(ε::AbstractMatrix{<:Complex}; kwargs...) =
    loss_function(eigvals(ε); kwargs...)
loss_function(χ::AbstractMatrix{<:Complex}, V::AbstractMatrix{<:Real}; kwargs...) =
    loss_function(dielectric(χ, V); kwargs...)


function dispersion(
    data,
    lattice::AbstractVector{<:SiteInfo};
    δrs::AbstractVector{NTuple{3, Float64}},
    direction::NTuple{3, Float64} = NTuple{3, Float64}((1, 0, 0)),
    n::Int = 100,
)
    @assert n > 1
    indices = choose_full_unit_cells(lattice; δrs = δrs)
    lattice = lattice[indices]
    qs = collect(0:(π / (n - 1)):π)
    x = map(i -> i.position[1], lattice)
    y = map(i -> i.position[2], lattice)
    z = map(i -> i.position[3], lattice)
    ωs = Float64[]
    function transform(t)
        push!(ωs, t[1])
        t[2][indices, indices]
    end
    matrix = dispersion((transform(t) for t in data), map(q -> q .* direction, qs), x, y, z)
    return permutedims(matrix), qs, ωs
end

function _plot_dispersion(matrix, qs, ωs; transform, title)
    heatmap(
        qs ./ π,
        ωs,
        transform(matrix),
        xlabel = raw"$q$, $\pi/a$",
        ylabel = raw"$\hbar\omega$, eV",
        clims = (0, 1000),
        title = title,
    )
end
plot_polarizability_dispersion(matrix, qs, ωs) = _plot_dispersion(
    matrix,
    qs,
    ωs;
    transform = (@. χ -> -imag(χ)),
    title = L"$-\mathrm{Im}\left[\chi(q, \omega)\right]$",
)

function plot_single_layer_graphene_polarizability()
    setup_plots()
    χᵃᵃ, χᵃᵇ, qs, ωs = h5open("data/single_layer/polarizability_dispersion_1626_11.h5", "r") do io
        read(io["χᵃᵃ"]), read(io["χᵃᵇ"]), read(io["qs"]), read(io["ωs"])
    end
    p₁ = plot_polarizability_dispersion(χᵃᵃ, qs, ωs)
    p₂ = plot_polarizability_dispersion(χᵃᵇ, qs, ωs)
    plot!(p₂, title = nothing)
    savefig(plot(p₁, p₂, size=(800, 400)), "assets/single_layer/polarizability_dispersion.pdf")
    nothing
end

function single_layer_graphene_1626_polarizability_dispersion(filenames::AbstractVector{<:AbstractString}; output::AbstractString)
    matrix, qs, ωs = dispersion(
        all_matrices(filenames),
        armchair_hexagon(10);
        δrs = graphene_δrs,
        direction = NTuple{3, Float64}((0, 1, 0)),
        n = 100
    )
    h5open(output, "w") do io
        io["qs"] = qs
        io["ωs"] = ωs
        io["χ"] = matrix
    end
    nothing
end
