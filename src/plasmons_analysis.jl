using Plasmons
import Plasmons.dispersion
using LaTeXStrings
using LinearAlgebra
using DSP
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
    lattice::AbstractVector{<:SiteInfo},
    sublattices::Tuple{Int, Int};
    δrs::AbstractVector{NTuple{3, Float64}},
    direction::NTuple{3, Float64} = NTuple{3, Float64}((1, 0, 0)),
    n::Int = 100,
)
    @assert n > 1
    indices = choose_full_unit_cells(lattice; δrs = δrs)
    left_indices = filter(i -> lattice[i].sublattice == sublattices[1], indices)
    right_indices = filter(i -> lattice[i].sublattice == sublattices[2], indices)
    lattice = lattice[left_indices]
    qs = collect(0:(π / (n - 1)):π)
    x = map(i -> i.position[1], lattice)
    y = map(i -> i.position[2], lattice)
    z = map(i -> i.position[3], lattice)
    ωs = Float64[]
    function transform(t)
        push!(ωs, t[1])
        t[2][left_indices, right_indices]
    end
    matrix = dispersion((transform(t) for t in data), map(q -> q .* direction, qs), x, y, z)
    return permutedims(matrix), qs, ωs
end

function padvalid(matrix::AbstractMatrix, n₁::Int, n₂::Int)
    out = similar(matrix, size(matrix) .+ 2 .* (n₁, n₂))
    (d₁, d₂) = size(matrix)
    out[(n₁ + 1):(d₁ + n₁), (n₂ + 1):(d₂ + n₂)] .= matrix
    out[(n₁ + 1):(d₁ + n₁), 1:n₂] .= view(matrix, :, 1:1)
    out[(n₁ + 1):(d₁ + n₁), (d₂ + n₂ + 1):(d₂ + 2 * n₂)] .= view(matrix, :, d₂:d₂)
    out[1:n₁, :] .= view(out, (n₁ + 1):(n₁ + 1), :)
    out[(d₁ + n₁ + 1):(d₁ + 2 * n₂), :] .= view(out, (d₁ + n₁):(d₁ + n₁), :)
    out
end
function smoothen(matrix::AbstractMatrix, kernel::AbstractMatrix)
    @assert all(k -> mod(k, 2) == 1, size(kernel))
    k = size(kernel)
    out = conv(padvalid(matrix, div.(k, 2)...), kernel)[k[1]:(end - (k[1] - 1)), k[2]:(end - (k[2] - 1))]
    @assert size(out) == size(matrix)
    out
end
function smoothen(xs::AbstractMatrix; σ::Real = 3)
    k₁ = min(round(6 * σ, RoundUp), size(xs, 1) - 1)
    k₂ = min(round(6 * σ, RoundUp), size(xs, 2) - 1)
    if mod(k₁, 2) == 0
        k₁ += 1
    end
    if mod(k₂, 2) == 0
        k₂ += 1
    end
    μ = (div(k₁ + 1, 2), div(k₂ + 1, 2))
    kernel = similar(xs, eltype(xs), k₁, k₂)
    for y in 1:k₂
        for x in 1:k₁
            kernel[x, y] = exp(- norm((x, y) .- μ)^2 / σ)
        end
    end
    kernel ./= sum(kernel)
    smoothen(xs, kernel)
end

function _plot_dispersion(matrix, qs, ωs; σ::Union{<:Real, Nothing} = nothing, transform, title)
    matrix = transform(matrix)
    matrix = clamp.(matrix, 0, 2000)
    if !isnothing(σ)
        matrix = smoothen(matrix, σ = σ)
    end
    heatmap(
        qs ./ π,
        ωs,
        matrix,
        xlabel = raw"$q$, $\pi/a$",
        ylabel = raw"$\hbar\omega$, eV",
        title = title,
    )
end
plot_polarizability_dispersion(matrix, qs, ωs; kwargs...) = _plot_dispersion(
    matrix,
    qs,
    ωs;
    transform = (@. χ -> -imag(χ)),
    title = L"$-\mathrm{Im}\left[\chi(q, \omega)\right]$",
    kwargs...
)

function plot_single_layer_graphene_polarizability()
    χᵃᵃ, χᵃᵇ, qs, ωs =
        h5open("data/single_layer/polarizability_dispersion_1626_11.h5", "r") do io
            read(io["χᵃᵃ"]), read(io["χᵃᵇ"]), read(io["qs"]), read(io["ωs"])
        end
    p₁ = plot_polarizability_dispersion(χᵃᵃ, qs, ωs, σ = 3)
    p₂ = plot_polarizability_dispersion(χᵃᵇ, qs, ωs, σ = 3)
    plot!(p₁, colorbar = false)
    plot!(p₂, title = nothing)
    savefig(
        plot(p₁, p₂, size = (900, 400), dpi = 600),
        "assets/single_layer/polarizability_dispersion.png",
    )
    nothing
end

function single_layer_graphene_1626_polarizability_dispersion(
    filenames::AbstractVector{<:AbstractString};
    output::AbstractString,
)
    χs = []
    qs = nothing
    ωs = nothing
    for sublattices in [(1, 1), (1, 2)]
        m, qs, ωs = dispersion(
            all_matrices(filenames),
            armchair_hexagon(10),
            sublattices;
            δrs = graphene_δrs,
            direction = NTuple{3, Float64}((1, 0, 0)),
            n = 100,
        )
        push!(χs, m)
    end
    h5open(output, "w") do io
        io["qs"] = qs
        io["ωs"] = ωs
        io["χᵃᵃ"] = χs[1]
        io["χᵃᵇ"] = χs[2]
    end
    nothing
end
