using Glob
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

function combine_outputs(
    filenames::AbstractVector{<:AbstractString},
    output::AbstractString,
)
    files = [h5open(f, "r") for f in filenames]
    h5open(output, "w") do io
        for group_name in ["/χ", "/ε"]
            datasets = []
            for file in files
                if !haskey(file, group_name)
                    continue
                end
                for d in file[group_name]
                    ω = real(read(attributes(d), "ħω"))
                    push!(datasets, (ω, file.filename, HDF5.name(d)))
                end
            end
            if isempty(datasets)
                continue
            end
            sort!(datasets; by = x -> x[1])
            g = create_group(io, group_name)
            for (i, (ω, filename, path)) in enumerate(datasets)
                h5open(filename, "r") do input
                    name = string(i, pad = 4)
                    g[name] = read(input[path])
                    attributes(g[name])["ħω"] = ω
                end
            end
        end
    end
    nothing
end
combine_outputs(pattern::AbstractString, output::AbstractString) =
    combine_outputs(glob(basename(pattern), dirname(pattern)), output)


function loss_function(ϵ::AbstractVector{<:Complex}; count::Integer = 1)
    loss = sort!(@. imag(1 / ϵ))
    loss[start:count]
end
loss_function(ε::AbstractMatrix{<:Complex}; kwargs...) =
    loss_function(eigvals(ε); kwargs...)
loss_function(χ::AbstractMatrix{<:Complex}, V::AbstractMatrix{<:Real}; kwargs...) =
    loss_function(dielectric(χ, V); kwargs...)

_group_for_observable(::Val{:χ}) = "/χ"
_group_for_observable(::Val{:ε}) = "/ε"
_sort_by_for_observable(::Val{:χ}) = χ -> -imag(χ)
_sort_by_for_observable(::Val{:ε}) = ε -> -imag(1 / ε)

function leading_eigenvalues(file::HDF5.File; observable::Symbol = :χ, count::Int = 1)
    group = file[_group_for_observable(Val(observable))]
    by = _sort_by_for_observable(Val(observable))
    table = Array{Float64, 2}(undef, length(group), 1 + count)
    for (i, d) in enumerate(group)
        ħω = read(attributes(d), "ħω")
        eigenvalues = eigvals!(read(d); sortby = by)
        table[i, 1] = ħω
        table[i, 2:end] .=
            by.(view(eigenvalues, (length(eigenvalues) - (count - 1)):length(eigenvalues)))
    end
    table
end
leading_eigenvalues(filename::AbstractString; kwargs...) =
    h5open(file -> leading_eigenvalues(file; kwargs...), filename, "r")

# function dispersion(
#     data,
#     lattice::AbstractVector{<:SiteInfo},
#     sublattices::Tuple{Int, Int};
#     δrs::AbstractVector{NTuple{3, Float64}},
#     direction::NTuple{3, Float64} = NTuple{3, Float64}((1, 0, 0)),
#     n::Int = 100,
# )
#     @assert n > 1
#     indices = choose_full_unit_cells(lattice; δrs = δrs)
#     left_indices = filter(i -> lattice[i].sublattice == sublattices[1], indices)
#     right_indices = filter(i -> lattice[i].sublattice == sublattices[2], indices)
#     lattice = lattice[left_indices]
#     qs = collect(0:(π / (n - 1)):π)
#     x = map(i -> i.position[1], lattice)
#     y = map(i -> i.position[2], lattice)
#     z = map(i -> i.position[3], lattice)
#     ωs = Float64[]
#     function transform(t)
#         push!(ωs, t[1])
#         t[2][left_indices, right_indices]
#     end
#     matrix = dispersion((transform(t) for t in data), map(q -> q .* direction, qs), x, y, z)
#     return permutedims(matrix), qs, ωs
# end

function dispersion(
    A::AbstractMatrix,
    qs::AbstractVector{NTuple{3, Float64}},
    lattice::Lattice{3},
    δrs::AbstractVector{NTuple{3, Float64}},
)
    indices = choose_full_unit_cells(lattice; δrs = δrs)
    out = similar(A, complex(eltype(A)), length(qs), length(δrs), length(δrs))
    for a in 1:length(δrs)
        idxᵃ = filter(i -> lattice[i].sublattice == a, indices)
        rs = map(i -> i.position, lattice[idxᵃ])
        for b in 1:length(δrs)
            idxᵇ = filter(i -> lattice[i].sublattice == b, indices)
            out[:, a, b] .= dispersion(A[idxᵃ, idxᵇ], qs, rs)
        end
    end
    out
end
function dispersion(
    H::AbstractMatrix,
    lattice::Lattice{3};
    k₀::NTuple{3, Float64},
    k₁::NTuple{3, Float64},
    δrs::AbstractVector{NTuple{3, Float64}},
    n::Integer,
)
    @assert n > 1
    qs = map(q -> k₀ .+ (k₁ .- k₀) .* q, 0:(1 / (n - 1)):1)
    @assert all(qs[1] .≈ k₀) && all(qs[end] .≈ k₁)
    Hk = dispersion(H, qs, lattice, δrs)
    Ek = similar(H, real(eltype(H)), size(Hk, 1), size(Hk, 2))
    for i in 1:size(Hk, 1)
        Ek[i, :] .= sort!(eigvals(Hermitian(Hk[i, :, :])))
    end
    Ek
end

function band_structure(
    H::AbstractMatrix,
    lattice::Lattice;
    ks::AbstractVector{NTuple{3, Float64}},
    δrs::AbstractVector{NTuple{3, Float64}},
    n::Int = 100,
)
    parts = []
    ticks = []
    scale = minimum((norm(ks[i + 1] .- ks[i]) for i in 1:length(ks) - 1))
    offset = 0
    for i in 1:(length(ks) - 1)
        number_points = round(Int, norm(ks[i + 1] .- ks[i]) / scale * n)
        xs = (offset + 1):(offset + number_points)
        ys = dispersion(H, lattice; k₀ = ks[i], k₁ = ks[i + 1], δrs = δrs, n = number_points)
        if i != length(ks) - 1
            number_points -= 1
        end
        offset += number_points
        push!(parts, hcat(xs, ys))
        push!(ticks, xs[1])
    end
    push!(ticks, offset)
    vcat(parts...), ticks
end

function graphene_high_symmetry_points(Gs = graphene_Gs)
    k(i, j) = @. i * Gs[1] + j * Gs[2]
    ks = [k(0, 0), k(1 / 2, 0), k(2 / 3, 1 / 3), k(0, 0)]
    ticks = [raw"$\Gamma$", raw"$M$", raw"$K$", raw"$\Gamma$"]
    ks, ticks
end

function energy_at_dirac_point(
    H::AbstractMatrix,
    lattice::Lattice;
    Gs = graphene_Gs,
    δrs = graphene_δrs,
)
    K = @. 2 / 3 * Gs[1] + 1 / 3 * Gs[2]
    Hk = dispersion(H, [K], lattice, δrs)[1, :, :]
    sum(eigvals(Hermitian(Hk))) / size(Hk, 1)
end

function plot_electronic_properties(
    hamiltonians::AbstractVector{<:AbstractMatrix},
    labels::AbstractVector{<:AbstractString},
    lattice::Lattice{3};
    δrs,
    colors = [1, 2],
    n::Integer = 50,
    σ::Real = 0.12,
)
    for H in hamiltonians
        H[diagind(H)] .-= energy_at_dirac_point(H, lattice)
    end

    p₁ = plot()
    for (i, H) in enumerate(hamiltonians)
        ks, ticks = graphene_high_symmetry_points()
        Ek, tick_locations = band_structure(H, lattice; ks = ks, δrs = graphene_δrs, n = n)
        plot!(
            p₁,
            Ek[:, 1],
            Ek[:, 2:end];
            xticks = (tick_locations, ticks),
            ylabel = raw"$E\,,\;\mathrm{eV}$",
            label = [labels[i] "" "" ""],
            color = [i i i i],
            fontfamily = "computer modern",
            legend = :top,
            lw = 2,
        )
    end

    p₂ = plot()
    for (i, H) in enumerate(hamiltonians)
        Es, dos = density_of_states(H, σ = σ)
        plot!(
            p₂,
            Es,
            dos.(Es);
            xlabel = raw"$E\,,\;\mathrm{eV}$",
            ylabel = raw"DoS",
            color = i,
            fontfamily = "computer modern",
            lw = 2,
            label = "",
        )
    end

    plot(
        p₁,
        p₂,
        layout = grid(1, 2),
        left_margin = 3mm,
        bottom_margin = 3mm,
        size = (800, 300),
        dpi = 150,
    )
end

function plot_graphene_electronic_properties(k::Integer; kwargs...)
    lattice = armchair_hexagon(k)
    hamiltonians =
        [nearest_neighbor_hamiltonian(lattice, 2.7), graphene_hamiltonian_from_dft(lattice)]
    labels = ["Nearest-neighbor", "DFT"]
    plot_electronic_properties(hamiltonians, labels, lattice; δrs = graphene_δrs, kwargs...)
end
function plot_bilayer_graphene_electronic_properties(k::Integer; θ::Real = 0, kwargs...)
    lattice = armchair_bilayer_hexagon(k, rotate = θ)
    hamiltonians =
        [slater_koster_hamiltonian(lattice), bilayer_graphene_hamiltonian_from_dft(lattice)]
    labels = ["Slater-Koster", "DFT"]
    plot_electronic_properties(
        hamiltonians,
        labels,
        lattice;
        δrs = bilayer_graphene_δrs,
        kwargs...,
    )
end


function bilayer_graphene(
    k::Int,
    output::AbstractString;
    θ::Real = 0.0,
    dataset::AbstractString = "/H",
)
    lattice = armchair_bilayer_hexagon(k, rotate = θ)
    hamiltonian = bilayer_graphene_hamiltonian_from_dft(lattice)
    folder = dirname(output)
    if !isdir(folder)
        mkpath(folder)
    end
    h5open(io -> io[dataset] = hamiltonian, output, "w")
    nothing
end
bilayer_graphene_3252(output; kwargs...) = bilayer_graphene(10, output; kwargs...)



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
    out = conv(padvalid(matrix, div.(k, 2)...), kernel)[
        k[1]:(end - (k[1] - 1)),
        k[2]:(end - (k[2] - 1)),
    ]
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
            kernel[x, y] = exp(-norm((x, y) .- μ)^2 / σ)
        end
    end
    kernel ./= sum(kernel)
    smoothen(xs, kernel)
end

function _plot_dispersion(
    matrix,
    qs,
    ωs;
    σ::Union{<:Real, Nothing} = nothing,
    transform,
    title,
)
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
    kwargs...,
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
