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

function dispersion(
    A::AbstractMatrix,
    qs::AbstractVector{NTuple{3, Float64}},
    lattice::AbstractVector{<:SiteInfo},
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
    lattice::AbstractVector{<:SiteInfo};
    k₀::NTuple{3, Float64},
    k₁::NTuple{3, Float64},
    δrs::AbstractVector{NTuple{3, Float64}},
    n::Int,
)
    @assert n > 1
    qs = map(q -> k₀ .+ (k₁ .- k₀) .* q, 0:(1 / (n - 1)):1)
    Hk = dispersion(H, qs, lattice, δrs)
    Ek = similar(H, real(eltype(H)), size(Hk, 1), size(Hk, 2))
    for i in 1:size(Hk, 1)
        Ek[i, :] .= sort!(eigvals(Hermitian(Hk[i, :, :])))
    end
    Ek
end

function band_structure(
    H::AbstractMatrix,
    lattice::AbstractVector{<:SiteInfo};
    ks::AbstractVector{NTuple{3, Float64}},
    δrs::AbstractVector{NTuple{3, Float64}},
    n::Int = 100,
)
    parts = []
    for i in 1:(length(ks) - 1)
        push!(parts, dispersion(H, lattice; k₀ = ks[i], k₁ = ks[i + 1], δrs = δrs, n = n))
    end
    vcat(parts...)
end

function graphene_high_symmetry_points(Gs = graphene_Gs)
    k(i, j) = @. i * Gs[1] + j * Gs[2]
    return (
        [k(0, 0), k(1 / 2, 0), k(2 / 3, 1 / 3), k(0, 0)],
        [raw"$\Gamma$", raw"$M$", raw"$K$", raw"$\Gamma$"],
    )
end
function energy_at_dirac_point(
    H::AbstractMatrix,
    lattice::AbstractVector{<:SiteInfo};
    Gs = graphene_Gs,
    δrs = graphene_δrs,
)
    K = @. 2 / 3 * Gs[1] + 1 / 3 * Gs[2]
    Hk = dispersion(H, [K], lattice, δrs)[1, :, :]
    sum(eigvals(Hermitian(Hk))) / size(Hk, 1)
end

function plot_graphene_band_structure(k::Int; n::Int = 100, kwargs...)
    lattice = armchair_hexagon(k)
    H₁ = build_hamiltonian(lattice, 2.7)
    H₂ = slater_koster_hamiltonian(lattice)

    ks, ticks = graphene_high_symmetry_points()
    Ek₁ = band_structure(H₁, lattice; ks = ks, δrs = graphene_δrs, n = n, kwargs...)
    Ek₂ = band_structure(H₂, lattice; ks = ks, δrs = graphene_δrs, n = n, kwargs...)
    Ek₂ .-= energy_at_dirac_point(H₂, lattice)
    plot(
        1:size(Ek₁, 1),
        hcat(Ek₁, Ek₂);
        xticks = ([1 + i * n for i in 0:(length(ks) - 1)], ticks),
        ylabel = raw"$E\,,\;\mathrm{eV}$",
        label = [raw"Nearest neighbor" "" raw"Slater-Koster" ""],
        fontfamily = "computer modern",
        lw = 2,
        color = [1 1 2 2],
        kwargs...
    )
end
function plot_graphene_density_of_states(k::Int; σ::Real = 0.15, kwargs...)
    lattice = armchair_hexagon(k)
    H₁ = build_hamiltonian(lattice, 2.7)
    H₂ = slater_koster_hamiltonian(lattice)
    H₂[diagind(H₂)] .-= energy_at_dirac_point(H₂, lattice)

    Es₁, dos₁ = density_of_states(H₁, σ = σ)
    Es₂, dos₂ = density_of_states(H₂, σ = σ)

    p = plot(
        Es₁,
        dos₁.(Es₁);
        xlabel = raw"$E\,,\;\mathrm{eV}$",
        ylabel = raw"DoS",
        fontfamily = "computer modern",
        lw = 2,
        label = "Nearest neighbor",
        kwargs...
    )
    plot!(p,
        Es₂,
        dos₂.(Es₂);
        xlabel = raw"$E\,,\;\mathrm{eV}$",
        ylabel = raw"DoS",
        fontfamily = "computer modern",
        lw = 2,
        label = "Slater-Koster"
    )
    p
end

function plot_bilayer_graphene_electronic_structure(k::Int; n::Int = 100, σ::Real = 0.12)
    lattice = armchair_bilayer_hexagon(k, rotate = 0)
    H = slater_koster_hamiltonian(lattice)
    H[diagind(H)] .-= energy_at_dirac_point(H, lattice, δrs = bilayer_graphene_δrs)

    ks, ticks = graphene_high_symmetry_points()
    Ek = band_structure(H, lattice; ks = ks, δrs = bilayer_graphene_δrs, n = n)
    p₁ = plot(
        1:size(Ek, 1),
        Ek;
        xticks = ([1 + i * n for i in 0:(length(ks) - 1)], ticks),
        ylabel = raw"$E\,,\;\mathrm{eV}$",
        label = "",
        fontfamily = "computer modern",
        color = [1 1 1 1],
        lw = 1,
    )

    Es, dos = density_of_states(H, σ = σ)
    p₂ = plot(
        Es,
        dos.(Es);
        xlabel = raw"$E\,,\;\mathrm{eV}$",
        ylabel = raw"DoS",
        fontfamily = "computer modern",
        lw = 1,
        label = "",
    )
    plot(p₁, p₂, layout = grid(1, 2), left_margin=3mm, bottom_margin=3mm, size=(800, 300), dpi=150)
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
