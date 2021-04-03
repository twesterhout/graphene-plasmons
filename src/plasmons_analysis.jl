using Glob
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
    χᵃᵃ, χᵃᵇ, qs, ωs =
        h5open("data/single_layer/polarizability_dispersion_1626_11.h5", "r") do io
            read(io["χᵃᵃ"]), read(io["χᵃᵇ"]), read(io["qs"]), read(io["ωs"])
        end
    p₁ = plot_polarizability_dispersion(χᵃᵃ, qs, ωs)
    p₂ = plot_polarizability_dispersion(χᵃᵇ, qs, ωs)
    plot!(p₂, title = nothing)
    savefig(
        plot(p₁, p₂, size = (800, 400)),
        "assets/single_layer/polarizability_dispersion.pdf",
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
