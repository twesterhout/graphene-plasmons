using Plots
using LinearAlgebra
using HDF5: h5open

export single_layer_graphene_1626
export plot_example_zigzag_samples


const graphene_Rs =
    (NTuple{3, Float64}((3 / 2, sqrt(3) / 2, 0)), NTuple{3, Float64}((0, sqrt(3), 0)))
const graphene_δrs =
    [NTuple{3, Float64}((0, 0, 0)), NTuple{3, Float64}((1 / 2, sqrt(3) / 2, 0))]


struct SiteInfo{N} # , M}
    position::NTuple{N, Float64}
    sublattice::Int
    # tag::NTuple{M, Int}
end
# SiteInfo(p::NTuple{N, Float64}, i::Int, t::NTuple{M, Int}) where {N, M} =
#     M == N + 1 ? SiteInfo{N, N + 1}(p, i, t) : TypeError("expected M == N + 1")

"""
    square_lattice(n; R₁, R₂, δrs) -> Vector{SiteInfo}

Create a square lattice with given lattice vectors `R₁`, `R₂` and orbital positions `δrs`.
"""
function square_lattice(
    n::Int;
    R₁::NTuple{N, Float64},
    R₂::NTuple{N, Float64},
    δrs::AbstractVector{NTuple{N, Float64}},
) where {N}
    @assert n >= 1
    dim = n + 1
    lattice = Vector{SiteInfo{N}}(undef, length(δrs) * dim^2)
    # Vector{SiteInfo{N, N + 1}}(undef, length(δrs) * dim^2)
    offset = 1
    for y in 0:n
        for x in 0:n
            for (i, δr) in enumerate(δrs)
                R = @. R₁ * x + R₂ * y + δr
                lattice[offset] = SiteInfo(R, i) # SiteInfo(R, i, (x, y, 0, i))
                offset += 1
            end
        end
    end
    lattice
end

"""
    armchain_hexagon_constraints(k) -> Function

Create a constraint which can be used to filter sites belonging to an armchair hexagon with
side length `k`.
"""
function armchain_hexagon_constraints(k::Int)
    @assert k >= 1
    L = 3 * k - 2 # Side length of our hexagon
    ε = 5 / 1000 # To avoid issues with float comparisons
    line(a, b) = i -> i.position[2] - (a * i.position[1] + b)
    above(f) = i -> f(i) >= -ε
    below(f) = i -> f(i) <= ε
    constraints = [
        above(line(0, sqrt(3) / 2 * L)),
        below(line(0, 3 * sqrt(3) / 2 * L)),
        above(line(-sqrt(3), sqrt(3) * L)),
        below(line(-sqrt(3), 3 * sqrt(3) * L)),
        above(line(sqrt(3), -sqrt(3) * L)),
        below(line(sqrt(3), sqrt(3) * L)),
    ]
    return i -> mapfoldl(p -> p(i), &, constraints)
end

function zigzag_hexagon_constraints(k::Int)
    @assert k >= 1
    ε = 5 / 1000 # To avoid issues with float comparisons
    line(a, b) = i -> i.position[2] - (a * i.position[1] + b)
    above(f) = i -> f(i) >= -ε
    below(f) = i -> f(i) <= ε
    constraints = [
        above(line(-1 / sqrt(3), sqrt(3) * (k - 1 / 2))),
        below(line(-1 / sqrt(3), sqrt(3) * (3 * k + 1 / 2))),
    ]
    return i -> mapfoldl(p -> p(i), &, constraints)
end

"""
    choose_full_unit_cells(sites::AbstractVector{<:SiteInfo}; δrs) -> Vector{SiteInfo}

Given a lattice choose only those sites which constitute full unit cells. `δrs` specifies
positions of atoms within one unit cell.
"""
function choose_full_unit_cells(
    sites::AbstractVector{SiteInfo{N}};
    δrs::AbstractVector{NTuple{N, Float64}},
) where {N}
    function find_unit_cell(site::Int)::Union{Vector{Int}, Nothing}
        sites[site].sublattice != 1 && return nothing
        isclose(a, b) = mapreduce((x₁, x₂) -> isapprox(x₁, x₂), &, a, b)
        cell = [site]
        for i in 2:length(δrs)
            r = sites[site].position .+ δrs[i]
            predicate(x) = x.sublattice == i && isclose(x.position, r)
            other = findfirst(predicate, sites)
            isnothing(other) && return nothing
            push!(cell, other)
        end
        cell
    end

    indices = Int[]
    for cell in filter(!isnothing, map(find_unit_cell, 1:length(sites)))
        for site in cell
            push!(indices, site)
        end
    end
    indices
end

"""
    nearest_neighbours(sites::AbstractVector{<:SiteInfo}) -> Vector{NTuple{2, Int}}

Return a list of nearest neighbours. Indices of sites rather than their positions are
returned. Also note that this list does not contain duplicates, i.e. of `(i, j)` and `(j,
i)` only one is returned.
"""
function nearest_neighbours(sites::AbstractVector{<:SiteInfo})
    ε = 1 / 1000
    cutoff = 1 + ε
    distance(i, j) = norm(sites[i].position .- sites[j].position)
    edges = NTuple{2, Int}[]
    for i in 1:length(sites)
        for j in (i + 1):length(sites)
            if distance(i, j) < cutoff
                push!(edges, (i, j))
            end
        end
    end
    edges
end

"""
    armchair_hexagon(k::Int) -> Vector{SiteInfo}

Construct a hexagon lattice with side length `k` with armchair boundaries.
"""
function armchair_hexagon(k::Int)
    @assert k >= 1
    R₁, R₂ = graphene_Rs
    δrs = graphene_δrs
    n = 4 * (k - 1) + 1
    rhombus = square_lattice(n; R₁ = R₁, R₂ = R₂, δrs = δrs)
    constraints = armchain_hexagon_constraints(k)
    return filter(constraints, rhombus)
end

function zigzag_hexagon(k::Int)
    @assert k >= 1
    R₁, R₂ = graphene_Rs
    δrs = graphene_δrs
    n = 2 * k
    rhombus = square_lattice(n; R₁ = R₁, R₂ = R₂, δrs = δrs)
    constraints = zigzag_hexagon_constraints(k)
    return filter(constraints, rhombus)
end

function _make_edges_plottable(sites, edges)
    data = similar(edges, Float64, 3 * size(edges, 1), 2)
    for i in 1:length(edges)
        offset = 1 + 3 * (i - 1)
        data[offset, :] .= sites[edges[i][1]].position[1:2]
        data[offset + 1, :] .= sites[edges[i][2]].position[1:2]
        data[offset + 2, :] = [NaN, NaN]
    end
    return view(data, :, 1), view(data, :, 2)
end

"""
    plot_lattice(sites::AbstractVector{<:SiteInfo}) -> Plot

Visualize a lattice. Different sublattices are shown in different color.
"""
function plot_lattice(sites::AbstractVector{<:SiteInfo}; kwargs...)
    edges = nearest_neighbours(sites)

    function limits(axis::Int)
        (m, M) = extrema((i.position[axis] for i in sites))
        return (m - 0.5, M + 0.5)
    end

    p = plot(
        _make_edges_plottable(sites, edges)...,
        linecolor = :black,
        axis = ([], false),
        label = nothing,
        xlims = limits(1),
        ylims = limits(2),
        aspect_ratio = 1,
    )
    for a in 1:2
        sublattice = filter(i -> i.sublattice == a, sites)
        scatter!(
            p,
            map(i -> i.position[1], sublattice),
            map(i -> i.position[2], sublattice),
            label = nothing,
            kwargs...,
        )
    end
    p
end

function plot_example_zigzag_samples(output::Union{AbstractString, Nothing} = nothing)
    plotone(k; kwargs...) = plot_lattice(zigzag_hexagon(k); kwargs...)
    p = plot(
        plotone(1),
        plotone(2),
        plotone(3),
        plotone(4),
        layout = grid(1, 4, widths = [0.1, 0.2, 0.3, 0.4]),
        size = (640, 260),
    )
    if isnothing(output)
        return p
    else
        savefig(p, output)
        return nothing
    end
end


function build_hamiltonian(
    ::Type{ℝ},
    sites::AbstractVector{<:SiteInfo},
    t₁::Real,
) where {ℝ <: Real}
    edges = nearest_neighbours(sites)
    H = zeros(ℝ, length(sites), length(sites))
    for (i, j) in edges
        H[i, j] = -t₁
        H[j, i] = -t₁
    end
    H
end
build_hamiltonian(sites::AbstractVector{<:SiteInfo}, t₁::Real) =
    build_hamiltonian(Float64, sites, t₁)


function density_of_states(eigenvalues::AbstractVector{<:Real}; σ::Real = 1e-1)
    # NOTE: We divide by the number of eigenvalues here to ensure proper normalization
    gaussian(x, μ) =
        1 / (sqrt(2π) * σ * length(eigenvalues)) * exp(-1 / 2 * ((x - μ) / σ)^2)
    Eₘᵢₙ = first(eigenvalues)
    Eₘᵢₙ -= 1 / 20 * abs(Eₘᵢₙ) # Make the range slightly larger to see DoS go to zero
    Eₘₐₓ = last(eigenvalues)
    Eₘₐₓ += 1 / 20 * abs(Eₘₐₓ)
    ΔE = (Eₘₐₓ - Eₘᵢₙ) / 1000
    dos(x) = mapreduce(E -> gaussian(x, E), +, eigenvalues)
    return Eₘᵢₙ:ΔE:Eₘₐₓ, dos
end
density_of_states(hamiltonian::Hermitian; kwargs...) =
    density_of_states(eigvals(hamiltonian); kwargs...)
density_of_states(hamiltonian::AbstractMatrix{<:Union{Real, Complex}}; kwargs...) =
    density_of_states(Hermitian(hamiltonian); kwargs...)
density_of_states(filename::AbstractString; dataset::AbstractString = "/H", kwargs...) =
    h5open(io -> density_of_states(read(io[dataset]); kwargs...), filename, "r")

function plot_density_of_states(Es::AbstractRange, dos::Function)
    plot(
        Es,
        dos.(Es),
        xlabel = raw"$E$, eV",
        ylabel = raw"Density",
        label = nothing,
        size = (640, 480),
        dpi = 300,
    )
end


"""
    single_layer_graphene_1626(output::AbstractString; t₁ = 2.7, dataset = "/H")

Generate HDF5 file `output` which can be used as input for `Plasmons.jl`. The output file
contains a single dataset -- tight-binding Hamiltonian for single layer graphene hexagon of
1626 sites. Nearest-neighbour hopping parameter is `t₁` (all other hoppings are assumed to
be 0).
"""
function single_layer_graphene_1626(
    output::AbstractString;
    t₁::Real = 2.7,
    dataset::AbstractString = "/H",
)
    lattice = armchair_hexagon(10)
    @assert length(lattice) == 1626
    hamiltonian = build_hamiltonian(lattice, t₁)
    folder = dirname(output)
    if !isdir(folder)
        mkpath(folder)
    end
    h5open(io -> io[dataset] = hamiltonian, output, "w")
    nothing
end
