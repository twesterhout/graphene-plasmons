using Plots
using LinearAlgebra


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
    R₁ = NTuple{3, Float64}((3 / 2, sqrt(3) / 2, 0))
    R₂ = NTuple{3, Float64}((0, sqrt(3), 0))
    δrs = [NTuple{3, Float64}((0, 0, 0)), NTuple{3, Float64}((1 / 2, sqrt(3) / 2, 0))]
    n = 4 * (k - 1) + 1
    rhombus = square_lattice(n; R₁ = R₁, R₂ = R₂, δrs = δrs)
    constraints = armchain_hexagon_constraints(k)
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
function plot_lattice(sites::AbstractVector{<:SiteInfo})
    edges = nearest_neighbours(sites)
    p = plot(
        _make_edges_plottable(sites, edges)...,
        linecolor = :black,
        axis = ([], false),
        label = nothing,
        aspect_ratio = 1,
    )
    for a in 1:2
        sublattice = filter(i -> i.sublattice == a, sites)
        scatter!(
            p,
            map(i -> i.position[1], sublattice),
            map(i -> i.position[2], sublattice),
            label = nothing,
        )
    end
    p
end
