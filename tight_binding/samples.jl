using Plots



struct SiteInfo{N}
    position::NTuple{N, Float64}
    sublattice::Int
end

function graphene_rhombus(
    n::Int;
    R₁ = NTuple{3, Float64}((3 / 2, sqrt(3) / 2, 0)),
    R₂ = NTuple{3, Float64}((0, sqrt(3), 0)),
    δr = NTuple{3, Float64}((1 / 2, sqrt(3) / 2, 0)),
)
    @assert n >= 1
    dim = n + 1
    lattice = Vector{SiteInfo{3}}(undef, 2 * dim^2)
    offset = 1
    for y in 0:n
        for x in 0:n
            R = @. R₁ * x + R₂ * y
            lattice[offset] = SiteInfo(R, 1)
            lattice[offset + 1] = SiteInfo(R .+ δr, 2)
            offset += 2
        end
    end
    lattice
end

function armchain_hexagon_constraints(k::Int)
    @assert k >= 1
    L = 3 * k - 2 # Side length of our hexagon
    ε = 5 / 1000 # To avoid issues with float comparisons
    line(a, b) = t -> t.position[2] - (a * t.position[1] + b)
    above(f) = t -> f(t) >= -ε
    below(f) = t -> f(t) <= ε
    return [
        above(line(0, sqrt(3) / 2 * L)),
        below(line(0, 3 * sqrt(3) / 2 * L)),
        above(line(-sqrt(3), sqrt(3) * L)),
        below(line(-sqrt(3), 3 * sqrt(3) * L)),
        above(line(sqrt(3), -sqrt(3) * L)),
        below(line(sqrt(3), sqrt(3) * L)),
    ]
end

apply_constraints(constraints, sites) =
    foldl((acc, p) -> filter(p, acc), constraints; init = sites)

function armchair_hexagon(k::Int)
    @assert k >= 1
    n = 4 * (k - 1) + 1
    rhombus = graphene_rhombus(n)
    constraints = armchain_hexagon_constraints(k)
    return apply_constraints(constraints, rhombus)
end

function plot_lattice(sites::AbstractVector{<:SiteInfo})
    x = map(i -> i.position[1], sites)
    y = map(i -> i.position[2], sites)
    scatter(x, y, aspect_ratio = 1)
end
