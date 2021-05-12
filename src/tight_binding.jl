using Plots
using Plots.PlotMeasures
using LinearAlgebra
using HDF5: h5open

export single_layer_graphene_1626
export plot_example_zigzag_samples, plot_example_bilayer_samples
export plot_density_of_states
export armchair_hexagon, zigzag_hexagon

"""
Lattice vectors for graphene. Carbon-carbon distance is taken exactly `1`. Rescale them if
necessary.
"""
const graphene_Rs = (
    NTuple{3, Float64}((3 / 2, sqrt(3) / 2, 0)),
    NTuple{3, Float64}((0, sqrt(3), 0)),
    # Assume that carbon-carbon distance in graphene is 1.42... Å and interlayer distance is
    # 3.35... Å.
    NTuple{3, Float64}((0, 0, 3.35 / 1.424919)),
)

"""
Orbital positions within a unit cell for monolayer graphene.
"""
const graphene_δrs =
    [NTuple{3, Float64}((0, 0, 0)), NTuple{3, Float64}((1 / 2, sqrt(3) / 2, 0))]

"""
Reciprocal lattice vectors for monolayer graphene.
"""
const graphene_Gs = (
    2π .* NTuple{3, Float64}((2 / 3, 0, 0)),
    2π .* NTuple{3, Float64}((-1 / 3, 1 / sqrt(3), 0)),
)

"""
Orbital positions within a unit cell for AB-stacked bilayer graphene.
"""
const bilayer_graphene_δrs = [
    NTuple{3, Float64}((0, 0, 0)),
    NTuple{3, Float64}((1 / 2, sqrt(3) / 2, 0)),
    NTuple{3, Float64}((1, 0, 0)) .+ graphene_Rs[3],
    NTuple{3, Float64}((1 + 1 / 2, sqrt(3) / 2, 0)) .+ graphene_Rs[3],
]

"""
    SiteInfo{N}

All we need to know about a site. Contains two fields:

  * `position` specifies position of the site in N-dimensional space;
  * `sublattice` specifies the sublattice to which the site belongs.
"""
struct SiteInfo{N}
    position::NTuple{N, Float64}
    sublattice::Int
end

"""
    Lattice{N}

A lattice is just a collection of sites, i.e. of `SiteInfo`.
"""
const Lattice{N} = AbstractVector{SiteInfo{N}}

"""
    square_lattice(n; R₁, R₂, δrs) -> Lattice

Create a square lattice with given lattice vectors `R₁`, `R₂` and orbital positions `δrs`.
"""
function square_lattice(
    n::Integer;
    R₁::NTuple{N, Float64},
    R₂::NTuple{N, Float64},
    δrs::AbstractVector{NTuple{N, Float64}},
) where {N}
    if n < 1
        throw(ArgumentError("invalid 'n': $n; expected a positive integer"))
    end
    dim = n + 1
    lattice = Vector{SiteInfo{N}}(undef, length(δrs) * dim^2)
    offset = 1
    for y in 0:n
        for x in 0:n
            for (i, δr) in enumerate(δrs)
                lattice[offset] = SiteInfo(NTuple(@. R₁ * x + R₂ * y + δr), i)
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
    zigzag_hexagon_constraints(k) -> Function

Create a constraint which can be used to filter sites belonging to a hexagon with
zigzag edges of side length `k`.
"""
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
    choose_full_unit_cells(sites::Lattice; δrs) -> Lattice

Given a lattice choose only those sites which constitute full unit cells. `δrs` specifies
positions of atoms within one unit cell.
"""
function choose_full_unit_cells(
    sites::Lattice{N};
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
    nearest_neighbours(sites::Lattice)
    nearest_neighbours(sites::Lattice, system::Symbol) -> Vector{NTuple{2, Int}}
    nearest_neighbours(sites::Lattice, condition::Function) -> Vector{NTuple{2, Int}}

Return a list of nearest neighbours. Indices of sites rather than their positions are
returned. Also note that this list does not contain duplicates, i.e. of `(i, j)` and `(j,
i)` only one is returned.
"""
function nearest_neighbours(sites::Lattice, condition::Function)
    edges = NTuple{2, Int}[]
    for i in 1:length(sites)
        for j in (i + 1):length(sites)
            if condition(i, j)
                push!(edges, (i, j))
            end
        end
    end
    edges
end
function nearest_neighbours(sites::Lattice, system::Symbol)
    ε = 1 / 1000
    cutoff = 1 + ε
    distance(i, j) = norm(sites[i].position .- sites[j].position)
    if system == :bilayer
        condition =
            (i, j) ->
                sites[i].position[3] == sites[j].position[3] && distance(i, j) < cutoff
        nearest_neighbours(sites, condition)
    elseif system == :single_layer
        condition = (i, j) -> distance(i, j) < cutoff
        nearest_neighbours(sites, condition)
    else
        throw(ArgumentError("invalid system: $system; expected either :single_layer or :bilayer"))
    end
end
nearest_neighbours(sites::Lattice{N}) where {N} =
    N < 3 ? nearest_neighbours(sites, :single_layer) : nearest_neighbours(sites, :bilayer)

"""
    armchair_hexagon(k::Int) -> Lattice{3}

Construct a hexagon lattice with side length `k` with armchair boundaries.
"""
function armchair_hexagon(k::Integer)
    k < 1 && throw(ArgumentError("invalid 'k': $k; expected a positive integer."))
    R₁, R₂, _ = graphene_Rs
    δrs = graphene_δrs
    n = 4 * (k - 1) + 1
    rhombus = square_lattice(n; R₁ = R₁, R₂ = R₂, δrs = δrs)
    constraints = armchain_hexagon_constraints(k)
    return filter(constraints, rhombus)
end

"""
    zigzag_hexagon(k::Int) -> Lattice{3}

Construct a hexagon lattice with side length `k` with zigzag boundaries.
"""
function zigzag_hexagon(k::Integer)
    k < 1 && throw(ArgumentError("invalid 'k': $k; expected a positive integer."))
    R₁, R₂, _ = graphene_Rs
    δrs = graphene_δrs
    n = 2 * k
    rhombus = square_lattice(n; R₁ = R₁, R₂ = R₂, δrs = δrs)
    constraints = zigzag_hexagon_constraints(k)
    return filter(constraints, rhombus)
end

shift_sample(sites::AbstractVector{SiteInfo{N}}; shift::NTuple{N, Float64}) where {N} =
    map(i -> SiteInfo(i.position .+ shift, i.sublattice), sites)

function rotate_sample_in_plane(
    sites::AbstractVector{SiteInfo{N}};
    θ::Real,
    origin::NTuple{N, Float64},
) where {N}
    θ = deg2rad(θ)
    function transformation(p)
        v = p .- origin
        v = (v[1] * cos(θ) - v[2] * sin(θ), v[1] * sin(θ) + v[2] * cos(θ), v[3:end]...)
        return v .+ origin
    end
    map(i -> SiteInfo(transformation(i.position), i.sublattice), sites)
end

"""
    armchair_bilayer_hexagon(k::Integer; rotate::Real) -> Lattice{3}

Construct AB-stacked bilayer hexagon sample with side length `k`. Second layer is rotated by
`rotate` degrees.
"""
function armchair_bilayer_hexagon(
    k::Integer;
    shift::NTuple{3, Float64} = graphene_Rs[3] .+ NTuple{3, Float64}((1, 0, 0)),
    rotate::Real = 0.0,
)
    R₁, R₂, R₃ = graphene_Rs
    δrs = graphene_δrs
    layer₁ = armchair_hexagon(k)
    layer₂ = map(i -> SiteInfo(i.position, i.sublattice + 2), layer₁)
    layer₂ = shift_sample(layer₂, shift = shift)
    center_of_mass = (1 / length(layer₁)) .* reduce(.+, (i.position for i in layer₁))
    layer₂ = rotate_sample_in_plane(
        layer₂;
        θ = rotate,
        origin = center_of_mass .+ graphene_δrs[2],
    )
    [layer₁; layer₂]
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
    plot_lattice(sites::Lattice; kwargs...) -> Plot

Visualize a lattice. Different sublattices are shown in different color.
"""
function plot_lattice(sites::Lattice; kwargs...)
    edges = nearest_neighbours(sites)

    function limits(axis::Int)
        (m, M) = extrema((i.position[axis] for i in sites))
        return (m - 0.5, M + 0.5)
    end

    p = plot(
        _make_edges_plottable(sites, edges)...,
        linewidth = 2,
        linecolor = RGB(80 / 255, 80 / 255, 80 / 255),
        xlims = limits(1),
        ylims = limits(2),
        alpha = 0.75,
        axis = ([], false),
        label = nothing,
        aspect_ratio = 1,
    )

    (aₘᵢₙ, aₘₐₓ) = extrema((i.sublattice for i in sites))
    colors = [2 3 2 3]
    alpha = [1.0 1.0 0.6 0.6]
    for a in aₘₐₓ:-1:aₘᵢₙ
        sublattice = filter(i -> i.sublattice == a, sites)
        scatter!(
            p,
            map(i -> i.position[1], sublattice),
            map(i -> i.position[2], sublattice),
            markerstrokewidth = 2,
            color = colors[a],
            alpha = alpha[a],
            label = nothing;
            kwargs...,
        )
    end
    p
end

function plot_example_zigzag_samples()
    plotone(k; kwargs...) = plot_lattice(zigzag_hexagon(k); kwargs...)
    plot(
        plotone(1),
        plotone(2),
        plotone(3),
        plotone(4),
        layout = grid(1, 4, widths = [0.1, 0.2, 0.3, 0.4]),
        size = (640, 260),
    )
end
plot_example_zigzag_samples(output::AbstractString) =
    savefig(plot_example_zigzag_samples(), output)

function plot_example_bilayer_samples()
    plotone(k, θ; kwargs...) =
        plot_lattice(armchair_bilayer_hexagon(k, rotate = θ); markersize = 3.0, kwargs...)
    plot(
        plotone(1, 0, title = raw"$\theta=0\degree$"),
        plotone(1, 10, title = raw"$\theta=10\degree$"),
        plotone(1, 20, title = raw"$\theta=20\degree$"),
        plotone(1, 30, title = raw"$\theta=30\degree$"),
        plotone(2, 0),
        plotone(2, 10),
        plotone(2, 20),
        plotone(2, 30),
        layout = grid(2, 4, heights = [0.30, 0.7]),
        size = (600, 200),
        dpi = 150,
    )
end
plot_example_bilayer_samples(output::AbstractString) =
    savefig(plot_example_bilayer_samples(), output)

function slater_koster_prb_86_125413(
    rᵢⱼ::NTuple{3, Float64};
    γ₀::Real = 2.7,
    γ₁::Real = 0.48,
    a₁::Real = graphene_Rs[3][3],
    a::Real = 1,
)
    # a = 1.418 Å            (nearest-neighbor distance)
    # a₀ = 2.456 Å = √3 a    (next-nearest-neighbor distance)
    # a₁ = 3.349 Å           (interlayer distance)
    # γ₀ = 2.7 eV            (first-neighbors interaction)
    # γ₀' = 0.1 γ₀           (next-nearest-neighbor interaction)
    # γ₁ = 0.48 eV
    r = norm(rᵢⱼ)

    # r_c = 2.5 a₀ = 2.5 √3 a
    # l_c = 0.265 Å = (0.265 / 1.418) a
    Fc = let r_c = 2.5 * sqrt(3) * a, l_c = (0.265 / 1.418) * a
        1 / (1 + exp((r - r_c) / l_c))
    end
    # q_σ / a₁ = q_π / a = 2.218 Å⁻¹
    Vₚₚσ = let q_σ = 2.218 * 1.418 * a₁
        γ₁ * exp(q_σ * (1 - r / a₁)) * Fc
    end
    Vₚₚπ = let q_π = 2.218 * 1.418 * a
        -γ₀ * exp(q_π * (1 - r / a)) * Fc
    end

    n = rᵢⱼ[3] / r
    return n^2 * Vₚₚσ + (1 - n^2) * Vₚₚπ
end


function nearest_neighbor_hamiltonian(sites::Lattice, t₁::Real)
    edges = nearest_neighbours(sites)
    H = zeros(Float64, length(sites), length(sites))
    for (i, j) in edges
        H[i, j] = -t₁
        H[j, i] = -t₁
    end
    H
end

"""
    density_of_states(eigenvalues::AbstractVector{<:Real}; σ::Real) -> Tuple{AbstractRange, Function}
    density_of_states(hamiltonian::Hermitian; σ::Real) -> Tuple{AbstractRange, Function}
    density_of_states(hamiltonian::AbstractMatrix; σ::Real) -> Tuple{AbstractRange, Function}
    density_of_states(hamiltonian::AbstractString; dataset::AbstractString, σ::Real) -> Tuple{AbstractRange, Function}

Calculate density of states.
"""
function density_of_states(eigenvalues::AbstractVector{<:Real}; σ::Real = 1e-1)
    # NOTE: We divide by the number of eigenvalues here to ensure proper normalization
    gaussian(x, μ) =
        1 / (sqrt(2π) * σ * length(eigenvalues)) * exp(-1 / 2 * ((x - μ) / σ)^2)
    Eₘᵢₙ = first(eigenvalues)
    Eₘᵢₙ -= 1 / 10 * abs(Eₘᵢₙ) # Make the range slightly larger to see DoS go to zero
    Eₘₐₓ = last(eigenvalues)
    Eₘₐₓ += 1 / 10 * abs(Eₘₐₓ)
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

function plot_density_of_states(
    Es::AbstractRange,
    dos::Function;
    output::Union{AbstractString, Nothing} = nothing,
    kwargs...,
)
    p = plot(
        Es,
        dos.(Es),
        xlabel = raw"$E\,,\;\mathrm{eV}$",
        ylabel = raw"DoS",
        fontfamily = "computer modern",
        lw = 2,
        label = nothing,
        size = (640, 480),
        dpi = 150;
        kwargs...,
    )
    if isnothing(output)
        return p
    else
        savefig(p, output)
        return nothing
    end
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

"""
    nearest_neighbor_distances(lattice::Lattice, n::Integer; scale::Real = 2) -> Vector{Float64}

Get distances to the first `n` nearest neighbors.
"""
function nearest_neighbor_distances(lattice::Lattice, n::Integer; scale::Real = 2)
    #! format: off
    lattice |>
        v -> Iterators.drop(v, 1) |>
        v -> Iterators.map(i -> norm(first(lattice).position .- i.position), v) |>
        v -> Iterators.map(d -> round(d; digits = 10), v) |>
        v -> Iterators.filter(d -> d < scale * n, v) |>
        collect |> sort |> unique |>
        v -> Iterators.take(v, n) |>
        collect
    #! format: on
end

"""
    graphene_hamiltonian_from_dft(lattice::Lattice) -> Matrix

Construct tight-binding Hamiltonian for monolayer graphene using hoppings obtained from ab
initio calculations performed by Malte Roesner.
"""
function graphene_hamiltonian_from_dft(lattice::Lattice)
    # Raw data from ab initio calculations
    ts = [
        -2.433921, # onsite
        -2.868647, # nearest neighbor
        0.23901, # next-nearest neighbor
        -0.263193, # etc.
        0.023851,
        0.051767,
        -0.020566,
        -0.015075,
        -0.020681,
    ]

    distances = nearest_neighbor_distances(lattice, 8)
    H = zeros(Float64, length(lattice), length(lattice))
    for j in 1:size(H, 2)
        H[j, j] = ts[1]
        for i in (j + 1):size(H, 1)
            r = norm(lattice[j].position .- lattice[i].position)
            index = findfirst(d -> d ≈ r, distances)
            if !isnothing(index)
                H[i, j] = ts[index + 1]
                H[j, i] = ts[index + 1]
            end
        end
    end
    H
end

"""
    bilayer_graphene_hamiltonian_from_dft(lattice::Lattice) -> Matrix

Construct tight-binding Hamiltonian for bilayer graphene using hoppings obtained from ab
initio calculations performed by Malte Roesner. Interlayer hoppings are modelled using
Slater-Koster parametrization from PRB 99, 205134.
"""
function bilayer_graphene_hamiltonian_from_dft(
    lattice::Lattice;
    r₀::Real = graphene_Rs[3][3],
)
    tᵢₙₜᵣₐ = [
        -0.991157, # onsite
        -2.85655, # nearest neighbor
        0.244362, # next-nearest neighbor
        -0.25776, # etc.
        0.024267,
        0.051976,
        -0.020609,
        -0.01445,
        -0.021728,
    ]
    dᵢₙₜᵣₐ = nearest_neighbor_distances(filter(i -> i.sublattice <= 2, lattice), 8)

    tᵢₙₜₑᵣ = [0.290394, 0.117477, 0.066704]
    dᵢₙₜₑᵣ = [r₀, sqrt(r₀^2 + 1), sqrt(r₀^2 + 1)]

    @. prb_99_205134(r, p) = tᵢₙₜₑᵣ[1] * exp(-p[1] * (r - r₀))
    fit = curve_fit(prb_99_205134, dᵢₙₜₑᵣ, tᵢₙₜₑᵣ, [1.0])

    H = zeros(Float64, length(lattice), length(lattice))
    for j in 1:size(H, 2)
        H[j, j] = tᵢₙₜᵣₐ[1]
        for i in (j + 1):size(H, 1)
            r = norm(lattice[j].position .- lattice[i].position)
            if isapprox(lattice[i].position[3] - lattice[j].position[3], 0; atol = 1e-5)
                index = findfirst(d -> d ≈ r, dᵢₙₜᵣₐ)
                if !isnothing(index)
                    H[i, j] = tᵢₙₜᵣₐ[index + 1]
                    H[j, i] = tᵢₙₜᵣₐ[index + 1]
                end
            else
                t = prb_99_205134(r, fit.param)
                H[i, j] = t
                H[j, i] = t
            end
        end
    end
    tₘₐₓ = mapreduce(abs, max, H)
    @inbounds for j in 1:size(H, 2)
        @simd for i in 1:size(H, 1)
            if abs(H[i, j]) < tₘₐₓ * eps(eltype(H))
                H[i, j] = zero(eltype(H))
            end
        end
    end
    H
end


function slater_koster_hamiltonian(lattice::AbstractVector{<:SiteInfo}, Δe::Real = 0)
    H = zeros(Float64, length(lattice), length(lattice))
    @inbounds for j in 1:size(H, 2)
        H[j, j] = Δe
        for i in (j + 1):size(H, 1)
            rᵢⱼ = lattice[j].position .- lattice[i].position
            tᵢⱼ = slater_koster_prb_86_125413(rᵢⱼ)
            H[i, j] = tᵢⱼ
            H[j, i] = tᵢⱼ
        end
    end
    tₘₐₓ = mapreduce(abs, max, H)
    @inbounds for j in 1:size(H, 2)
        @simd for i in 1:size(H, 1)
            if abs(H[i, j]) < tₘₐₓ * eps(eltype(H))
                H[i, j] = zero(eltype(H))
            end
        end
    end
    H
end

# function bilayer_graphene_hamiltonian(k::Int, θ::Real = 0)
#     lattice = armchair_bilayer_hexagon(k, rotate = θ)
#     slater_koster_hamiltonian(lattice)
# end
# function bilayer_graphene(
#     k::Int,
#     output::AbstractString;
#     θ::Real = 0.0,
#     dataset::AbstractString = "/H",
# )
#     hamiltonian = bilayer_graphene_hamiltonian(k, θ)
#     folder = dirname(output)
#     if !isdir(folder)
#         mkpath(folder)
#     end
#     h5open(io -> io[dataset] = hamiltonian, output, "w")
#     nothing
# end
# bilayer_graphene_3252(output; kwargs...) = bilayer_graphene(10, output; kwargs...)

# function plot_bilayer_graphene_density_of_states(
#     output::Union{AbstractString, Nothing} = nothing,
# )
#     plotone(θ; kwargs...) = plot_density_of_states(
#         density_of_states(bilayer_graphene_hamiltonian(10, θ); σ = 0.15)...;
#         kwargs...,
#     )
#     p = plot(
#         plotone(0, xlabel = "", title = raw"$\theta = 0\degree$"),
#         plotone(10, xlabel = "", ylabel = "", title = raw"$\theta = 10\degree$"),
#         plotone(20, title = raw"$\theta = 20\degree$"),
#         plotone(30, ylabel = "", title = raw"$\theta = 30\degree$"),
#         layout = grid(2, 2),
#         size = (640, 480),
#     )
#     if isnothing(output)
#         return p
#     else
#         savefig(p, output)
#         return nothing
#     end
# end

# function single_layer_graphene_zigzag_1633(
#     output::AbstractString = nothing;
#     t₁::Real = 2.7,
#     dataset::AbstractString = "/H",
# )
#     lattice = zigzag_hexagon(16)
#     @assert length(lattice) == 1633
#     hamiltonian = build_hamiltonian(lattice, t₁)
#     folder = dirname(output)
#     if !isdir(folder)
#         mkpath(folder)
#     end
#     h5open(io -> io[dataset] = hamiltonian, output, "w")
#     nothing
# end
