using Glob
using Plasmons
import Plasmons.dispersion
using LaTeXStrings
using LinearAlgebra
using DSP
using HDF5
using Plots
using ColorSchemes
# using Dates


# function all_matrices(filenames::AbstractVector{<:AbstractString}; group_name = "/χ")
#     groups = [h5open(f, "r")[group_name] for f in filenames]
#     datasets = []
#     for g in groups
#         for d in g
#             ω = real(read(attributes(d), "ħω"))
#             push!(datasets, (ω, g.file.filename, HDF5.name(d)))
#         end
#     end
#     sort!(datasets; by = x -> x[1])
#     return ((t[1], h5open(io -> read(io[t[3]]), t[2], "r")) for t in datasets)
# end

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

function combine_datasets(
    filenames::AbstractVector{<:AbstractString};
    datasets::AbstractVector{<:AbstractString},
    output::AbstractString,
    dim::Integer = -1,
)
    files = [h5open(f, "r") for f in filenames]
    h5open(output, "w") do io
        for d in datasets
            k = dim < 0 ? ndims(first(files)[d]) + (1 + dim) : dim
            io[d] = cat((read(f, d) for f in files)...; dims = k)
        end
    end
    nothing
end
combine_datasets(pattern::AbstractString; kwargs...) =
    combine_datasets(glob(basename(pattern), dirname(pattern)); kwargs...)

function _dielectric(χ::AbstractMatrix{Complex{ℝ}}, V::AbstractMatrix{ℝ}) where {ℝ <: Real}
    # ℂ = complex(ℝ)
    if size(χ, 1) != size(χ, 2) || size(χ) != size(V)
        throw(DimensionMismatch(
            "dimensions of χ and V do not match: $(size(χ)) != $(size(V)); " *
            "expected two square matrices of the same size",
        ))
    end
    A = V * real(χ)
    @inbounds A[diagind(A)] .-= one(ℝ)
    B = V * imag(χ)
    return @. -(A + 1im * B)
end

function compute_leading_eigenvalues(
    filename::AbstractString,
    V::AbstractMatrix;
    output::AbstractString,
    n::Integer = 1,
)
    h5open(filename, "r") do io
        group = io["/χ"]
        number_frequencies = length(group)
        ωs = similar(V, number_frequencies)
        eigenvalues = similar(V, complex(eltype(V)), n, number_frequencies)
        eigenvectors = similar(V, complex(eltype(V)), size(V, 1), n, number_frequencies)
        densities = similar(V, complex(eltype(V)), size(V, 1), n, number_frequencies)
        for (i, d) in enumerate(io["/χ"])
            ωs[i] = real(read(attributes(d), "ħω"))
            @info "Handling ω = $(ωs[i])..."
            χ = read(d)
            @info "Computing ε..."
            ε = _dielectric(χ, V)
            @info "Computing eigen decomposition of ε..."
            t₀ = time_ns()
            f = eigen!(ε)
            t₁ = time_ns()
            @info "Done in $((t₁ - t₀) / 1e9) seconds."
            @info "Computing loss..."
            for (j, k) in
                enumerate(sortperm(map(z -> -imag(1 / z), f.values), rev = true)[1:n])
                eigenvalues[j, i] = f.values[k]
                eigenvectors[:, j, i] .= view(f.vectors, :, k)
                densities[:, j, i] .= χ * view(f.vectors, :, k)
            end
        end
        h5open(output, "w") do outfile
            outfile["frequencies"] = ωs
            outfile["eigenvalues"] = eigenvalues
            outfile["eigenvectors"] = eigenvectors
            outfile["densities"] = densities
        end
    end
    nothing
end

function compute_V₀_and_Π₀(χ::AbstractMatrix, V::AbstractMatrix)
    @info "Computing ε..."
    ε = _dielectric(χ, V)
    @info "Computing eigen decomposition of ε..."
    t₀ = time_ns()
    f = eigen!(ε)
    t₁ = time_ns()
    @info "Done in $((t₁ - t₀) / 1e9)..."
    @info "Computing loss..."
    index = argmax(map(z -> -imag(1 / z), f.values))
    v₀ = f.vectors[:, index]
    Π₀ = χ * v₀
    V₀ = conj(V * v₀)
    return Π₀, V₀
end
# function compute_V₀_and_Π₀(ħω::Real, θ::Real; output::AbstractString)
#     V = bilayer_graphene_coulomb_model(armchair_bilayer_hexagon(10; rotate = θ))
#     χ = nothing
#     filename = "/vol/tcmscratch04/twesterhout/graphene-plasmons/data/bilayer/output_3252_θ=$(θ)_0.0_1.0.h5"
#     # filename = "/vol/tcmscratch04/twesterhout/graphene-plasmons/data/bilayer/output_3252_θ=$(θ)_0.84_0.88.h5"
#     h5open(filename, "r") do io
#         group = io["/χ"]
#         for (i, d) in enumerate(io["/χ"])
#             @info "" i real(read(attributes(d), "ħω"))
#             if ħω ≈ real(read(attributes(d), "ħω"))
#                 χ = read(d)
#                 break
#             end
#         end
#     end
#     Π₀, V₀ = compute_V₀_and_Π₀(χ, V)
#     h5open(output, "w") do io
#         io["Π₀"] = Π₀
#         io["V₀"] = V₀
#     end
#     nothing
# end
function compute_V₀_and_Π₀(; output::AbstractString)
    prefix = "/vol/tcmscratch04/twesterhout/graphene-plasmons/data/bilayer"
    table = [
        (0, 0.9775, "$prefix/output_3252_θ=0_0.0_1.0.h5"),
        (0, 1.65, "$prefix/output_3252_θ=0_1.63_1.67.h5"),
        (30, 1.695, "$prefix/output_3252_θ=30_1.0_2.0.h5"),
        (30, 0.8566, "$prefix/output_3252_θ=30_0.84_0.88.h5"),
    ]
    for (θ, ħω, filename) in table
        lattice = armchair_bilayer_hexagon(10; rotate = θ)
        V = bilayer_graphene_coulomb_model(lattice)
        χ = nothing
        h5open(filename, "r") do io
            for (i, d) in enumerate(io["/χ"])
                if ħω ≈ real(read(attributes(d), "ħω"))
                    χ = read(d)
                    break
                end
            end
        end
        Π₀, V₀ = compute_V₀_and_Π₀(χ, V)
        savefig(
            plot_eigenvector_bilayer(lattice, Π₀; ω = ħω),
            "$output/Π₀_θ=$(θ)_ω=$(ħω).png",
        )
        savefig(
            plot_eigenvector_bilayer(lattice, V₀; ω = ħω),
            "$output/V₀_θ=$(θ)_ω=$(ħω).png",
        )
    end
    nothing
end

function compute_screened_coulomb_interaction(
    ε::AbstractMatrix{<:Complex},
    V::AbstractMatrix{<:Real},
)
    inverse_ε = inv(ε)
    real(inverse_ε) * V .+ 1im .* (imag(inverse_ε) * V)
end
function compute_screened_coulomb_interaction(
    filename::AbstractString,
    lattice::Lattice,
    ω::Real = 0,
)
    V = bilayer_graphene_coulomb_model(lattice)
    χ = h5open(filename, "r") do io
        for d in io["/χ"]
            if real(read(attributes(d), "ħω")) ≈ ω
                return read(d)
            end
        end
        @assert false
    end
    ε = _dielectric(χ, V)
    W = compute_screened_coulomb_interaction(ε, V)
    # @assert ε ≈ I - V * χ
    # @assert W ≈ V .+ V * χ * W
    @assert W ≈ transpose(W)
    @assert all(@. abs(imag(W)) < 1e-5)
    χ, V, W
    # h5open("$output/W_$(length(lattice))_θ=$(θ).h5", "w") do io
    #     io["W"] = W
    # end
    # nothing
end

function plot_bilayer_screened_coulomb_interaction(θ::Real)
    prefix = "."
    distance(lattice, i, j) = 1.42492 * norm(lattice[i].position .- lattice[j].position)
    g = plot(
        xlabel = raw"$r\,,\;\mathrm{\AA}$",
        ylabel = raw"$W(r)$",
        size = (640, 480),
        dpi = 150,
        fontfamily = "computer modern",
    )
    for k in [6, 8, 10, 12, 14, 16]
        lattice = armchair_bilayer_hexagon(k; rotate = θ)
        site = div(length(lattice), 2) # 546 # 843
        r = [distance(lattice, i, site) for i in 1:length(lattice)]
        indices = sortperm(r)
        r = r[indices]
        W = h5open(
            io -> real(read(io, "W")[:, site][indices]),
            "$prefix/W_$(length(lattice))_θ=0.h5",
            "r",
        )
        plot!(g, r, W, label = length(lattice))
        if k == 10
            V = bilayer_graphene_coulomb_model(lattice)[:, site][indices]
            plot!(g, r, V, color = :black, label = "cRPA")
        end
    end
    g
end

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

# function dispersion(
#     A::AbstractMatrix,
#     qs::AbstractVector{NTuple{3, Float64}},
#     lattice::Lattice{3},
#     δrs::AbstractVector{NTuple{3, Float64}},
# )
#     indices = choose_full_unit_cells(lattice; δrs = δrs)
#     out = similar(A, complex(eltype(A)), length(qs), length(δrs), length(δrs))
#     for a in 1:length(δrs)
#         idxᵃ = filter(i -> lattice[i].sublattice == a, indices)
#         rs = map(i -> i.position, lattice[idxᵃ])
#         for b in 1:length(δrs)
#             idxᵇ = filter(i -> lattice[i].sublattice == b, indices)
#             out[:, a, b] .= dispersion(A[idxᵃ, idxᵇ], qs, rs)
#         end
#     end
#     out
# end
# function dispersion(
#     H::AbstractMatrix,
#     lattice::Lattice{3};
#     k₀::NTuple{3, Float64},
#     k₁::NTuple{3, Float64},
#     δrs::AbstractVector{NTuple{3, Float64}},
#     n::Integer,
# )
#     @assert n > 1
#     qs = map(q -> k₀ .+ (k₁ .- k₀) .* q, 0:(1 / (n - 1)):1)
#     @assert all(qs[1] .≈ k₀) && all(qs[end] .≈ k₁)
#     Hk = dispersion(H, qs, lattice, δrs)
#     Ek = similar(H, real(eltype(H)), size(Hk, 1), size(Hk, 2))
#     for i in 1:size(Hk, 1)
#         Ek[i, :] .= sort!(eigvals(Hermitian(Hk[i, :, :])))
#     end
#     Ek
# end

# function band_structure(
#     H::AbstractMatrix,
#     lattice::Lattice;
#     ks::AbstractVector{NTuple{3, Float64}},
#     δrs::AbstractVector{NTuple{3, Float64}},
#     n::Int = 100,
# )
#     parts = []
#     ticks = []
#     scale = minimum((norm(ks[i + 1] .- ks[i]) for i in 1:(length(ks) - 1)))
#     offset = 0
#     for i in 1:(length(ks) - 1)
#         number_points = round(Int, norm(ks[i + 1] .- ks[i]) / scale * n)
#         xs = (offset + 1):(offset + number_points)
#         ys =
#             dispersion(H, lattice; k₀ = ks[i], k₁ = ks[i + 1], δrs = δrs, n = number_points)
#         if i != length(ks) - 1
#             number_points -= 1
#         end
#         offset += number_points
#         push!(parts, hcat(xs, ys))
#         push!(ticks, xs[1])
#     end
#     push!(ticks, offset)
#     vcat(parts...), ticks
# end

# function graphene_high_symmetry_points(Gs = graphene_Gs)
#     k(i, j) = @. i * Gs[1] + j * Gs[2]
#     ks = [k(0, 0), k(1 / 2, 0), k(2 / 3, 1 / 3), k(0, 0)]
#     ticks = [raw"$\Gamma$", raw"$M$", raw"$K$", raw"$\Gamma$"]
#     ks, ticks
# end

# function energy_at_dirac_point(
#     H::AbstractMatrix,
#     lattice::Lattice;
#     Gs = graphene_Gs,
#     δrs = graphene_δrs,
# )
#     K = @. 2 / 3 * Gs[1] + 1 / 3 * Gs[2]
#     Hk = dispersion(H, [K], lattice, δrs)[1, :, :]
#     sum(eigvals(Hermitian(Hk))) / size(Hk, 1)
# end

function compute_dispersion_relation(
    filename::AbstractString,
    lattice::Lattice;
    n::Integer,
    Gs = graphene_Gs,
    δrs = bilayer_graphene_δrs,
)
    high_symmetry_points, _ = graphene_high_symmetry_points(Gs)
    Γ, K = high_symmetry_points[1], high_symmetry_points[3]
    qs = map(q -> Γ .+ (K .- Γ) .* q, 0:(1 / (n - 1)):1)
    V = bilayer_graphene_coulomb_model(lattice)

    m = h5open(io -> length(io["/χ"]), filename, "r")
    sublattices = 4
    ωs = Vector{Float64}(undef, m)
    χₖ = Array{ComplexF64, 4}(undef, m, n, sublattices, sublattices)
    εₖ = Array{ComplexF64, 4}(undef, m, n, sublattices, sublattices)
    h5open(filename, "r") do io
        for (i, d) in enumerate(io["/χ"])
            ω = real(read(attributes(d), "ħω"))
            @info "Calculating $ω..."
            χ = read(d)
            ε = _dielectric(χ, V)
            χₖ[i, :, :, :] .= dispersion(χ, qs, lattice, δrs)
            εₖ[i, :, :, :] .= dispersion(ε, qs, lattice, δrs)
            ωs[i] = ω
        end
    end
    ωs, χₖ, εₖ
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

function plot_bilayer_graphene_density_of_states(k::Integer; σ::Real = 0.12)
    g = plot(
        xlabel = raw"$E\,,\;\mathrm{eV}$",
        ylabel = raw"DoS",
        fontfamily = "computer modern",
        size = (640, 480),
        dpi = 150,
    )
    for θ in [0, 5, 10, 20, 30]
        lattice = armchair_bilayer_hexagon(k, rotate = θ)
        hamiltonian = bilayer_graphene_hamiltonian_from_dft(lattice)
        Es, dos = density_of_states(hamiltonian, σ = σ)
        plot!(
            g,
            Es,
            dos.(Es);
            fontfamily = "computer modern",
            lw = 1,
            label = raw"$\theta = " * string(θ) * raw"\degree$",
        )
    end
    vline!(g, [-1.3436710579345084])
    g
end


function bilayer_graphene(
    k::Int,
    output::AbstractString;
    shift::Union{Real, Nothing} = nothing,
    θ::Real = 0.0,
    dataset::AbstractString = "/H",
)
    if isnothing(shift)
        lattice = armchair_bilayer_hexagon(k, rotate = 0)
        hamiltonian = bilayer_graphene_hamiltonian_from_dft(lattice)
        shift = -energy_at_dirac_point(hamiltonian, lattice)
    end
    lattice = armchair_bilayer_hexagon(k, rotate = θ)
    hamiltonian = bilayer_graphene_hamiltonian_from_dft(lattice)
    @info "Shifting Hamiltonian with $shift..."
    hamiltonian[diagind(hamiltonian)] .+= shift
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
function padvalid(vector::AbstractVector, n::Int)
    d = length(vector)
    out = similar(vector, d + 2 * n)
    out[(n + 1):(d + n)] .= vector
    out[1:n] .= vector[1]
    out[(d + n + 1):(d + 2 * n)] .= vector[d]
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
function smoothen(vector::AbstractVector, kernel::AbstractVector)
    @assert mod(length(kernel), 2) == 1
    k = length(kernel)
    out = conv(padvalid(vector, div(k, 2)), kernel)[k:(end - (k - 1))]
    @assert size(out) == size(vector)
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
function smoothen(xs::AbstractVector; σ::Real = 3)
    k = min(round(6 * σ, RoundUp), length(xs) - 1)
    if mod(k, 2) == 0
        k += 1
    end
    μ = div(k + 1, 2)
    kernel = similar(xs, eltype(xs), k)
    for x in 1:k
        kernel[x] = exp(-(x - μ)^2 / σ)
    end
    kernel ./= sum(kernel)
    smoothen(xs, kernel)
end

# function _plot_dispersion(
#     matrix,
#     qs,
#     ωs;
#     σ::Union{<:Real, Nothing} = nothing,
#     transform,
#     title,
# )
#     matrix = transform(matrix)
#     matrix = clamp.(matrix, 0, 2000)
#     if !isnothing(σ)
#         matrix = smoothen(matrix, σ = σ)
#     end
#     heatmap(
#         qs ./ π,
#         ωs,
#         matrix,
#         xlabel = raw"$q$, $\pi/a$",
#         ylabel = raw"$\hbar\omega$, eV",
#         title = title,
#     )
# end
# plot_polarizability_dispersion(matrix, qs, ωs; kwargs...) = _plot_dispersion(
#     matrix,
#     qs,
#     ωs;
#     transform = (@. χ -> -imag(χ)),
#     title = L"$-\mathrm{Im}\left[\chi(q, \omega)\right]$",
#     kwargs...,
# )

# function plot_single_layer_graphene_polarizability()
#     χᵃᵃ, χᵃᵇ, qs, ωs =
#         h5open("data/single_layer/polarizability_dispersion_1626_11.h5", "r") do io
#             read(io["χᵃᵃ"]), read(io["χᵃᵇ"]), read(io["qs"]), read(io["ωs"])
#         end
#     p₁ = plot_polarizability_dispersion(χᵃᵃ, qs, ωs, σ = 3)
#     p₂ = plot_polarizability_dispersion(χᵃᵇ, qs, ωs, σ = 3)
#     plot!(p₁, colorbar = false)
#     plot!(p₂, title = nothing)
#     savefig(
#         plot(p₁, p₂, size = (900, 400), dpi = 600),
#         "assets/single_layer/polarizability_dispersion.png",
#     )
#     nothing
# end

# function single_layer_graphene_1626_polarizability_dispersion(
#     filenames::AbstractVector{<:AbstractString};
#     output::AbstractString,
# )
#     χs = []
#     qs = nothing
#     ωs = nothing
#     for sublattices in [(1, 1), (1, 2)]
#         m, qs, ωs = dispersion(
#             all_matrices(filenames),
#             armchair_hexagon(10),
#             sublattices;
#             δrs = graphene_δrs,
#             direction = NTuple{3, Float64}((1, 0, 0)),
#             n = 100,
#         )
#         push!(χs, m)
#     end
#     h5open(output, "w") do io
#         io["qs"] = qs
#         io["ωs"] = ωs
#         io["χᵃᵃ"] = χs[1]
#         io["χᵃᵇ"] = χs[2]
#     end
#     nothing
# end

function plot_eigenvector(lattice::Lattice{3}, vector::AbstractVector; kwargs...)
    x = map(i -> i.position[1], lattice)
    y = map(i -> i.position[2], lattice)
    z = real.(vector)
    scatter(
        x,
        y,
        marker_z = z,
        aspect_ratio = 1,
        markersize = 5,
        markerstrokewidth = 0.1,
        seriescolor = :balance,
        label = "",
        showaxis = false,
        ticks = false,
        size = (480, 480),
        dpi = 150;
        kwargs...,
    )
end
function plot_eigenvector_bilayer(
    lattice::Lattice{3},
    vector::AbstractVector;
    ω::Union{<:Real, Nothing} = nothing,
    title::Union{<:AbstractString, Nothing} = nothing,
    type::Union{<:AbstractString, Nothing} = nothing,
    colorbar = true,
    limit::Union{<:Real, Nothing} = nothing,
    kwargs...,
)
    mask = [lattice[i].position[3] ≈ 0 for i in 1:length(lattice)]
    zₘₐₓ = maximum(abs.(extrema((real(z) for z in vector))))
    xlims = (-1, +1) .+ extrema((i.position[1] for i in lattice))
    ylims = (-1, +1) .+ extrema((i.position[2] for i in lattice))
    color = cgrad(:coolwarm)

    if isnothing(title)
        title =
            isnothing(ω) ? "" :
            raw"$\omega = " * string(round(ω; digits = 4)) * raw"\;\;\mathrm{eV}$"
    end
    p₁ = plot_eigenvector(
        lattice[mask],
        vector[mask];
        title = title,
        colorbar = false,
        # color = color,
        kwargs...,
    )
    if !isnothing(type)
        x = xlims[1] - 3
        y = sum(map(i->i.position[2], lattice[mask])) / length(lattice[mask])
        annotate!(p₁,
            [(x, y, Plots.text(type, 24, :black, :bottom, "computer modern", rotation = 90))]
        )
    end
    p₂ = plot_eigenvector(
        lattice[.!mask],
        vector[.!mask];
        colorbar = colorbar,
        # color = color,
        kwargs...,
    )
    plot(
        p₁,
        p₂,
        xlims = xlims,
        ylims = ylims,
        clims = (-zₘₐₓ, zₘₐₓ),
        layout = grid(1, 2, widths = colorbar ? [0.45, 0.55] : [0.5, 0.5]),
        fontfamily = "computer modern",
        size = (960, 480),
        right_margin = 0mm,
        left_margin = 0mm; # isnothing(type) ? 0mm : 6mm;
        kwargs...,
    )
end
function plot_eigenvector_bilayer(
    lattice::Lattice{3},
    filename::AbstractString;
    output::AbstractString,
)
    frequencies, eigenvectors, densities = h5open(filename, "r") do io
        read(io, "frequencies"), read(io, "eigenvectors"), read(io, "densities")
    end
    for (i, ω) in enumerate(frequencies)
        @info "ω = $ω..."
        savefig(
            plot_eigenvector_bilayer(lattice, eigenvectors[:, 1, i]; ω = ω),
            "$output/eigenvector_" * string(round(Int, 10000 * ω), pad = 6) * ".png",
        )
        # savefig(
        #     plot_eigenvector_bilayer(lattice, densities[:, 1, i]; ω = ω),
        #     "$output/density_" * string(round(Int, 10000 * ω), pad = 6) * ".png",
        # )
    end
    nothing
end
function plot_test_bilayer(
    lattice::Lattice{3},
    filename::AbstractString;
    ω::Real,
    output::AbstractString,
)
    Π₀, V₀ = h5open(filename, "r") do io
        read(io, "Π₀"), read(io, "V₀")
    end
    savefig(
        plot_eigenvector_bilayer(lattice, Π₀; ω = ω),
        "$output/Pi0_" * string(round(Int, 10000 * ω), pad = 6) * ".png",
    )
    savefig(
        plot_eigenvector_bilayer(lattice, V₀; ω = ω),
        "$output/V0_" * string(round(Int, 10000 * ω), pad = 6) * ".png",
    )
    nothing
end
function plot_eels(filename::AbstractString; σ::Real = 10, kwargs...)
    frequencies, eigenvalues = h5open(filename, "r") do io
        read(io, "frequencies"), read(io, "eigenvalues")
    end
    loss = @. -imag(1 / eigenvalues)
    # p = plot(
    #     frequencies,
    #     loss
    # )
    plot(
        frequencies,
        smoothen(reshape(loss, :, 1); σ = σ),
        xlims = (0, 20),
        width = 3,
        xlabel = raw"$\omega\,,\;\mathrm{eV}$",
        ylabel = raw"$-\mathrm{Im}[1/\varepsilon_1]$",
        fontfamily = "computer modern";
        kwargs...,
    )
    # p
end
function plot_eels(; σ::Real = 5, kwargs...)
    filenames = [
        # "data/bilayer/loss_3252_θ=0_0.5_1.0.h5",
        # "data/bilayer/loss_3252_θ=0_0.5_1.0_new.h5",
        # "data/bilayer/loss_3252_θ=0_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=5_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=10_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=20_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=30_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=0_0.0_1.0.h5",
        # "data/bilayer/loss_3252_θ=5_0.0_1.0.h5",
        # "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=10_0.0_1.0.h5",
        # "data/bilayer/loss_3252_θ=20_0.0_1.0.h5",
        # "data/bilayer/loss_3252_θ=30_0.0_1.0.h5",
        # "data/bilayer/loss_3252_θ=0_1.0_2.0.h5",
        # "data/bilayer/loss_3252_θ=5_1.0_2.0.h5",
        # "data/bilayer/loss_3252_θ=10_1.0_2.0.h5",
        # "data/bilayer/loss_3252_θ=20_1.0_2.0.h5",
        # "data/bilayer/loss_3252_θ=30_1.0_2.0.h5",
        # "data/bilayer/loss_3252_θ=0_0.0_1.0.h5",
        # "data/bilayer/loss_3252_θ=5_0.8_1.0.h5",
        # "data/bilayer/loss_3252_θ=10_0.8_1.0.h5",
        # "data/bilayer/loss_3252_θ=20_0.8_1.0.h5",
        # "data/bilayer/loss_3252_θ=30_0.0_1.0.h5",
        # "data/bilayer/loss_3252_θ=30_0.84_0.88.h5",
        # "data/bilayer/loss_3252_θ=0.h5",
        # "data/bilayer/loss_3252_θ=10.h5",
        # "data/bilayer/loss_3252_θ=20.h5",
        # "data/bilayer/loss_3252_θ=30.h5"
        "paper/analysis/combined_loss_k=10_θ=0.h5",
        # "paper/analysis/loss_k=10_θ=30_0.0_2.0.h5",
    ]
    labels = hcat([raw"$\theta = 0\degree$"
    # raw"$\theta = 5\degree$"
    # raw"$\theta = 10\degree$"
    # raw"$\theta = 20\degree$"
    # raw"$\theta = 30\degree$"
]...)
    lines = nothing # [0.7749] # [0.9775 0.985 0.93 1.1925 0.8575]
    p = plot(
        xlabel = raw"$\omega\,,\;\mathrm{eV}$",
        ylabel = raw"$-\mathrm{Im}[1/\varepsilon_1]$",
        size = (640, 480),
        dpi = 150,
        fontfamily = "computer modern",
    )

    for i in 1:length(filenames)
        frequencies, eigenvalues = h5open(
            io -> (read(io, "frequencies"), read(io, "eigenvalues")),
            filenames[i],
            "r",
        )
        eigenvalues = permutedims(eigenvalues)[:, 1]

        indices = sortperm(frequencies)
        frequencies = frequencies[indices]
        eigenvalues = eigenvalues[indices]

        mask = @. (frequencies >= 0.0) & (frequencies <= 1.0)
        frequencies = frequencies[mask]
        eigenvalues = eigenvalues[mask]

        loss = @. -imag(1 / eigenvalues)
        scale = maximum(loss) / maximum(abs.(real.(eigenvalues)))
        # loss = hcat([smoothen(loss[:, i]; σ = σ) for i in 1:size(loss, 2)]...)
        plot!(
            p,
            frequencies,
            hcat(loss, scale .* real.(eigenvalues)),
            width = [2 1],
            color = [i i],
            alpha = [1.0 0.5],
            labels = [labels[i] ""],
            xlabel = raw"$\omega\,,\;\mathrm{eV}$",
            ylabel = raw"$-\mathrm{Im}[1/\varepsilon_1]$",
            # markershape = :circle,
            size = (640, 480),
            dpi = 150,
            fontfamily = "computer modern";
            kwargs...,
        )
        if !isnothing(lines)
            vline!(p, [lines[i]], color = :black, label = "")
        end
    end
    p


    # frequencies = h5open(io->read(io, "frequencies"), first(filenames), "r")
    # eigenvalues = hcat([h5open(io->read(io, "eigenvalues"), f, "r") for f in filenames]...)
    # loss = @. -imag(1 / eigenvalues)
    # # loss = hcat([smoothen(loss[:, i]; σ = σ) for i in 1:size(loss, 2)]...)
    # p = plot(
    #     frequencies,
    #     # loss,
    #     hcat(loss, 100 .* real.(eigenvalues[:, 1])),
    #     # xlims = (-0.5, 20),
    #     width = [2 1],
    #     labels = labels,
    #     xlabel = raw"$\omega\,,\;\mathrm{eV}$",
    #     ylabel = raw"$-\mathrm{Im}[1/\varepsilon_1]$",
    #     size = (640, 480),
    #     dpi = 150,
    #     fontfamily = "computer modern";
    #     kwargs...
    # )
    # vline!(p, [1.665])
    # p
end
function plot_our_eels(; σ::Real = 5, kwargs...)
    filenames = [
        # "data/bilayer/loss_3252_θ=0_0.5_1.0.h5",
        # "data/bilayer/loss_3252_θ=0_0.5_1.0_new.h5",
        # "data/bilayer/loss_3252_θ=0_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=5_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=10_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=20_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=30_1.63_1.67.h5",
        # "data/bilayer/loss_3252_θ=0_1.0_2.0.h5",
        # "data/bilayer/loss_3252_θ=5_1.0_2.0.h5",
        # "data/bilayer/loss_3252_θ=10_1.0_2.0.h5",
        # "data/bilayer/loss_3252_θ=5_0.5_1.0.h5",
        "data/bilayer/loss_3252_θ=0.h5",
        "data/bilayer/loss_3252_θ=10.h5",
        "data/bilayer/loss_3252_θ=20.h5",
        "data/bilayer/loss_3252_θ=30.h5",
    ]
    labels = hcat([
        raw"$\theta = 0\degree$"
        # raw"$\theta = 5\degree$"
        raw"$\theta = 10\degree$"
        raw"$\theta = 20\degree$"
        raw"$\theta = 30\degree$"
        ""
    ]...) #  raw"$\theta = 30\degree$"]
    frequencies = h5open(io -> read(io, "frequencies"), first(filenames), "r")
    eigenvalues =
        hcat([h5open(io -> read(io, "eigenvalues"), f, "r") for f in filenames]...)
    loss = @. -imag(1 / eigenvalues)
    loss = hcat([smoothen(loss[:, i]; σ = σ) for i in 1:size(loss, 2)]...)
    plot(
        frequencies,
        loss,
        # hcat(loss, 1000 .* real.(eigenvalues[:, 1])),
        xlims = (-0.5, 20),
        width = 2,
        labels = labels,
        xlabel = raw"$\omega\,,\;\mathrm{eV}$",
        ylabel = raw"$-\mathrm{Im}[1/\varepsilon_1]$",
        size = (640, 480),
        dpi = 150,
        fontfamily = "computer modern";
        kwargs...,
    )
    # p
end

function plot_plasmon_dispersion(
    filename::AbstractString = "dispersion_3252_θ=0.h5";
    variable = :χ,
)
    dataset = variable == :χ ? "polarizability_dispersion" : "dielectric_dispersion"
    title =
        variable == :χ ? raw"$\mathrm{Im}[\chi(\hbar\omega, q)]$" :
        raw"$-\mathrm{Im}[1/\varepsilon(\hbar\omega, q)]$"
    ωs, εₖ = h5open(io -> (read(io, "frequencies"), read(io, dataset)), filename, "r")
    qs = 0:(1 / (size(εₖ, 2) - 1)):1
    function preprocess(A)
        # -imag(1 / A[1, 1])
        eigenvalues = eigvals(A)
        if variable == :χ
            sum(@. imag(eigenvalues))
        else
            maximum(@. -imag(1 / eigenvalues))
        end
    end
    data = zeros(real(eltype(εₖ)), size(εₖ)[1:2]...)
    for j in 1:size(data, 2)
        for i in 1:size(data, 1)
            data[i, j] = preprocess(εₖ[i, j, :, :])
        end
    end
    heatmap(
        qs,
        ωs,
        data,
        xticks = ([0.0, 1.0], [raw"$\Gamma$", raw"$K$"]),
        xlabel = raw"$q$",
        ylabel = raw"$\hbar\omega\,,\;\mathrm{eV}$",
        color = cgrad(:heat),
        clims = (0, 7),
        title = title,
        xtickfontsize = 12,
        size = (640, 480),
        dpi = 150,
        fontfamily = "computer modern",
        right_margin = 2mm,
        left_margin = 2mm,
    )
end
