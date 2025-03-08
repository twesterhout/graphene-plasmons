import Plasmons.dispersion

function dispersion(
    A::AbstractMatrix,
    qs::AbstractVector{NTuple{3, Float64}},
    lattice::Lattice{3},
    δrs::AbstractVector{NTuple{3, Float64}},
)
    indices = choose_full_unit_cells(lattice; δrs = δrs)
    out = similar(A, complex(eltype(A)), length(qs), length(δrs), length(δrs))
    @inbounds for a in 1:length(δrs)
        idxᵃ = filter(i -> lattice[i].sublattice == a, indices)
        rs = map(i -> i.position, lattice[idxᵃ])
        for b in 1:length(δrs)
            idxᵇ = filter(i -> lattice[i].sublattice == b, indices)
            out[:, a, b] .= dispersion(A[idxᵃ, idxᵇ], qs, rs)
        end
    end
    out
end

function band_structure(
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
    n::Integer = 100,
)
    parts = []
    ticks = []
    scale = minimum((norm(ks[i + 1] .- ks[i]) for i in 1:(length(ks) - 1)))
    offset = 0
    for i in 1:(length(ks) - 1)
        number_points = round(Int, norm(ks[i + 1] .- ks[i]) / scale * n)
        xs = (offset + 1):(offset + number_points)
        ys = band_structure(
            H,
            lattice;
            k₀ = ks[i],
            k₁ = ks[i + 1],
            δrs = δrs,
            n = number_points,
        )
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

function energy_at_dirac_point(H::AbstractMatrix, lattice::Lattice; Gs, δrs)
    K = @. 2 / 3 * Gs[1] + 1 / 3 * Gs[2]
    Hk = dispersion(H, [K], lattice, δrs)[1, :, :]
    sum(eigvals(Hermitian(Hk))) / size(Hk, 1)
end
