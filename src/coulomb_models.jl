using FFTW
using HDF5
using LinearAlgebra
using LsqFit

function transform_to_real_space(
    filename::AbstractString;
    qpoints::Int = 18,
    sublattices::Int = 2,
)
    (U_q, R₁, R₂, rs, ref_U_r) = h5open(filename, "r") do io
        shape = (sublattices, sublattices, qpoints, qpoints)
        # permutedims is needed to account for column-major order used by Julia
        U_q = reshape(read(io, "u_q"), shape)
        U_q = permutedims(U_q, (4, 3, 1, 2))
        ref_U_r = haskey(io, "u_r") ? read(io, "u_r") : nothing
        if !isnothing(ref_U_r)
            ref_U_r = reshape(ref_U_r, shape)
            ref_U_r = permutedims(ref_U_r, (4, 3, 2, 1))
        end
        (R₁, R₂, R₃) = mapslices(R -> tuple(R...), read(io, "r_lattice"), dims = [1])
        rs = reshape(
            mapslices(r -> tuple(r...), read(io, "orbital_positions"), dims = [1]),
            :,
        )
        for i in 1:length(rs)
            r = rs[i]
            rs[i] = @. R₁ * r[1] + R₂ * r[2] + R₃ * r[3]
        end
        return U_q, R₁, R₂, rs, ref_U_r
    end
    _U_r = ifft(U_q, [1, 2])
    @assert all(@. abs(imag(_U_r)) < 2e-5)
    if !isnothing(ref_U_r)
        # NOTE: In the following we should not have to use conj!
        @assert _U_r ≈ conj.(ref_U_r)
    end
    U_r = real.(_U_r)

    δR = similar(U_r, eltype(U_r))
    @inbounds for b in 1:length(rs)
        for a in 1:length(rs)
            δr = rs[b] .- rs[a]
            for j in 1:size(U_r, 2)
                for i in 1:size(U_r, 1)
                    δR[i, j, b, a] = norm(@. (i - 1) * R₁ + (j - 1) * R₂ + δr)
                end
            end
        end
    end
    return U_r, δR, R₁, R₂, rs
end

function image_charge_model(ρ, d, ε₁, ε₂, ε₃, δ::Real; maxiter::Integer)
    # e / 4πε₀ in eV⋅Å
    scale = 14.39964547842567
    L₁₂ = (ε₁ - ε₂) / (ε₁ + ε₂)
    L₁₃ = (ε₁ - ε₃) / (ε₁ + ε₃)
    t₀ = 1 / sqrt(ρ^2 + δ^2)
    t₁ = sum(
        ((L₁₂ * L₁₃)^n / sqrt(ρ^2 + δ^2 + (2 * n * d)^2) for n in 1:maxiter);
        init = 0.0,
    )
    t₂ = sum(
        ((L₁₂ * L₁₃)^n / sqrt(ρ^2 + δ^2 + ((2 * n + 1) * d)^2) for n in 0:maxiter);
        init = 0.0,
    )
    return scale / ε₁ * (t₀ + 2 * t₁ + (L₁₂ + L₁₃) * t₂)
end
function image_charge_model_fit(
    r,
    U;
    d::Real,
    maxiter::Integer = 20,
    ε₂::Real = 1,
    ε₃::Real = 1,
)
    @. model(r, p) = image_charge_model(r, d, p[1], ε₂, ε₃, p[2]; maxiter = maxiter)
    fit = curve_fit(model, r, U, [2.0, 0.5])
    @info "d = $d, ε₁ = $(fit.param[1]), δ = $(fit.param[2])"
    return x -> model(
        x,
        # [6.7, 2.20151908, 0.78159943])
        fit.param,
    )
end

function bilayer_graphene_coulomb_model(;
    filename::AbstractString, # = joinpath(bilayer_cRPA_folder, "uqr.h5"),
    cutoff::Real = 10,
)
    U_r, δR, R₁, R₂, rs = transform_to_real_space(filename; sublattices = 4)
    table = sortslices([reshape(δR, :) reshape(U_r, :)], dims = 1)
    mask = table[:, 1] .< cutoff # Only use points up to 10 Å.
    image_charge_model_fit(table[mask, 1], table[mask, 2], d = 6.7)
end
function bilayer_graphene_coulomb_model(lattice::Lattice{3}; kwargs...)
    fit = bilayer_graphene_coulomb_model(; kwargs...)
    model(i, j) = fit(1.42492 * norm(lattice[i].position .- lattice[j].position))
    V = Matrix{Float64}(undef, length(lattice), length(lattice))
    @inbounds Threads.@threads for j in 1:size(V, 2)
        V[j, j] = model(j, j)
        for i in (j + 1):size(V, 1)
            V[i, j] = model(i, j)
            V[j, i] = V[i, j]
        end
    end
    V
end
bilayer_graphene_coulomb_model(k::Integer, θ::Real; kwargs...) =
    bilayer_graphene_coulomb_model(armchair_bilayer_hexagon(k; rotate = θ); kwargs...)

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

function compute_screened_coulomb_interaction(
    ε::AbstractMatrix{<:Complex},
    V::AbstractMatrix{<:Real},
)
    inverse_ε = inv(ε)
    real(inverse_ε) * V .+ 1im .* (imag(inverse_ε) * V)
end
