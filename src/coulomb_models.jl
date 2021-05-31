using FFTW
using HDF5
using LinearAlgebra
using Plots
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

function multipole_expansion_fit(r, U)
    # e / 4πε₀ in eV⋅Å
    scale = 14.39964547842567
    @. model(r, p) = scale * (p[1] / sqrt(r^2 + p[2]^2)) * (1 + exp(-p[3] * r^2))
    fit = curve_fit(model, r, U, [1.0, 0.1, 1.0])
    @info "" fit.param
    return x -> model(x, fit.param)
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
function image_charge_model_fit(r, U; maxiter::Integer = 20, ε₂::Real = 1, ε₃::Real = 1)
    @. model(r, p) = image_charge_model(r, p[1], p[2], ε₂, ε₃, p[3]; maxiter = maxiter)
    fit = curve_fit(
        model,
        r,
        U,
        [6.7, 2.2, 0.78],
        lower = [1.0, 1.0, 0.01],
        upper = [10.0, 10.0, 2.0],
    )
    @info "d = $(fit.param[1]), ε₁ = $(fit.param[2]), δ = $(fit.param[3])"
    return x -> model(
        x,
        # [6.7, 2.20151908, 0.78159943])
        fit.param,
    )
end
# def get_u_r(r, h, e_m, e_env, d):
#     n_max = 10
#     e2 = 14.3999
#     beta = (e_m - e_env) / (e_m + e_env)
#     
#     def z(r, n, h, d):
#         return np.sqrt(r**2. + d**2.0 + (n*h)**2.)
#     
#     u_r = e2 / (e_m * z(r, 0, h, d))
#     print(u_r)
#     
#     for n in range(1, n_max):
#         print(2. * e2 * beta**n / (e_m * z(r, n, h, d)))
#         u_r += 2. * e2 * beta**n / (e_m * z(r, n, h, d))
#     
#     return u_r

function bilayer_graphene_coulomb_model(
    filename::AbstractString = "data/03_BL_AB/H_25_K_18_B_128_d_3.35/03_cRPA/uqr.h5",
)
    U_r, δR, R₁, R₂, rs = transform_to_real_space(filename; sublattices = 4)
    table = sortslices([reshape(δR, :) reshape(U_r, :)], dims = 1)
    mask = table[:, 1] .< 10
    fit = image_charge_model_fit(table[mask, 1], table[mask, 2]) # multipole_expansion_fit(table[mask, 1], table[mask, 2])
    # @info "" table[1:5, :]
    # r-> abs(r) < 1 ? table[1, 2] : fit(r)
end
function bilayer_graphene_coulomb_model(
    lattice::Lattice{3};
    filename::AbstractString = "data/03_BL_AB/H_25_K_18_B_128_d_3.35/03_cRPA/uqr.h5",
)
    fit = bilayer_graphene_coulomb_model(filename)
    model(i, j) = fit(1.42492 * norm(lattice[i].position .- lattice[j].position))
    V = Matrix{Float64}(undef, length(lattice), length(lattice))
    for j in 1:size(V, 2)
        V[j, j] = model(j, j)
        for i in (j + 1):size(V, 1)
            V[i, j] = model(i, j)
            V[j, i] = V[i, j]
        end
    end
    V
end

function plot_bilayer_graphene_coulomb_model(
    filename::AbstractString = "data/03_BL_AB/H_25_K_18_B_128_d_3.35/03_cRPA/uqr.h5",
)
    U_r, δR, R₁, R₂, rs = transform_to_real_space(filename; sublattices = 4)
    _get_table(a, b) = sortslices(
        [reshape(view(δR, :, :, a, b), :) reshape(view(U_r, :, :, a, b), :)],
        dims = 1,
    )
    p = plot(
        ylabel = raw"$U\,,\;\mathrm{eV}$",
        xlabel = raw"$r\,,\;\AA$",
        fontfamily = "computer modern",
        legend = :topright,
        size = (900, 700),
        xlims = (0, 10),
        ylims = (0, 10),
        left_margin = 3mm,
        bottom_margin = 1mm,
    )
    sublattices = ["A", "B", "A'", "B'"]
    coulomb = bilayer_graphene_coulomb_model(filename)
    x = 0.0:0.01:20
    plot!(p, x, coulomb.(x), color = :black, lw = 4, label = raw"Fit")

    a = 1
    for b in 1:4
        table = _get_table(a, b)
        scatter!(p,
            table[:, 1],
            table[:, 2],
            markersize = 5,
            color = b,
            label = "$(sublattices[a])-$(sublattices[b])",
        )
    end
    p
end
