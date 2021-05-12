using FFTW
using HDF5
using LinearAlgebra
using Plots
using LsqFit


function setup_plots!()
    pgfplotsx()
    theme(
        :solarized_light;
        fg = RGB(88 / 255, 110 / 255, 117 / 255),
        fgtext = RGB(7 / 255, 54 / 255, 66 / 255),
        fgguide = RGB(7 / 255, 54 / 255, 66 / 255),
        fglegend = RGB(7 / 255, 54 / 255, 66 / 255),
    )
end

function transform_to_real_space(
    filename::AbstractString;
    qpoints::Int = 18,
    sublattices::Int = 2,
)
    (U_q, R₁, R₂, rs, ref_U_r) = h5open(filename, "r") do io
        shape = (sublattices, sublattices, qpoints, qpoints)
        # permutedims is needed to account for column-major order used by Julia
        U_q = reshape(read(io, "u_q"), shape)
        U_q = permutedims(U_q, (4, 3, 2, 1))
        # NOTE: We symmetrize u_r because interaction between A->B and B->A should really be
        # the same!
        U_q = (U_q .+ permutedims(U_q, (1, 2, 4, 3))) ./ 2
        ref_U_r = haskey(io, "u_r") ? read(io, "u_r") : nothing
        if !isnothing(ref_U_r)
            ref_U_r = reshape(ref_U_r, shape)
            # NOTE: We symmetrize u_r because interaction between A->B and B->A should really be
            # the same!
            ref_U_r = permutedims(ref_U_r, (4, 3, 2, 1))
            ref_U_r = (ref_U_r .+ permutedims(ref_U_r, (1, 2, 4, 3))) ./ 2
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
    # @info maximum(@. abs(imag(_U_r)))
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
                    δR[i, j, a, b] = norm(@. (i - 1) * R₁ + (j - 1) * R₂ + δr)
                end
            end
        end
    end
    return U_r, δR, R₁, R₂, rs
end

function ohno_potential_fit(r, U)
    # e / 4πε₀ in eV⋅Å
    scale = 14.39964547842567
    @. model(r, p) = scale * (p[1] / r + p[2] / r^3)
    mask = 0.1 .< r .< 10
    fit = curve_fit(model, r[mask], U[mask], [0.9, -1.0])
    @info "" fit.param
    return x -> model(x, fit.param)
end

function _get_sub_U(a::Integer, b::Integer, δR, U_r)
    r = reshape(view(δR, :, :, a, b), :)
    U = reshape(view(U_r, :, :, a, b), :)
    order = sortperm(r)
    r = r[order]
    U = U[order]
    # for i in 1:(length(order) - 1)
    #     if r[i] ≈ r[i + 1] && U[i] ≉ U[i + 1]
    #         @warn "Conflicting values for U($(r[i])): $(U[i]) vs. $(U[i + 1])"
    #     end
    # end
    r, U
end

function bilayer_graphene_coulomb_model(
    filename::AbstractString = "data/03_BL_AB/H_25_K_18_B_128_d_3.35/03_cRPA/uqr.h5",
)
    U_r, δR, R₁, R₂, rs = transform_to_real_space(filename; sublattices = 4)
    _get_table(a, b) = sortslices(
        [reshape(view(δR, :, :, a, b), :) reshape(view(U_r, :, :, a, b), :)],
        dims = 1,
    )
    p = [plot(), plot(), plot(), plot()]
    sublattices = ["A", "B", "A'", "B'"]
    a = 1
    for b in 1:4
        table = _get_table(a, b)
        fit = nothing
        if b > 2
            full_table = sortslices([_get_table(a, 3); _get_table(a, 4)], dims=1)
            fit = ohno_potential_fit(full_table[:, 1], full_table[:, 2])
        else
            fit = ohno_potential_fit(table[:, 1], table[:, 2])
        end

        rₘᵢₙ = first(table[:, 1]) - 0.3
        if rₘᵢₙ < 0
            rₘᵢₙ = 1.2
        end
        plot!(p[b], rₘᵢₙ:0.01:20, fit.(rₘᵢₙ:0.01:20), lw = 3, label = b > 1 ? "" : raw"Fit")
        scatter!(p[b],
            table[:, 1],
            table[:, 2],
            markersize = 4,
            ylabel = b % 2 == 0 ? "" : raw"$U\,,\;\mathrm{eV}$",
            xlabel = b <= 2 ? "" : raw"$r\,,\;\AA$",
            label = b > 1 ? "" : "cRPA",
            title = "$(sublattices[a])-$(sublattices[b])",
        )
    end
    plot(
        p...,
        layout = grid(2, 2),
        fontfamily = "computer modern",
        legend = :topright,
        size = (900, 700),
        xlims = (0, 20),
        ylims = (0, 10),
        left_margin = 3mm,
        bottom_margin = 1mm,
    )
end

function analyze_in_real_space(U_r, δR, R₁, R₂, rs)
    rᵃᵃ, Uᵃᵃ = _get_sub_U(1, 1, δR, U_r)
    fit = ohno_potential_fit(rᵃᵃ, Uᵃᵃ)
    # rᵃᵇ, Uᵃᵇ = _get_sub_U(1, 2, δR, U_r)
    g = plot(
        ylabel = raw"$U\,,\;\mathrm{eV}$",
        xlabel = raw"$r\,,\;\AA$",
        fontfamily = "computer modern",
        legend = :topright,
        size = (640, 480),
        xlims = (0, 26),
    )
    plot!(g, 2.4:0.01:26, fit.(2.4:0.01:26), lw = 3, label = raw"$C_1\cdot r^{-C_2} + C_3$")
    scatter!(g, rᵃᵃ, Uᵃᵃ, markersize = 5, markerstrokewidth = 1.5, label = raw"A-A")
    # scatter!(g, rᵃᵇ, Uᵃᵇ, markersize = 3, label = "A-B")
    g
end

function visualize_in_real_space(U_r, δR, R₁, R₂, rs)
    function get_U(a, b)
        r = reshape(view(δR, :, :, a, b), :)
        order = sortperm(r)
        r[order], reshape(view(U_r, :, :, a, b), :)[order]
    end

    rᵃᵃ, Uᵃᵃ = get_U(1, 1)
    rᵃᵇ, Uᵃᵇ = get_U(1, 2)
    g = plot(ylabel = raw"$U$, eV", xlabel = raw"$r$, \r{A}", legend = :topright)
    scatter!(g, rᵃᵃ, Uᵃᵃ, markersize = 3, label = "A-A")
    scatter!(g, rᵃᵇ, Uᵃᵇ, markersize = 3, label = "A-B")
    g
end

function main()
    p = visualize_in_real_space(transform_to_real_space("data/cRPA/uq.h5")...)
    savefig(p, "coulomb_interaction_from_cRPA.png")
end
