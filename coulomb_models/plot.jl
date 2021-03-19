using FFTW
using HDF5
using LinearAlgebra
using Plots

pgfplotsx()
theme(
    :solarized_light;
    fg = RGB(88 / 255, 110 / 255, 117 / 255),
    fgtext = RGB(7 / 255, 54 / 255, 66 / 255),
    fgguide = RGB(7 / 255, 54 / 255, 66 / 255),
    fglegend = RGB(7 / 255, 54 / 255, 66 / 255),
)

function transform_to_real_space(
    filename::AbstractString;
    qpoints::Int = 18,
    sublattices::Int = 2,
)
    (U_q, R₁, R₂, rs) = h5open(filename, "r") do io
        # permutedims is needed to account for column-major order used by Julia
        shape = (qpoints, qpoints, sublattices, sublattices)
        U_q = reshape(permutedims(read(io["u_q"]), (3, 2, 1)), shape)
        # Symmetrize U_q!
        U_q = (U_q .+ permutedims(U_q, (2, 1, 3, 4))) ./ 2
        (R₁, R₂, _) = mapslices(R -> tuple(R...), read(io["r_lattice"]), dims = [1])
        rs = reshape(
            mapslices(r -> tuple(r...), read(io["orbital_positions"]), dims = [1]),
            :,
        )
        return U_q, R₁, R₂, rs
    end
    _U_r = ifft(U_q, [1, 2])
    @assert all(@. abs(imag(_U_r)) < 1e-4)
    U_r = real.(_U_r)

    δR = similar(U_r, eltype(U_r))
    for b in 1:length(rs)
        for a in 1:length(rs)
            δr = rs[b] .- rs[a]
            for j in 1:size(U_r, 2)
                for i in 1:size(U_r, 1)
                    δR[i, j, a, b] = norm(@. i * R₁ + j * R₂ + δr)
                end
            end
        end
    end
    return U_r, δR, R₁, R₂, rs
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
