function generate_input_file(k::Integer, θ::Real, filename::AbstractString; shift::Real)
    lattice = armchair_bilayer_hexagon(k, rotate = θ)
    hamiltonian = bilayer_graphene_hamiltonian_from_dft(lattice)
    hamiltonian[diagind(hamiltonian)] .+= shift
    h5open(io -> io["H"] = hamiltonian, filename, "w")
    nothing
end
function generate_input_files(
    output::AbstractString = joinpath(@__DIR__, "..", "paper", "input"),
)
    k = 10
    θs = [0, 5, 10, 20, 30]

    if !isdir(output)
        mkpath(output)
    end
    shift =
        let l = armchair_bilayer_hexagon(k, rotate = 0),
            h = bilayer_graphene_hamiltonian_from_dft(l)

            -energy_at_dirac_point(h, l, Gs = graphene_Gs, δrs = bilayer_graphene_δrs)
        end
    for θ in θs
        @info "Generating input file for θ=$(θ)°..."
        generate_input_file(
            k,
            θ,
            joinpath(output, "bilayer_graphene_k=$(k)_θ=$(θ).h5"),
            shift = shift,
        )
    end
end

function plot_density_of_states(
    input::AbstractString = joinpath(@__DIR__, "..", "paper", "input");
    σ::Real = 0.12,
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "density_of_states.pdf",
    ),
)
    filenames =
        filter(s -> !isnothing(match(r"bilayer_graphene.*_θ=.+\.h5", s)), readdir(input))
    table = []
    for f in filenames
        θ = parse(Int, match(r"θ=([^._]+)", f).captures[1])
        H = h5open(io -> read(io, "H"), joinpath(input, f), "r")
        push!(table, (θ, H))
    end
    sort!(table, by = t -> t[1])

    g = plot(
        xlabel = raw"$E\,,\;\mathrm{eV}$",
        ylabel = raw"Density of States",
        fontfamily = "computer modern",
        palette = :Set2_8,
        size = (600, 400),
        dpi = 150,
    )
    for (θ, H) in table
        Es, dos = density_of_states(H, σ = σ)
        plot!(g, Es, dos.(Es); lw = 2, label = raw"$\theta = " * string(θ) * raw"\degree$")
    end
    if !isnothing(output)
        savefig(g, output)
    end
    g
end

function plot_coulomb_model(;
    filename::AbstractString = joinpath(bilayer_cRPA_folder, "uqr.h5"),
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "coulomb_model.pdf",
    ),
)
    U, δR, _, _, _ = transform_to_real_space(filename; sublattices = 4)
    table = sortslices([reshape(δR, :) reshape(U, :)], dims = 1)

    g = plot(
        ylabel = raw"$U\,,\;\mathrm{eV}$",
        xlabel = raw"$r\,,\;\AA$",
        fontfamily = "computer modern",
        palette = :Set2_8,
        legend = :topright,
        xlims = (-0.15, 10.1),
        ylims = (0, 10),
        size = (480, 320),
        dpi = 150,
        left_margin = 3mm,
        bottom_margin = 1mm,
    )

    coulomb = bilayer_graphene_coulomb_model(filename = filename)
    x = 0.0:0.01:20
    plot!(g, x, coulomb.(x), color = 3, lw = 4, label = raw"Fit")

    scatter!(g, table[:, 1], table[:, 2], markersize = 5, label = "cRPA data")
    if !isnothing(output)
        savefig(g, output)
    end
    g
end

function _plot_eels_part(; σ::Union{Real, Nothing} = 2)
    ωs, eigenvalues = h5open(
        io -> (read(io, "frequencies"), read(io, "eigenvalues")),
        joinpath(paper_folder, "analysis", "loss_k=10_θ=0.h5"),
        "r",
    )
    eigenvalues = permutedims(eigenvalues)
    loss = @. -imag(1 / eigenvalues)
    if !isnothing(σ)
        loss = hcat([smoothen(loss[:, i]; σ = σ) for i in 1:size(loss, 2)]...)
    end

    g = plot(
        # ylabel = raw"$\omega\,,\;\mathrm{eV}$",
        xlabel = raw"$-\mathrm{Im}[1/\varepsilon_1(\omega)]$",
        xticks = [0, 20, 40],
        fontfamily = "computer modern",
        xmirror = false,
        ymirror = true,
        palette = :Set2_8,
        legend = :bottomright,
        ylims = (0, 20),
        size = (200, 400),
        dpi = 150,
    )
    plot!(g, loss[:, 1], ωs, lw = 2, label = raw"$\theta = 0\degree$")
    g
end
function _plot_dispersion_part()
    ωs, ε = h5open(
        io -> (read(io, "ω"), read(io, "ε")),
        joinpath(paper_folder, "analysis", "dispersion_k=10_θ=0.h5"),
        "r",
    )
    qs = 0:(1 / (size(ε, 2) - 1)):1

    function preprocess(A)
        eigenvalues = eigvals(A)
        maximum(@. -imag(1 / eigenvalues))
    end

    data = zeros(real(eltype(ε)), size(ε)[1:2]...)
    for j in 1:size(data, 2)
        for i in 1:size(data, 1)
            data[i, j] = preprocess(ε[i, j, :, :])
        end
    end
    cutoff = findfirst(ω -> ω >= 20, ωs) - 1
    data = data[3:cutoff, :]
    ωs = ωs[3:cutoff]
    data = data[:, 3:(end - 2)]
    qs = qs[3:(end - 2)]
    heatmap(
        qs,
        ωs,
        data,
        tick_direction = :out,
        xticks = ([0.0, 1.0], [raw"$\Gamma$", raw"$K$"]),
        framestyle = :box,
        grid = false,
        colorbar = false,
        xlims = (0, 1),
        ylims = (0, 20),
        clims = (0, 3),
        xlabel = raw"$q$",
        ylabel = raw"$\omega\,,\;\mathrm{eV}$",
        color = cgrad(:heat),
        title = raw"$-\mathrm{Im}[1/\varepsilon(\omega, q)]$",
        xtickfontsize = 12,
        fontfamily = "computer modern",
        size = (400, 400),
        dpi = 150,
    )
end
function plot_dispersion_and_eels(
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "dispersion_and_eels.pdf",
    ),
)
    g = plot(
        _plot_dispersion_part(),
        _plot_eels_part(),
        size = (320 + 160, 320),
        layout = grid(1, 2, widths = [320 / 480, 160 / 480]),
        dpi = 200,
    )
    if !isnothing(output)
        savefig(g, output)
    end
    g
end
