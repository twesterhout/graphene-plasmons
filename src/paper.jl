function generate_input_file(k::Integer, θ::Real, filename::AbstractString; shift::Real)
    lattice = armchair_bilayer_hexagon(k, rotate = θ)
    hamiltonian = bilayer_graphene_hamiltonian_from_dft(lattice)
    hamiltonian[diagind(hamiltonian)] .+= shift
    h5open(io -> io["H"] = hamiltonian, filename, "w")
    nothing
end
function generate_input_files(
    output::AbstractString = joinpath(@__DIR__, "..", "paper", "input");
    μ::Real = -1.3436710579345084,
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
    shift += μ
    for θ in θs
        @info "Generating input file for θ=$(θ)°..."
        generate_input_file(
            k,
            θ,
            joinpath(
                output,
                "bilayer_graphene_k=$(k)_μ=$(round(-μ, digits = 2))_θ=$(θ).h5",
            ),
            shift = shift,
        )
    end
end

function compute_leading_eigenvalues(
    filename::AbstractString;
    output::AbstractString,
    k::Union{Integer, Nothing} = nothing,
    θ::Union{Real, Nothing} = nothing,
    n::Integer = 1,
)
    if isnothing(k)
        k = parse(Int, match(r"k=([^._]+)", filename).captures[1])
    end
    if isnothing(θ)
        θ = parse(Int, match(r"θ=([^._]+)", filename).captures[1])
    end
    V = bilayer_graphene_coulomb_model(k, θ)
    compute_leading_eigenvalues(filename, V; output = output, n = n)
end

function compute_dispersion_relation(
    filename::AbstractString;
    output::AbstractString,
    k::Union{Integer, Nothing} = nothing,
    θ::Union{Real, Nothing} = nothing,
    n::Integer = 400,
)
    if isnothing(k)
        k = parse(Int, match(r"k=([^._]+)", filename).captures[1])
    end
    if isnothing(θ)
        θ = parse(Int, match(r"θ=([^._]+)", filename).captures[1])
        @assert θ == 0
    end
    lattice = armchair_bilayer_hexagon(k, rotate = θ)
    ωs, χₖ, εₖ = compute_dispersion_relation(filename, lattice; n = n)
    h5open(output, "w") do io
        io["ω"] = ωs
        io["χ"] = χₖ
        io["ε"] = εₖ
    end
    nothing
end

function compute_screened_coulomb_interaction(
    filename::AbstractString;
    output::AbstractString,
)
    k = parse(Int, match(r"k=([^._]+)", filename).captures[1])
    θ = parse(Int, match(r"θ=([^._]+)", filename).captures[1])
    lattice = armchair_bilayer_hexagon(k, rotate = θ)
    χ, V, W = compute_screened_coulomb_interaction(filename, lattice)
    h5open(output, "w") do io
        io["χ"] = χ
        io["V"] = V
        io["W"] = W
    end
    nothing
end

function compute_density_of_states(
    input::AbstractString = joinpath(@__DIR__, "..", "paper", "input");
    μ::Real,
    σ::Real = 0.12,
    output::Union{AbstractString, Nothing} = joinpath(paper_folder, "analysis"),
)
    match_fn(s) =
        !isnothing(match(
            r"bilayer_graphene.*_μ=" * string(round(-μ, digits = 2)) * r"_θ=.+\.h5",
            s,
        ))
    filenames = filter(match_fn, readdir(input))
    table = []
    for f in filenames
        θ = parse(Int, match(r"θ=([^._]+)", f).captures[1])
        H = h5open(io -> read(io, "H"), joinpath(input, f), "r")
        push!(table, (θ, H))
    end
    sort!(table, by = t -> t[1])

    output = joinpath(output, first(filenames))
    output = replace(output, "bilayer_graphene" => "density_of_states")
    output = replace(output, r"_?θ=[0-9]+" => "")
    h5open(output, "w") do io
        for (θ, H) in table
            g = create_group(io, string(θ))
            eigenvalues = eigvals(Hermitian(H))
            Es, dos = density_of_states(eigenvalues, σ = σ)
            g["eigenvalues"] = eigenvalues
            g["energies"] = collect(Es)
            g["densities"] = dos.(Es)
        end
    end
    nothing
end

function plot_density_of_states(
    input::AbstractString = joinpath(
        @__DIR__,
        "..",
        "paper",
        "analysis",
        "density_of_states_k=10_μ=1.34.h5",
    );
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "density_of_states_k=10_μ=1.34.pdf",
    ),
)
    # filenames =
    #     filter(s -> !isnothing(match(r"bilayer_graphene.*_θ=.+\.h5", s)), readdir(input))
    # table = []
    # for f in filenames
    #     θ = parse(Int, match(r"θ=([^._]+)", f).captures[1])
    #     H = h5open(io -> read(io, "H"), joinpath(input, f), "r")
    #     push!(table, (θ, H))
    # end
    # sort!(table, by = t -> t[1])

    g = plot(
        xlabel = raw"$E\,,\;\mathrm{eV}$",
        ylabel = raw"Density of States",
        fontfamily = "computer modern",
        palette = :Set2_8,
        size = (600, 400),
        dpi = 150,
    )
    @info input
    h5open(input, "r") do io
        for group in io
            θ = strip(HDF5.name(group), ['/'])
            energies = read(group, "energies")
            densities = read(group, "densities")
            plot!(
                g,
                energies,
                densities;
                lw = 2,
                label = raw"$\theta = " * θ * raw"\degree$",
            )
        end
    end
    if !isnothing(output)
        savefig(g, output)
    end
    g
end
function plot_number_of_states(
    input::AbstractString = joinpath(
        @__DIR__,
        "..",
        "paper",
        "analysis",
        "density_of_states_k=10_μ=0.0.h5",
    );
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "number_of_states_k=10_μ=0.0.pdf",
    ),
    σ::Real = 0.12,
)
    # filenames =
    #     filter(s -> !isnothing(match(r"bilayer_graphene.*_θ=.+\.h5", s)), readdir(input))
    # table = []
    # for f in filenames
    #     θ = parse(Int, match(r"θ=([^._]+)", f).captures[1])
    #     H = h5open(io -> read(io, "H"), joinpath(input, f), "r")
    #     push!(table, (θ, H))
    # end
    # sort!(table, by = t -> t[1])
    function number_of_states(e, ρ)
        n = @. (e[2:end] - e[1:(end - 1)]) * (ρ[2:end] + ρ[1:(end - 1)]) / 2
        return e[2:end], cumsum(n)
    end

    g = plot(
        xlabel = raw"$E\,,\;\mathrm{eV}$",
        ylabel = raw"Number of States",
        fontfamily = "computer modern",
        legend = :bottomright,
        palette = :Set2_8,
        size = (600, 400),
        dpi = 150,
    )
    @info input
    h5open(input, "r") do io
        for group in io
            θ = strip(HDF5.name(group), ['/'])
            eigenvalues = read(group, "eigenvalues")
            energies, densities = density_of_states(eigenvalues, σ = σ)
            # energies = energies[begin]:((energies[end] - energies[begin])/(5000 - 1)):energies[end]
            x, y = number_of_states(energies, densities.(energies))
            y .*= length(eigenvalues)
            plot!(
                g,
                x,
                y;
                xlims = (-0.2, 2),
                ylims = (1500, 2100),
                lw = 2,
                label = raw"$\theta = " * θ * raw"\degree$",
            )
            @info θ
            @info y[findfirst(e -> e >= 0.2, x)]
            @info y[findfirst(e -> e >= 0.5, x)]
            @info y[findfirst(e -> e >= 1.3436710579345084, x)]
        end
    end
    vline!(g, [0.2, 0.5, 1.3436710579345084], color = :black, alpha = 0.5, label = "")
    if !isnothing(output)
        savefig(g, output)
    end
    g
end

function _extract_screened_coulomb_interaction(filename::AbstractString)
    k = parse(Int, match(r"k=([^._]+)", filename).captures[1])
    θ = parse(Int, match(r"θ=([^._]+)", filename).captures[1])
    @assert θ == 0

    lattice = armchair_bilayer_hexagon(k; rotate = θ)
    site = center_site_index(lattice)
    distance(lattice, i, j) = 1.42492 * norm(lattice[i].position .- lattice[j].position)
    r = [distance(lattice, i, site) for i in 1:length(lattice)]
    indices = sortperm(r)
    r = r[indices]
    W = h5open(io -> real(read(io, "W")[:, site][indices]), filename, "r")
    return r, W
end
function plot_static_polarizability(
    filename::AbstractString = joinpath(
        paper_folder,
        "analysis",
        "screened_coulomb_interaction_k=10_μ=1.34_θ=0.h5",
    );
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "static_polarizability.png",
    ),
)
    k = parse(Int, match(r"k=([^._]+)", filename).captures[1])
    θ = parse(Int, match(r"θ=([^._]+)", filename).captures[1])
    @assert θ == 0

    lattice = armchair_bilayer_hexagon(k; rotate = θ)
    plots = []
    for (site, location) in [(center_site_index(lattice), "center"), (1, "edge")]
        χ = h5open(io -> real(read(io, "χ")[:, site]), filename, "r")
        p = plot_eigenvector_bilayer(
            lattice,
            χ,
            title = raw"$\Pi(r, r' = \mathrm{" * location * raw"}, \omega = 0)$",
            titlefontsize = 24,
            clims = (-0.0005, 0.0005),
            colorbar = true,
        )
        push!(plots, p)
    end
    g = plot(plots..., layout = grid(2, 1), size = (960, 960))
    if !isnothing(output)
        savefig(g, output)
    end
    g
end
function plot_coulomb_model(;
    crpa::AbstractString = joinpath(bilayer_cRPA_folder, "uqr.h5"),
    screened::AbstractString = joinpath(
        paper_folder,
        "analysis",
        "screened_coulomb_interaction_k=10_μ=1.34_θ=0.h5",
    ),
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "coulomb_model.pdf",
    ),
)
    U, δR, _, _, _ = transform_to_real_space(crpa; sublattices = 4)
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

    bare(x; ε::Real, δ::Real) = 14.39964547842567 / (ε * sqrt(x^2 + δ^2))

    coulomb = bilayer_graphene_coulomb_model(filename = crpa)
    x = 0.0:0.01:20
    plot!(
        g,
        x,
        bare.(x; δ = 0.7626534576229569, ε = 2.2597469530289955),
        color = 5,
        lw = 4,
        label = raw"Locally screened",
    )
    plot!(g, x, coulomb.(x), color = 3, lw = 4, label = raw"Image charge model")

    scatter!(g, table[:, 1], table[:, 2], color = 2, markersize = 5, label = "cRPA data")
    scatter!(
        g,
        _extract_screened_coulomb_interaction(screened)...,
        color = 4,
        markersize = 5,
        markershape = :diamond,
        label = "Fully screened",
    )
    if !isnothing(output)
        savefig(g, output)
    end
    g
end

function _plot_eels_part(; σ::Union{Real, Nothing} = 2)
    ωs, eigenvalues = h5open(
        io -> (read(io, "frequencies"), read(io, "eigenvalues")),
        joinpath(paper_folder, "analysis.doping_1.34", "combined_loss_k=10_μ=1.34_θ=0.h5"),
        "r",
    )
    eigenvalues = permutedims(eigenvalues)

    indices = sortperm(ωs)
    indices = unique(i->ωs[i], indices)
    ωs = ωs[indices]
    eigenvalues = eigenvalues[indices]

    loss = @. -imag(1 / eigenvalues)
    if !isnothing(σ)
        loss = hcat([smoothen(loss[:, i]; σ = σ) for i in 1:size(loss, 2)]...)
    end

    g = plot(
        # ylabel = raw"$\omega\,,\;\mathrm{eV}$",
        xlabel = raw"$-\mathrm{Im}[1/\varepsilon_1(\omega)]$",
        xticks = [0, 20, 40],
        xlims = (-5, 55),
        fontfamily = "computer modern",
        xmirror = false,
        ymirror = true,
        palette = :Set2_8,
        # legend = :bottomright,
        ylims = (0, 3),
        size = (200, 400),
        dpi = 150,
    )
    plot!(g, loss[:, 1], ωs, lw = 2, label = "") # raw"$\theta = 0\degree$")
    plot!(g,
        [0, 60],
        [0.63 0.9775 1.195 1.65
         0.63 0.9775 1.195 1.65],
        label = "",
        lw = 2,
        color = :black,
        alpha = 0.5,
        linestyle = :dot
    )
    g
end
function _plot_dispersion_part(variable::Symbol = :ε)
    @assert variable == :ε || variable == :χ
    filename = joinpath(paper_folder, "analysis", "dispersion_k=10_μ=1.34_θ=0.h5")
    ωs = h5open(io -> read(io, "ω"), filename, "r")
    iₘₐₓ = findfirst(ω -> ω >= 3, ωs)
    A = h5open(io -> io[string(variable)][1:iₘₐₓ, :, :, :], filename, "r")
    qs = 0:(1 / (size(A, 2) - 1)):1
    jₘₐₓ = findfirst(q -> q >= 0.25, qs)

    ωs = ωs[3:iₘₐₓ]
    qs = collect(qs[3:jₘₐₓ - 2])
    A = A[3:end, 3:jₘₐₓ - 2, :, :]

    function preprocess(M)
        eigenvalues = eigvals(M)
        if variable == :ε
            maximum(@. -imag(1 / eigenvalues))
        else
            sum(@. -imag(eigenvalues))
        end
    end

    data = zeros(real(eltype(A)), size(A)[1:2]...)
    for j in 1:size(data, 2)
        for i in 1:size(data, 1)
            data[i, j] = preprocess(A[i, j, :, :])
        end
    end
    # Rescale qs to Å⁻¹
    Γ, _, K, _ = graphene_high_symmetry_points()[1]
    qs .*= norm(K .- Γ) / 1.424919
    p = heatmap(
        qs,
        ωs,
        data,
        tick_direction = :out,
        # xticks = ([0.0, 1.0], [raw"$\Gamma$", raw"$K$"]),
        # xtickfontsize = 12,
        framestyle = :box,
        grid = false,
        colorbar = false,
        xlims = (0, 0.25 * norm(K .- Γ) / 1.424919),
        ylims = (0, 3),
        clims = variable == :ε ? (0, 3) : (0, 0.4),
        xlabel = raw"$q\,,\;\AA^{-1}$",
        ylabel = raw"$\omega\,,\;\mathrm{eV}$",
        color = cgrad(:OrRd_9), # cgrad(:heat),
        title = variable == :ε ? raw"$-\mathrm{Im}[1/\varepsilon(\omega, q)]$" :
                raw"$-\mathrm{Im}[\Pi(\omega, q)]$",
        fontfamily = "computer modern",
        size = (400, 400),
        dpi = 150,
    )
    plot!(p,
        [0, 0.5],
        [0.63 0.9775 1.195 1.65
         0.63 0.9775 1.195 1.65],
        label = "",
        lw = 2,
        color = :black,
        alpha = 0.5,
        linestyle = :dot
    )
    # scatter!(p,
    #     [0.039, 0.039, 0.079],
    #     [1.65, 0.63, 1.195],
    #     color = palette(:Set2_8),
    #     markersize = 5,
    #     markershape = :circle,
    #     # lw = 5,
    #     # markerstrokewidth = 5,
    #     label = "",
    # )
    p
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

function _extract_eigenmode(ω::Real, filename::AbstractString)
    h5open(filename, "r") do io
        ωs = read(io, "frequencies")
        i = findfirst(x -> real(x) == ω, ωs)
        if isnothing(i)
            throw(ArgumentError("ω = $ω not found in $filename"))
        end
        load(d) = ndims(d) == 2 ? d[:, i] : d[:, 1, i]
        load(io["eigenvectors"]), load(io["densities"])
    end
end
function plot_non_twisted_eigenmodes(;
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "non_twisted_eigenmodes.png",
    ),
)
    k = 10 # parse(Int, match(r"k=([^._]+)", filename).captures[1])
    θ = 0 # parse(Int, match(r"θ=([^._]+)", filename).captures[1])
    lattice = armchair_bilayer_hexagon(k; rotate = θ)

    table = [
        (1.63, 1.67, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=0_1.63_1.67.h5"),
        (0.0, 1.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=0_0.0_1.0.h5"),
        (1.0, 2.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=0_1.0_2.0.h5"),
        (0.0, 22.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=0.h5"),
    ]

    function picture(ω; type = nothing)
        i = findfirst(t -> t[1] <= ω && ω <= t[2], table)
        filename = table[i][3]
        p = plot_eigenvector_bilayer(
            lattice,
            _extract_eigenmode(ω, filename)[1],
            ω = ω,
            titlefontsize = 24,
            colorbar = false,
            type = type,
            left_margin = 0mm,
            right_margin = 0mm,
            markersize = 4.4,
            markerstrokewidth = 0,
            # color = cgrad(:bwr),
        )
        p
    end

    none = plot(ticks=false, border=false, showaxis=false)

    header = plot(xlims = (0, 1), ylims = (0, 1), ticks = false, showaxis = false)
    annotate!(
        header,
        [
            (x, 1.0, Plots.text(t, 24, :top, "computer modern"))
            for (x, t) in [(0.1405, raw"bottom\nlayer"), (0.3765, raw"top\nlayer")]
        ],
    )
    g = plot(
        header,
        # Separator
        none,
        # First column
        picture(0.3225, type = "layer-polarized"),
        picture(1.195, type = "dipole"),
        picture(1.65, type = "1s"),
        none,
        # Separator
        none,
        # Second column
        none,
        picture(0.63),
        picture(0.9775),
        picture(1.265),
        layout = (@layout [header{0.1h}; [_{0.02w} [°; °; °; °] _{0.04w} [°; °; °; °]]]),
        size = (2 * 720 * 1.06, 4 * 360 * 1.1),
        # dpi = 80
    )
    if !isnothing(output)
        savefig(g, output)
    end
    g
end

function plot_one_eigenmode(ω::Real, θ::Real, lattice; type = nothing, U = nothing)
    table = Dict(
        0 => [
            (1.63, 1.67, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=0_1.63_1.67.h5"),
            (0.0, 1.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=0_0.0_1.0.h5"),
            (1.0, 2.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=0_1.0_2.0.h5"),
            (0.0, 22.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=0.h5"),
        ],
        10 => [
            (1.63, 1.67, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=10_1.63_1.67.h5"),
            (0.0, 1.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=10_0.0_1.0.h5"),
            (1.0, 2.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=10_1.0_2.0.h5"),
            (0.0, 22.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=10.h5"),
        ],
        20 => [
            (1.63, 1.67, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=20_1.63_1.67.h5"),
            (0.8, 1.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=20_0.8_1.0.h5"),
            (0.0, 1.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=20_0.0_1.0.h5"),
            (0.0, 22.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=20.h5"),
        ],
        30 => [
            (1.63, 1.67, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=30_1.63_1.67.h5"),
            (0.84, 0.88, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=30_0.84_0.88.h5"),
            (0.0, 1.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=30_0.0_1.0.h5"),
            (1.0, 2.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=30_1.0_2.0.h5"),
            (0.0, 22.0, "../graphene-plasmons-backup/data/bilayer/loss_3252_θ=30.h5"),
        ],
    )

    i = findfirst(t -> t[1] <= ω && ω <= t[2], table[θ])
    filename = table[θ][i][3]
    # lattice = armchair_bilayer_hexagon(10; rotate = θ)
    # U = bilayer_graphene_coulomb_model(lattice)
    eigenvector, density = _extract_eigenmode(ω, filename)
    p₁ = plot_eigenvector_bilayer(
        lattice,
        eigenvector,
        ω = ω,
        titlefontsize = 24,
        colorbar = false,
        type = type,
        left_margin = 0mm,
        right_margin = 0mm,
        markersize = 4.3,
        markerstrokewidth = 0,
        # color = cgrad(:bwr),
    )
    if isnothing(U)
        return plot(p₁, size = (720, 360))
    end
    p₂ = plot_eigenvector_bilayer(
        lattice,
        U * real(eigenvector),
        title = raw"$\langle\phi_1(\omega)|U$",
        titlefontsize = 16,
        colorbar = false,
        left_margin = 0mm,
        right_margin = 0mm,
        markersize = 2.2,
        markerstrokewidth = 0,
        # color = cgrad(:bwr),
    )
    p₃ = plot_eigenvector_bilayer(
        lattice,
        real(density),
        title = raw"$\Pi(\omega)|\phi_1(\omega)\rangle$",
        titlefontsize = 16,
        colorbar = false,
        left_margin = 0mm,
        right_margin = 0mm,
        markersize = 2.1,
        markerstrokewidth = 0,
        # color = cgrad(:bwr),
    )
    none = plot(ticks=false, border=false, showaxis=false)
    plot(p₁, none, p₃,
        layout = (@layout [°{0.66h}; [° °]]),
        size = (720, 360 + 120),
        # dpi = 150,
    )
end
function plot_twisted_eigenmodes(θ::Real; k::Integer = 10,
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
    ),
)
    ωs = Dict(
        0 => (0.63, 1.195, 0.9775, 1.65),
        10 => (0.405, 1.205, 0.795, 1.663),
        20 => (0.42, 1.2, 0.815, 1.667),
        30 => (0.425, 1.1925, 0.8566, 1.695),
    )
    lattice = armchair_bilayer_hexagon(k; rotate = θ)
    U = nothing # θ == 0 || θ == 30 ? bilayer_graphene_coulomb_model(lattice) : nothing

    none = plot(ticks=false, border=false, showaxis=false)
    header = plot(xlims = (0, 2), ylims = (0, 1), ticks = false, showaxis = false)
    annotate!(
        header,
        [
            (x, 1.0, Plots.text(t, 24, :top, "computer modern"))
            for (x, t) in [(0.147, raw"bottom\nlayer"), (0.394, raw"top\nlayer")]
        ],
    )
    plots = [
        none,
        plot_one_eigenmode(ωs[θ][1], θ, lattice, type = raw"$" * string(θ) * raw"\degree$", U = U),
        plot_one_eigenmode(ωs[θ][2], θ, lattice, U = U),
        none,
        plot_one_eigenmode(ωs[θ][3], θ, lattice, U = U),
        plot_one_eigenmode(ωs[θ][4], θ, lattice, U = U),
    ]
    if θ == 0
        plots = (header, plots...)
    end
    plot_layout = θ == 0 ? (@layout [°{0.3h}; [°{0.01w} ° ° °{0.02w} ° °]]) : (@layout [°{0.01w} ° ° °{0.02w} ° °])
    plot_size = (4 * 720, 1 * 360)
    if θ == 0
        plot_size = (4 * 720, 1 * 400 * 1.3)
    #     plot_size = (4 * 720, 1 * 600 * 1.2)
    # elseif θ == 30
    #     plot_size = (4 * 720, 1 * 600)
    end
    g = plot(
        plots...,
        layout = plot_layout,
        size = plot_size,
        dpi = 150,
        # dpi = 80
    )
    if !isnothing(output)
        savefig(g, joinpath(output, "twisted_eigenmodes_θ=$(θ).png"))
    end
    g
end

function plot_pretty_eels(; σ::Real = 2,
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "eels_zoom.pdf"
    ),
)
    prefix = joinpath(paper_folder, "analysis.doping_1.34")
    p = plot(
        xlabel = raw"$\omega\,,\;\mathrm{eV}$",
        ylabel = raw"$-\mathrm{Im}[1/\varepsilon_1(\omega)]$",
        palette = :Set2_8,
        yticks = [0, 20, 40, 60],
        # xlims = (0, 2),
        legend = :topleft,
        left_margin = 2mm,
        size = (480, 200),
        dpi = 150,
        fontfamily = "computer modern",
    )
    for (i, θ) in enumerate([0, 10, 20, 30])
        f = joinpath(prefix, "combined_loss_k=10_μ=1.34_θ=$(θ).h5")
        frequencies, eigenvalues =
            h5open(io -> (read(io, "frequencies"), read(io, "eigenvalues")), f, "r")
        if ndims(eigenvalues) > 1
            eigenvalues = permutedims(eigenvalues)
            eigenvalues = eigenvalues[:, 1]
        end

        indices = sortperm(frequencies)
        indices = unique(i->frequencies[i], indices)
        frequencies = frequencies[indices]
        eigenvalues = eigenvalues[indices]

        mask = @. (frequencies >= 0.0) & (frequencies <= 2.0)
        frequencies = frequencies[mask]
        eigenvalues = eigenvalues[mask]

        loss = @. -imag(1 / eigenvalues)
        # scale = maximum(loss) / maximum(abs.(real.(eigenvalues)))
        if !isnothing(σ)
            loss = hcat([smoothen(loss[:, i]; σ = σ) for i in 1:size(loss, 2)]...)
        end
        plot!(
            p,
            frequencies,
            loss,
            width = 2,
            color = i,
            label = raw"$\theta = " * string(θ) * raw"\degree$",
        )
    end
    if !isnothing(output)
        savefig(p, output)
    end
    p
end

function plot_non_twisted_eigenmodes_appendix(;
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "non_twisted_eigenmodes_small_doping.png",
    ),
)
    k = 10 # parse(Int, match(r"k=([^._]+)", filename).captures[1])
    θ = 0 # parse(Int, match(r"θ=([^._]+)", filename).captures[1])
    lattice = armchair_bilayer_hexagon(k; rotate = θ)

    function picture(ω; type = nothing, kwargs...)
        filename = joinpath(paper_folder, "analysis", "combined_loss_k=10_θ=0.h5")
        p = plot_eigenvector_bilayer(
            lattice,
            _extract_eigenmode(ω, filename)[1],
            ω = ω,
            titlefontsize = 24,
            colorbar = false,
            type = type,
            left_margin = 0mm,
            right_margin = 0mm,
            markersize = 4.4,
            markerstrokewidth = 0;
            # color = cgrad(:bwr),
            kwargs...
        )
        p
    end

    none = plot(ticks=false, border=false, showaxis=false)

    header = plot(xlims = (0, 1), ylims = (0, 1), ticks = false, showaxis = false)
    annotate!(
        header,
        [
            (x, 1.0, Plots.text(t, 24, :top, "computer modern"))
            for (x, t) in [(0.1405, raw"bottom\nlayer"), (0.3765, raw"top\nlayer")]
        ],
    )
    g = plot(
        header,
        # Separator
        none,
        # First column
        picture(0.303, type = "layer-polarized"),
        picture(0.5678, type = "dipole"),
        picture(0.5946),
        picture(0.7349, type = "1s"),
        # Separator
        none,
        # Second column
        none,
        picture(0.215, alpha = 0.5),
        none,
        picture(0.2493, alpha = 0.5),
        layout = (@layout [header{0.1h}; [_{0.02w} [°; °; °; °] _{0.04w} [°; °; °; °]]]),
        size = (2 * 720 * 1.06, 4 * 360 * 1.1),
        # dpi = 80
    )
    if !isnothing(output)
        savefig(g, output)
    end
    g
end

function plot_excitation_energies(;
    output::Union{AbstractString, Nothing} = joinpath(
        paper_folder,
        "plots",
        "excitation_energies.pdf",
    ),
)
    θs = [0, 10, 20, 30]
    table = [
        1.65 1.663 1.667 1.695
        1.195 1.205 1.2 1.1925
        0.9775 0.795 0.815 0.8566
        0.63 0.405 0.42 0.425
    ]
    table = permutedims(table)
    g = plot(
        θs,
        table,
        labels = [raw"bright 1s" raw"bright dipole" raw"dark 1s" raw"dark dipole"],
        markersize = 6,
        markerstrokewidth = 2,
        linestyle = [:solid :solid :dash :dash],
        lw = 2,
        color = [1 2 1 2],
        markershape = [:circle :diamond :circle :diamond],
        xlabel = raw"$\theta\,,\;\degree$",
        ylabel = raw"$\omega\,,\;\mathrm{eV}$",
        palette = :Set2_8,
        yticks = [0.5, 1.0, 1.5],
        ylims = (0.3, 1.8),
        legend = :outerright,
        size = (480, 200),
        dpi = 150,
        fontfamily = "computer modern",
        bottom_margin = 2mm,
    )
    if !isnothing(output)
        savefig(g, output)
    end
    g
end

function classical_model()
    h = 3.35
    d = 40.0

    position = Dict(
        1 => (θ -> (d * cos(0), d * sin(0), 0)),
        2 => (θ -> (d * cos(π + 0), d * sin(π + 0), 0)),
        3 => (θ -> (d * cos(θ), d * sin(θ), h)),
        4 => (θ -> (d * cos(π + θ), d * sin(π + θ), h))
    )
    sign = Dict(
        1 => 1,
        2 => -1,
        3 => -1,
        4 => 1,
    )

    function energy(θ)
        e = 0.0
        for i in 1:4
            for j in 1:4
                if i != j
                    t = sign[i] * sign[j] / norm(position[i](θ) .- position[j](θ))
                    e += t
                    # @info "" norm(position[i](θ) .- position[j](θ)) t
                end
            end
        end
        e
    end
    # @info "E = $(energy(deg2rad(0)))"
    # @info "E = $(energy(deg2rad(30)))"
    # @info "E = $(energy(deg2rad(60)))"
    plot(
        0:1:180,
        energy.(deg2rad.(0:1:180)),
        xlims=(0, 180),
    )
end
