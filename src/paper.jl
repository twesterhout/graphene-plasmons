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

function compute_screened_coulomb_interaction(filename::AbstractString; output::AbstractString)
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
