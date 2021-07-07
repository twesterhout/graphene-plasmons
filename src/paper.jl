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
    output::Union{AbstractString, Nothing} = joinpath(@__DIR__, "..", "paper", "plots"),
    σ::Real = 0.12,
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
        ylabel = raw"DoS",
        fontfamily = "computer modern",
        size = (640, 480),
        dpi = 150,
    )
    for (θ, H) in table
        Es, dos = density_of_states(H, σ = σ)
        plot!(g, Es, dos.(Es); lw = 1, label = raw"$\theta = " * string(θ) * raw"\degree$")
    end
    if !isnothing(output)
        if !isdir(output)
            mkpath(output)
        end
        savefig(g, joinpath(output, "bilayer_graphene_density_of_states.pdf"))
    end
    g
end
