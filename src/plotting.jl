using Plots
using Plots.PlotMeasures

export plot_example_armchair_samples,
    plot_example_zigzag_samples, plot_example_bilayer_samples

"""
    plot_lattice(sites::Lattice; kwargs...) -> Plot

Visualize a lattice. Different sublattices are shown in different color.
"""
function plot_lattice(sites::Lattice; sublattices::Bool = true, kwargs...)
    edges = nearest_neighbours(sites)

    function limits(axis::Int)
        (m, M) = extrema((i.position[axis] for i in sites))
        return (m - 0.5, M + 0.5)
    end

    p = plot(
        _make_edges_plottable(sites, edges)...,
        linewidth = 2,
        linecolor = RGB(80 / 255, 80 / 255, 80 / 255),
        xlims = limits(1),
        ylims = limits(2),
        alpha = 0.75,
        axis = ([], false),
        label = nothing,
        aspect_ratio = 1,
    )

    (aₘᵢₙ, aₘₐₓ) = extrema((i.sublattice for i in sites))
    colors = sublattices ? [2 3 2 3] : [2 2 3 3]
    alpha = [1.0 1.0 0.7 0.7]
    for a in aₘₐₓ:-1:aₘᵢₙ
        sublattice = filter(i -> i.sublattice == a, sites)
        scatter!(
            p,
            map(i -> i.position[1], sublattice),
            map(i -> i.position[2], sublattice),
            markerstrokewidth = 2,
            color = colors[a],
            alpha = alpha[a],
            label = nothing;
            kwargs...,
        )
    end
    # x = sum(map(i->i.position[1], sites)) / length(sites)
    # y = sum(map(i->i.position[2], sites)) / length(sites)
    index = center_site_index(sites)
    # argmin(map(i->(x - i.position[1])^2 + (y - i.position[2])^2 + i.sublattice, sites))
    # index = length(sites) ÷ 2 - 1
    scatter!(
        p,
        [sites[index].position[1]],
        [sites[index].position[2]],
        markersize = 5,
        markerstrokewidth = 0.1,
        color = :red,
    )
    p
end
function plot_lattice_v2(sites::Lattice; kwargs...)
    layers = []
    push!(layers, filter(i -> iszero(i.position[3]), sites))
    if length(layers[1]) != length(sites)
        push!(layers, filter(i -> !iszero(i.position[3]), sites))
    end
    @assert sum(map(l -> length(l), layers)) == length(sites)
    edges = [nearest_neighbours(l) for l in layers]

    function limits(axis::Int)
        (m, M) = extrema((i.position[axis] for i in sites))
        return (m - 1.0, M + 1.0)
    end

    p = plot(
        xlims = limits(1),
        ylims = limits(2),
        palette = :Set2_8,
        axis = ([], false),
        aspect_ratio = 1,
    )
    for (i, (sites, edges)) in enumerate(zip(layers, edges))
        plot!(
            p,
            _make_edges_plottable(sites, edges)...,
            linewidth = 2,
            linecolor = i + 1,
            label = "",
        )
        scatter!(
            p,
            map(i -> i.position[1], sites),
            map(i -> i.position[2], sites),
            markerstrokewidth = 0,
            color = i + 1,
            markerstrokecolor = i + 1,
            label = "";
            kwargs...,
        )
    end
    return p
end

function plot_example_zigzag_samples()
    plotone(k; kwargs...) = plot_lattice(zigzag_hexagon(k); kwargs...)
    sizes = [1, 2, 3, 4]
    widths = [1 + 2 * (2 * n) - 0.5 * (2 * n - 1) for n in sizes]
    scale = (1 + sqrt(3) * (2 * maximum(sizes))) / sum(widths)
    plot(
        (plotone(n) for n in sizes)...,
        layout = grid(1, 4, widths = widths ./ sum(widths)),
        size = (800, scale * 800),
        dpi = 150,
    )
end
plot_example_zigzag_samples(output::AbstractString) =
    savefig(plot_example_zigzag_samples(), output)

function plot_example_armchair_samples()
    plotone(k; kwargs...) = plot_lattice(armchair_hexagon(k); kwargs...)
    sizes = [1, 2, 3, 4]
    widths = [1 + 2 * (2 * n - 1) + (2 * n - 2) for n in sizes]
    scale = sqrt(3) / 2 * maximum(widths) / sum(widths)
    plot(
        (plotone(n) for n in sizes)...,
        layout = grid(1, 4, widths = widths ./ sum(widths)),
        size = (800, scale * 800),
        dpi = 150,
    )
end
plot_example_armchair_samples(output::AbstractString) =
    savefig(plot_example_armchair_samples(), output)

function plot_example_bilayer_samples()
    plotone(k, θ; kwargs...) = plot_lattice(
        armchair_bilayer_hexagon(k, rotate = θ);
        sublattices = false,
        markersize = 3.0,
        kwargs...,
    )
    heights = [1, 4]
    plot(
        plotone(1, 0, title = raw"$\theta=0\degree$"),
        plotone(1, 10, title = raw"$\theta=10\degree$"),
        plotone(1, 20, title = raw"$\theta=20\degree$"),
        plotone(1, 30, title = raw"$\theta=30\degree$"),
        plotone(2, 0),
        plotone(2, 10),
        plotone(2, 20),
        plotone(2, 30),
        layout = grid(2, 4, heights = heights ./ sum(heights)),
        size = (800, 300),
        dpi = 150,
    )
end
plot_example_bilayer_samples(output::AbstractString) =
    savefig(plot_example_bilayer_samples(), output)

function plot_our_sample()
    plotone(k, θ; kwargs...) = plot_lattice(
        armchair_bilayer_hexagon(k, rotate = θ);
        sublattices = false,
        markersize = 3.0,
        markerstrokewidth = 0.2,
        kwargs...,
    )
    plot(plotone(10, 10, title = raw"$\theta=10\degree$"), size = (800, 700), dpi = 150)
end
plot_our_sample(output::AbstractString) = savefig(plot_our_sample(), output)

function plot_our_sample_v2()
    plotone(k, θ; kwargs...) =
        plot_lattice_v2(armchair_bilayer_hexagon(k, rotate = θ); kwargs...)
    plot(plotone(10, 10), size = (800, 700), dpi = 150)
end

function plot_density_of_states(
    Es::AbstractRange,
    dos::Function;
    output::Union{AbstractString, Nothing} = nothing,
    kwargs...,
)
    p = plot(
        Es,
        dos.(Es),
        xlabel = raw"$E\,,\;\mathrm{eV}$",
        ylabel = raw"DoS",
        fontfamily = "computer modern",
        lw = 2,
        label = nothing,
        size = (640, 480),
        dpi = 150;
        kwargs...,
    )
    if isnothing(output)
        return p
    else
        savefig(p, output)
        return nothing
    end
end


