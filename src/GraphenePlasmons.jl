module GraphenePlasmons

const paper_folder = joinpath(@__DIR__, "..", "paper")
const bilayer_cRPA_folder =
    joinpath(paper_folder, "input", "03_BL_AB", "H_25_K_18_B_128_d_3.35", "03_cRPA")


include("tight_binding.jl")
include("coulomb_models.jl")
include("plasmons_analysis.jl")

include("paper.jl")

end
