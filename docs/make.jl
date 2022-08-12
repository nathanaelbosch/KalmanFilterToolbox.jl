using KalmanFilterToolbox
using Documenter

DocMeta.setdocmeta!(
    KalmanFilterToolbox,
    :DocTestSetup,
    :(using KalmanFilterToolbox);
    recursive=true,
)

makedocs(;
    modules=[KalmanFilterToolbox],
    authors="Nathanael Bosch <nathanael.bosch@uni-tuebingen.de> and contributors",
    repo="https://github.com/nathanaelbosch/KalmanFilterToolbox.jl/blob/{commit}{path}#{line}",
    sitename="KalmanFilterToolbox.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nathanaelbosch.github.io/KalmanFilterToolbox.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md"
        "API" => [
            "Filtering and Smoothing" => "filtsmooth.md"
            "Gauss-Markov Processes" => "priors.md"
        ]
        "Examples" => ["Sampling from Priors" => "prior_sampling.md"]
    ],
)

deploydocs(;
    repo="github.com/nathanaelbosch/KalmanFilterToolbox.jl",
    devbranch="main",
    push_preview=true,
)
