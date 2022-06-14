using GaussMarkovToolbox
using Documenter

DocMeta.setdocmeta!(
    GaussMarkovToolbox,
    :DocTestSetup,
    :(using GaussMarkovToolbox);
    recursive=true,
)

makedocs(;
    modules=[GaussMarkovToolbox],
    authors="Nathanael Bosch <nathanael.bosch@uni-tuebingen.de> and contributors",
    repo="https://github.com/nathanaelbosch/GaussMarkovToolbox.jl/blob/{commit}{path}#{line}",
    sitename="GaussMarkovToolbox.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://nathanaelbosch.github.io/GaussMarkovToolbox.jl",
        assets=String[],
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/nathanaelbosch/GaussMarkovToolbox.jl", devbranch="main")
