using TOML
using ProgressBars
using DataStructures
using Downloads: download as wget
using Base.Filesystem

my_display(x) = display(x), println()

abstract type Graph end

struct UnweightedGraph <: Graph
    nodes::Set{Int}
    edges::Set{NTuple{2,Int}}
    adjs::AbstractDict{Int,Set{Int}}
end

struct WeightedGraph <: Graph
    nodes::Set{Int}
    edges::Set{NTuple{3,Int}}
    adjs::AbstractDict{Int,Set{Int}}
end

size(g::Graph) = (n=length(g.nodes), m=length(g.edges))

UnweightedGraph() = UnweightedGraph(Set{Int}(), Set{NTuple{2,Int}}(), DefaultDict{Int,Set{Int}}(() -> Set{Int}()))
WeightedGraph() = WeightedGraph(Set{Int}(), Set{NTuple{3,Int}}(), DefaultDict{Int,Set{Int}}(() -> Set{Int}()))

SimilarGraph(::UnweightedGraph) = UnweightedGraph()
SimilarGraph(::WeightedGraph) = WeightedGraph()

Prefix(::UnweightedGraph) = "UnweightedGraph"
Prefix(::WeightedGraph) = "WeightedGraph"

function AddEdge(g::Graph, e::NTuple)
    u, v = e
    push!(g.nodes, u), push!(g.nodes, v)
    push!(g.adjs[u], v), push!(g.adjs[v], u)
    push!(g.edges, e)
end

function ReadGraph(input_path::AbstractString, is_weighted::Bool)
    g = is_weighted ? WeightedGraph() : UnweightedGraph()
    for line in ProgressBar(eachline(input_path))
        if line[begin] in "#%"
            continue
        end
        line = strip(line, ('\t', ' '))
        e = tuple(map(x -> parse(Int, x), split(line, ('\t', ' ')))...)
        u, v = e
        if u > v
            u, v = v, u
        end
        AddEdge(g, tuple(u, v, e[3:end]...))
    end
    return g
end

function FindLCC(g::Graph)
    fa = DefaultDict{Int,Int}(-1)
    find(x) = fa[x] < 0 ? x : (fa[x] = find(fa[x]); fa[x])
    for (u, v) in ProgressBar(g.edges)
        u, v = find(u), find(v)
        if u == v
            continue
        end
        if -fa[u] < -fa[v]
            u, v = v, u
        end
        fa[u] += fa[v]
        fa[v] = u
    end
    root = argmin(fa)
    new_g = SimilarGraph(g)
    nodes = filter(x -> find(x) == root, g.nodes)
    for e in g.edges
        u, v = e
        if u in nodes && v in nodes
            AddEdge(new_g, e)
        end
    end
    return new_g
end

function Renumber(g::Graph)
    o2n = DefaultDict{Int,Int}(() -> length(o2n) + 1)
    new_g = SimilarGraph(g)
    for e in sort(collect(g.edges))
        u, v = e
        AddEdge(new_g, (o2n[u], o2n[v], e[3:end]...))
    end
    return new_g
end

function WriteGraph(g::Graph, name::AbstractString, output_path::AbstractString)
    open(output_path, "w") do io
        n, m = size(g)
        write(io, "# $(Prefix(g)) graph: $name\n# Nodes: $n Edges: $m\n")
        for e in ProgressBar(sort(collect(g.edges)))
            e_str = join(e, "\t")
            write(io, "$e_str\n")
        end
    end
end

function PrepareKONECTFile(d::AbstractDict)
    dir_name, internal_name = d["dir_name"], d["internal_name"]
    del_path = "$dir_name$internal_name"
    del_path = joinpath(dir_name, internal_name)
    println("File not found. Try to get it from $del_path...")
    if !isdir(del_path)
        zip_path = joinpath(dir_name, "download.tsv.$internal_name.tar.bz2")
        println("Raw directory not found. Try to get it from $zip_path...")
        if !isfile(zip_path)
            url = d["url"]
            println("Gzip file not found. Try to get it from $url...")
            wget(url, zip_path)
        end
        run(`tar -xjvf $zip_path -C $dir_name`)
        rm(zip_path)
    end
    for raw_name in readdir(del_path)
        if startswith(raw_name, "out.")
            return joinpath(del_path, raw_name), del_path
        end
    end
    return "", del_path
end

function PrepareSNAPFile(d::AbstractDict)
    dir_name, file_name = d["dir_name"], d["file_name"]
    raw_path = joinpath(dir_name, "$file_name-raw.txt")
    println("File not found. Try to get it from $raw_path...")
    if !isfile(raw_path)
        zip_path = "$raw_path.gz"
        println("Raw file not found. Try to get it from $zip_path...")
        if !isfile(zip_path)
            url = d["url"]
            println("Gzip file not found. Try to get it from $url...")
            wget(url, zip_path)
        end
        run(`gzip -dv $zip_path`)
    end
    return raw_path, raw_path
end

function PrepareFile(d::AbstractDict)
    is_weighted, source = d["is_weighted"], d["source"]
    dir_name, file_name = d["dir_name"], d["file_name"]
    file_path = joinpath(dir_name, "$file_name.txt")
    @assert source in ["KONECT", "SNAP", "SELF"]
    println("Preparing $file_path...")
    if !isfile(file_path)
        source == "SELF" && error("Self-writing dataset not found.")
        Prepare = source == "KONECT" ? PrepareKONECTFile : PrepareSNAPFile
        raw_path, del_path = Prepare(d)
        g = ReadGraph(raw_path, is_weighted)
        @show size(g)
        println("Finding LCC of graph...")
        new_g = Renumber(FindLCC(g))
        println("Writing new graph...")
        WriteGraph(new_g, d["name"], file_path)
        rm(del_path, recursive=true)
    end
end
const tot_d = TOML.parsefile("graphs.toml")
foreach(PrepareFile, values(tot_d))
# PrepareFile(tot_d["CA-GrQc"])