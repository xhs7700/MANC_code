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
end

struct WeightedGraph <: Graph
    nodes::Set{Int}
    edges::Set{NTuple{3,Int}}
end

size(g::Graph) = (n=length(g.nodes), m=length(g.edges))

UnweightedGraph() = UnweightedGraph(Set{Int}(), Set{NTuple{2,Int}}())
WeightedGraph() = WeightedGraph(Set{Int}(), Set{NTuple{3,Int}}())

SimilarGraph(::UnweightedGraph) = UnweightedGraph()
SimilarGraph(::WeightedGraph) = WeightedGraph()

Prefix(::UnweightedGraph) = "UnweightedGraph"
Prefix(::WeightedGraph) = "WeightedGraph"

function AddEdge(g::Graph, e::NTuple)
    u, v = e
    push!(g.nodes, u), push!(g.nodes, v)
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
    println("Finding LCC of graph...")
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
    remove_nodes = filter(x -> find(x) != root, g.nodes)
    remove_edges = Set{NTuple}()
    if length(remove_nodes) > 0
        remove_edges = filter(e -> e[1] in remove_nodes || e[2] in remove_nodes, g.edges)
    end
    for u in ProgressBar(remove_nodes)
        delete!(g.nodes, u)
    end
    for e in ProgressBar(remove_edges)
        delete!(g.edges, e)
    end
    return g
end

function Renumber(g::Graph)
    println("Renumbering nodes...")
    o2n = DefaultDict{Int,Int}(() -> length(o2n) + 1)
    new_g = SimilarGraph(g)
    edges = sort(collect(g.edges))
    for e in ProgressBar(edges)
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

function WriteModelGraph(edges::Vector{Tuple{Int,Int}}, N::Int, name::AbstractString, output_path::AbstractString)
    M = length(edges)
    open(output_path, "w") do io
        write(io, "# UnweightedGraph graph: $name\n# Nodes: $N Edges: $M\n")
        for e in ProgressBar(sort(edges))
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
    @assert source in ["KONECT", "SNAP", "SELF", "MODEL"]
    println("Preparing $file_path...")
    if !isfile(file_path)
        if source == "SELF"
            error("Self-writing dataset not found.")
        elseif source == "MODEL"
            graph_name = d["name"]
            graph_args = d["args"]
            graph_func = Dict(
                "Pseudofractal" => Pseudofractal,
                "Koch" => Koch,
                "CayleyTree" => CayleyTree,
                "HanoiExt" => HanoiExt,
                "SmallHanoiExt" => HanoiExt
            )
            @assert haskey(graph_func, graph_name)
            edges, N = graph_func[graph_name](graph_args...)
            println("Writing new graph...")
            WriteModelGraph(edges, N, graph_name, file_path)
        else
            Prepare = source == "KONECT" ? PrepareKONECTFile : PrepareSNAPFile
            raw_path, del_path = Prepare(d)
            g = ReadGraph(raw_path, is_weighted)
            @show size(g)
            new_g = Renumber(FindLCC(g))
            println("Writing new graph...")
            WriteGraph(new_g, d["name"], file_path)
            rm(del_path, recursive=true)
        end
    end
end

function Pseudofractal(g::Int)
    ans = Tuple{Int,Int}[(1, 2), (2, 3), (3, 1)]
    N = 3
    for iter in 1:g
        M = length(ans)
        for i in 1:M
            u, v = ans[i]
            N += 1
            push!(ans, (u, N), (N, v))
        end
    end
    return ans, N
end

function Koch(g::Int)
    triangles = Tuple{Int,Int,Int}[(1, 2, 3)]
    N = 3
    for iter in 1:g
        M = length(triangles)
        for i in 1:M
            for u in triangles[i]
                push!(triangles, (u, N + 1, N + 2))
                N += 2
            end
        end
    end
    ans = Tuple{Int,Int}[]
    for (x, y, z) in triangles
        push!(ans, (x, y), (y, z), (z, x))
    end
    return ans, N
end

function CayleyTree(b::Int, g::Int)
    ans = Tuple{Int,Int}[]
    leafs = Int[]
    for leaf in 2:b+1
        push!(ans, (1, leaf))
        push!(leafs, leaf)
    end
    N = b + 1
    for iter in 2:g
        new_leafs = Int[]
        for leaf in leafs
            for new_leaf in N+1:N+b-1
                push!(ans, (leaf, new_leaf))
                push!(new_leafs, new_leaf)
            end
            N += b - 1
        end
        leafs = new_leafs
    end
    return ans, N
end

function HanoiExt(g::Int)
    ans = Tuple{Int,Int}[(1, 2), (2, 3), (3, 1)]
    inc = 1
    for iter in 2:g-1
        inc *= 3
        M = length(ans)
        for i in 1:M
            u, v = ans[i]
            push!(ans, (u + inc, v + inc), (u + 2 * inc, v + 2 * inc))
        end
        for x in 0:2
            y = (x + 1) % 3
            u = parse(Int, string(x) * repeat(string(y), iter - 1), base=3) + 1
            v = parse(Int, string(y) * repeat(string(x), iter - 1), base=3) + 1
            push!(ans, (u, v))
        end
    end
    inc *= 3
    M = length(ans)
    for i in 1:M
        u, v = ans[i]
        push!(ans, (u + inc, v + inc), (u + 2 * inc, v + 2 * inc), (u + 3 * inc, v + 3 * inc))
    end
    for x in 0:2
        y = (x + 1) % 3
        u = parse(Int, string(x) * repeat(string(y), g - 1), base=3) + 1
        v = parse(Int, string(y) * repeat(string(x), g - 1), base=3) + 1
        push!(ans, (u, v))
        u = parse(Int, repeat(string(x), g), base=3) + 1
        v = parse(Int, "10" * repeat(string(x), g - 1), base=3) + 1
        push!(ans, (u, v))
    end
    N = 4 * 3^(g - 1)
    return ans, N
end

const tot_d = TOML.parsefile("graphs.toml")

foreach(PrepareFile, values(tot_d))
# PrepareFile(tot_d["SmallHanoiExt"])