using DataStructures
using ProgressBars
using SparseArrays
using LinearAlgebra
using Statistics
using Random
using Base
using StatsBase
using Serialization

my_display(x) = display(x), println()

const IntTuple = NTuple{N,Int} where {N}

function subindices(N::Int, S::Vector{Int})
    mask = trues(N)
    for x in S
        mask[x] = false
    end
    return mask
end

function subvec(v::AbstractVector, S::Vector{Int})
    N = length(v)
    mask = subindices(N, S)
    return v[mask]
end

function submat(L::AbstractMatrix, S::Vector{Int})
    N = size(L, 1)
    mask = subindices(N, S)
    return L[mask, mask]
end

struct Graph
    nodes::Set{Int}
    edges::Set{NTuple{3,Int}}
    adjs::DefaultDict{Int,AbstractVector}
    weights::DefaultDict{Int,AbstractVector}
    normalized_weights::Dict{Int,ProbabilityWeights}
    function Graph(file_path::AbstractString; weight::Bool)
        println("Reading graph from $file_path...")
        nodes = Set{Int}()
        edges = Set{NTuple{3,Int}}()
        adjs = DefaultDict{Int,AbstractVector}(Vector{Int})
        weights = DefaultDict{Int,AbstractVector}(Vector{Float64})
        normalized_weights = Dict{Int,ProbabilityWeights}()
        for line in ProgressBar(eachline(file_path))
            if line[begin] in "#%"
                continue
            end
            line = strip(line, ('\t', ' '))
            if weight
                u, v, w = map(x -> parse(Int, x), split(line, ('\t', ' ')))
            else
                u, v = map(x -> parse(Int, x), split(line, ('\t', ' ')))
                w = rand(Float64)
            end
            if u == v
                continue
            end
            push!(nodes, u, v)
            push!(edges, (u, v, w))
            push!(adjs[u], v), push!(adjs[v], u)
            push!(weights[u], w), push!(weights[v], w)
        end
        for u in nodes
            normalized_weights[u] = ProbabilityWeights(normalize(weights[u], 1))
        end
        new(nodes, edges, adjs, weights, normalized_weights)
    end
    function Graph(A::SparseMatrixCSC{Float64,Int64})
        N = size(A, 1)
        nodes = Set{Int}(1:N)
        edges = Set{NTuple{3,Int}}()
        adjs = DefaultDict{Int,AbstractVector}(Vector{Int})
        weights = DefaultDict{Int,AbstractVector}(Vector{Float64})
        normalized_weights = Dict{Int,ProbabilityWeights}()
        rows, vals = rowvals(A), nonzeros(A)
        for j in 1:N
            for i in nzrange(A, j)
                row, val = rows[i], vals[i]
                if row >= j
                    break
                end
                push!(edges, (row, j, val))
                push!(adjs[row], j), push!(adjs[j], row)
                push!(weights[row], val), push!(weights[j], val)
            end
        end
        for u in 1:N
            normalized_weights[u] = ProbabilityWeights(normalize(weights[u], 1))
        end
        new(nodes, edges, adjs, weights, normalized_weights)
    end
end

struct ForestData
    name::IntTuple
    count::Int
    weight::Float64
    freq::Float64
    prob::Float64
    sons::Vector{Vector{Int}}
end

mysize(g::Graph) = (length(g.nodes), length(g.edges))

function DiagAdj(g::Graph)
    N, M = mysize(g)
    A_I, A_J, A_V = Int[], Int[], Float64[]
    sizehint!(A_I, 2 * M), sizehint!(A_J, 2 * M), sizehint!(A_V, 2 * M)
    d = zeros(Float64, N)
    for (u, v, w) in ProgressBar(g.edges)
        push!(A_I, u, v), push!(A_J, v, u), push!(A_V, w, w)
        d[u], d[v] = d[u] + w, d[v] + w
    end
    return d, sparse(A_I, A_J, A_V)
end

function Degree(A::SparseMatrixCSC{Float64,Int64})
    N = size(A, 1)
    return A * ones(Float64, N)
end

function LoopErasedMC(g::Graph, d::Vector{Float64}, q::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}; R::Int, S::Vector{Int}, compact::Bool)
    N, _ = mysize(g)
    counts = DefaultDict{IntTuple,Int}(0)
    weights = Dict{IntTuple,Float64}()
    sons_dict = Dict{IntTuple,Vector{Vector{Int}}}()
    for _ in ProgressBar(1:R)
        next = zeros(Int, N)
        inforest = falses(N)
        sons = Vector{Vector{Int}}(undef, N)
        for u in 1:N
            sons[u] = Int[]
        end
        for src in 1:N
            if src in S
                inforest[src] = true
                next[src] = 0
                continue
            end
            u = src
            while inforest[u] == false
                if rand(Float64) * (q[u] + d[u]) < q[u]
                    inforest[u] = true
                    next[u] = 0
                else
                    v = sample(g.adjs[u], g.normalized_weights[u])
                    if v in S
                        inforest[u] = true
                        next[u] = compact ? 0 : v
                    else
                        next[u] = v
                        u = v
                    end
                end
            end
            u = src
            while inforest[u] == false
                inforest[u] = true
                push!(sons[next[u]], u)
                u = next[u]
            end
        end
        t = Tuple(next)
        counts[t] += 1

        if !haskey(weights, t)
            ans = 1
            for u in 1:N
                if u in S
                    continue
                end
                tmp = next[u] == 0 ? q[u] : A[u, next[u]]
                ans *= tmp
            end
            weights[t] = ans
            sons_dict[t] = sons
        end
    end
    name_vec = IntTuple[]
    count_vec = Int[]
    weight_vec = Float64[]
    sons_vec = Vector{Vector{Int}}[]
    for t in keys(counts)
        push!(name_vec, t)
        push!(count_vec, counts[t])
        push!(weight_vec, weights[t])
        push!(sons_vec, sons_dict[t])
    end
    freq_vec = normalize(count_vec, 1)
    prob_vec = normalize(weight_vec, 1)
    len = length(name_vec)
    forests = ForestData[]
    for i in 1:len
        push!(forests, ForestData(name_vec[i], count_vec[i], weight_vec[i], freq_vec[i], prob_vec[i], sons_vec[i]))
    end
    sort!(forests; by=forest -> forest.name)
    return forests
end

function AnotherLE(d::Vector{Float64}, q::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}; R::Int, S::Vector{Int})
    L = submat(diagm(d) - Matrix(A), S)
    N = size(L, 1)
    mod_A = submat(A, S)
    g = Graph(mod_A)
    q = subvec(q, S) + L * ones(Float64, N)
    return LoopErasedMC(g, Degree(mod_A), q, mod_A; R=R, S=Int[], compact=false)
end

function WriteForests(file_path::AbstractString, forests::Vector{ForestData})
    open(file_path, "w") do io
        write(io, "name, count, weight, freq, prob\n")
        for forest in forests
            name = join(forest.name)
            count = forest.count
            weight = forest.weight
            freq = forest.freq
            prob = forest.prob
            write(io, "\'$name\', $count, $weight, $freq, $prob\n")
        end
    end
end

function RemoveNodes(forests::Vector{ForestData}, q::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, S::Vector{Int})
    weight_dict = DefaultDict{IntTuple,Float64}(0)
    for forest in forests
        name = forest.name
        fa = collect(name)
        w = 1.0
        for s in S
            w *= name[s] == 0 ? q[s] : A[s, name[s]]
            fa[s] = 0
            for u in forest.sons[s]
                fa[u] = 0
            end
        end
        new_name = Tuple(fa)
        weight_dict[new_name] += forest.weight / w
    end
    name_vec = IntTuple[]
    weight_vec = Float64[]
    for t in keys(weight_dict)
        push!(name_vec, t)
        push!(weight_vec, weight_dict[t])
    end
    prob_vec = normalize(weight_vec, 1)
    new_forests = ForestData[]
    len = length(name_vec)
    for i in 1:len
        push!(new_forests, ForestData(name_vec[i], 0, weight_vec[i], 0, prob_vec[i], Vector{Int}[Int[]]))
    end
    sort!(new_forests; by=x -> x.name)
    return new_forests
end

const g = Graph("data/clique-5.txt"; weight=true)
N, _ = mysize(g)
const q = Float64[i for i in 2:N+1]
d, sp_A = DiagAdj(g)

S = Int[3, 4, 5]

forests_modified = LoopErasedMC(g, d, q, sp_A; R=1000000, S=S, compact=true)
WriteForests("forests_modified_compact.csv", forests_modified)

forests = LoopErasedMC(g, d, q, sp_A; R=1000000, S=Int[], compact=false)

new_forests = RemoveNodes(forests, q, sp_A, S)

WriteForests("forests_test.csv", new_forests)