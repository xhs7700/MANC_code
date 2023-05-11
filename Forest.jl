using DataStructures
using ProgressBars
using SparseArrays
using LinearAlgebra
using Statistics
using Base

my_display(x) = display(x), println()

descrip(v::VecOrMat) = (min=reduce(min, v; init=Inf), max=reduce(max, v; init=0), mean=mean(v), median=median(v))

getratios(std::Vector{Float64}, test::Vector{Float64}) = abs.(std - test) ./ std

function writeratio(path::AbstractString, std::Vector{Float64}, test::Vector{Float64})
    ratios = getratios(std, test)
    N = length(ratios)
    open(path, "w") do io
        write(io, "i, std, test, ratio\n")
        for i in 1:N
            write(io, "$i, $(std[i]), $(test[i]), $(ratios[i])\n")
        end
    end
    @show descrip(ratios)
end

struct Graph
    nodes::Set{Int}
    edges::Set{Tuple{Int,Int}}
    adjs::DefaultDict{Int,AbstractVector}
    function Graph(file_path::AbstractString)
        println("Reading graph from $file_path...")
        nodes = Set{Int}()
        edges = Set{Tuple{Int,Int}}()
        adjs = DefaultDict{Int,AbstractVector}(Vector{Int})
        for line in ProgressBar(eachline(file_path))
            if line[begin] in "#%"
                continue
            end
            line = strip(line, ('\t', ' '))
            u, v = map(x -> parse(Int, x), split(line, ('\t', ' ')))
            if u == v
                continue
            end
            push!(nodes, u, v)
            push!(edges, (u, v))
            push!(adjs[u], v), push!(adjs[v], u)
        end
        new(nodes, edges, adjs)
    end
end

mysize(g::Graph) = (length(g.nodes), length(g.edges))

function DiagAdj(g::Graph; aug_param::Tuple{Bool,Float64})
    N, M = mysize(g)
    A_I, A_J, A_V = Int[], Int[], Float64[]
    sizehint!(A_I, 2 * M), sizehint!(A_J, 2 * M), sizehint!(A_V, 2 * M)
    aug, alpha = aug_param
    d = zeros(Float64, N)
    for (u, v) in ProgressBar(g.edges)
        push!(A_I, u, v), push!(A_J, v, u), push!(A_V, 1, 1)
        d[u], d[v] = d[u] + 1, d[v] + 1
    end
    return aug ? d / (1 - alpha) : d, sparse(A_I, A_J, A_V)
end

function kerneldiag(L::AbstractMatrix)
    N = size(L, 1)
    q = L * ones(eltype(L), N)
    return diag(inv(L) * diagm(q))
end

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

function NaiveRSF(g::Graph, S::Vector{Int}; R::Int, alpha::Float64)
    N, _ = mysize(g)
    ans = zeros(Int, N)
    for _ in ProgressBar(1:R)
        next = fill(-1, N)
        inforest = falses(N)
        for src in 1:N
            u = src
            while inforest[u] == false
                if rand(Float64) < alpha
                    inforest[u] = true
                    ans[u] += 1
                    next[u] = -1
                else
                    v = rand(g.adjs[u])
                    if v in S
                        inforest[u] = true
                        ans[u] += 1
                        next[u] = -1
                    else
                        next[u] = v
                        u = v
                    end
                end
            end
            u = src
            while inforest[u] == false
                inforest[u] = true
                u = next[u]
            end
        end
    end
    return subvec(ans / R, S)
end

function ComplexRSF(g::Graph; R::Int, alpha::Float64)
    N, _ = mysize(g)
    ans = zeros(Int, N)
    chs = Matrix{Vector{Int}}(undef, R, N)
    for i in ProgressBar(1:R*N)
        chs[i] = Int[]
    end
    for r in ProgressBar(1:R)
        next = fill(-1, N)
        inforest = falses(N)
        for src in 1:N
            u = src
            while inforest[u] == false
                if rand(Float64) < alpha
                    inforest[u] = true
                    ans[u] += 1
                    next[u] = -1
                else
                    next[u] = rand(g.adjs[u])
                    u = next[u]
                end
            end
            u = src
            while inforest[u] == false
                inforest[u] = true
                push!(chs[r, next[u]], u)
                u = next[u]
            end
        end
    end
    return ans, chs
end

const g = Graph("data/Jazz_musicians.txt")
const alpha = 0.05
d, sp_A = DiagAdj(g; aug_param=(true, alpha))
N = size(sp_A, 1)

S = sortperm(d; rev=true)[1:5]
@show S

L = diagm(d) - Matrix(sp_A)

R = 100000

# test = NaiveRSF(g, S; R=10000, alpha=alpha)
init_K = kerneldiag(L)
K = kerneldiag(submat(L, S))
writeratio("init2std.csv", subvec(init_K, S), K)

ans, chs = ComplexRSF(g; R=R, alpha=alpha)

writeratio("init2test.csv", init_K, ans / R)

for s in S
    println("s = $s")
    for r in ProgressBar(1:R)
        for u in chs[r, s]
            ans[u] += 1
        end
    end
end

writeratio("std2test.csv", K, subvec(ans / R, S))