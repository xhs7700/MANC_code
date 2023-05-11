using DataStructures
using ProgressBars
using SparseArrays
using LinearAlgebra
using Statistics
using Base
using StatsBase
using LogExpFunctions

my_display(x) = display(x), println()

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

function NonBiasedMC(g::Graph, U::Matrix; R::Int, p_term::Float64)
    N, _ = mysize(g)
    ans = zeros(Float64, (N, N))
    for src in ProgressBar(1:N)
        for _ in 1:R
            u = src
            load = 1
            ans[u, u] += 1
            len = 1
            while rand(Float64) > p_term
                len += 1
                v, p = rand(g.adjs[u]), length(g.adjs[u])
                load *= U[u, v] * p / (1 - p_term)
                ans[src, v] += load
                u = v
            end
        end
    end
    return ans / R
end

function LoopErasedMC(g::Graph, d::Vector; R::Int, alpha::Float64)
    N, _ = mysize(g)
    ans = zeros(Float64, N)
    for _ in ProgressBar(1:R)
        next = fill(-1, N)
        root = zeros(Int, N)
        d_sum = DefaultDict{Int,Int}(0)
        for src in 1:N
            u = src
            while root[u] == 0
                if rand(Float64) < alpha
                    root[u] = u
                    next[u] = -1
                else
                    next[u] = rand(g.adjs[u])
                    u = next[u]
                end
            end
            r = root[u]
            u = src
            while root[u] == 0
                root[u] = r
                u = next[u]
            end
        end
        for u in 1:N
            d_sum[root[u]] += 1
        end
        for u in 1:N
            ans[u] += 1 / d_sum[root[u]]
        end
    end
    return ans / (alpha * R)
end

function mysum(P::Matrix, K::Int)
    N = size(P, 1)
    ans = I(N)
    tmp = copy(P)
    for k in ProgressBar(1:K)
        ans += (k + 1) * tmp
        tmp *= P
    end
    return ans
end

function getratios(sets1::Vector, sets2::Vector)
    @assert length(sets1) == length(sets2)
    N = length(sets1)
    ratios = []
    for i in 1:N
        append!(ratios, length(intersect(sets1[i], sets2[i])) / N)
    end
    return ratios
end

descrip(v) = (min=reduce(min, v; init=Inf), max=reduce(max, v; init=0), mean=mean(v), median=median(v))

const g = Graph("data/Euroroads.txt")
const alpha = 0.05
d, sp_A = DiagAdj(g; aug_param=(true, alpha))
N = size(sp_A, 1)

invd = inv.(d)
sqrtinvd = sqrt.(invd)
# cbrtinvd = invd .* sqrtinvd

# P = diagm(inv.(d)) * Matrix(sp_A)
invL = inv(diagm(d) - Matrix(sp_A))
sqinvL = invL * invL

sqrtinvD = diagm(sqrtinvd)
P = sqrtinvD * sp_A * sqrtinvD
invP = inv(I - P)
sqinvP = invP * invP

B1 = NonBiasedMC(g, P; R=2000, p_term=0.05)

test_mat = B1

# test_mat = LoopErasedMC(g, d; R=10000, alpha=alpha)

ratios = []
open("tmp.csv", "w") do io
    write(io, "std, test, ratio\n")
    for i in ProgressBar(1:N)
        std = sqinvL[i, i]
        test = test_mat[i]
        ratio = abs(std - test) / std
        append!(ratios, ratio)
        write(io, "$std, $test, $ratio\n")
    end
end
@show descrip(ratios)

