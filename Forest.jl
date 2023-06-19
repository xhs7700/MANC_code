using DataStructures
using ProgressBars
using SparseArrays
using LinearAlgebra
using Statistics
using Random
using Base
using StatsBase
using TOML

my_display(x) = display(x), println()

descrip(v::VecOrMat) = (min=reduce(min, v; init=Inf), max=reduce(max, v; init=0), mean=mean(v), median=median(v))

getratio(std::Float64, test::Float64) = abs(std - test) / std

getratios(std::Vector{Float64}, test::Vector{Float64}) = getratio.(std, test)

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

function domindiag(L::AbstractMatrix)
    N = size(L, 1)
    return L * ones(eltype(L), N)
end

kerneldiag(L::AbstractMatrix) = diag(inv(L) * diagm(domindiag(L)))
kerneldiag(L::AbstractMatrix, q::AbstractVector) = diag(inv(L + diagm(q)) * diagm(q))

function subindices(N::Int, S::Vector{Int})
    mask = trues(N)
    for x in S
        mask[x] = false
    end
    return mask
end

function subindices(N::Int, s::Int)
    mask = trues(N)
    mask[s] = false
    return mask
end

function subvec(v::AbstractVector, S::Union{Vector{Int},Int})
    N = length(v)
    mask = subindices(N, S)
    return v[mask]
end

function submat(L::AbstractMatrix, S::Union{Vector{Int},Int})
    N = size(L, 1)
    mask = subindices(N, S)
    return L[mask, mask]
end

function subrowmat(L::AbstractMatrix, S::Union{Vector{Int},Int})
    N = size(L, 1)
    mask = subindices(N, S)
    return L[mask, :]
end

function subcolmat(L::AbstractMatrix, S::Union{Vector{Int},Int})
    N = size(L, 2)
    mask = subindices(N, S)
    return L[:, mask]
end

struct Graph
    nodes::Set{Int}
    edges::Set{Tuple{Int,Int,Float64}}
    adjs::DefaultDict{Int,Vector{Int}}
    weights::DefaultDict{Int,Vector{Float64}}
    normalized_weights::Dict{Int,ProbabilityWeights}
    function Graph(file_path::AbstractString; weight::Bool)
        println("Reading graph from $file_path...")
        nodes = Set{Int}()
        edges = Set{Tuple{Int,Int,Float64}}()
        adjs = DefaultDict{Int,Vector{Int}}(Vector{Int})
        weights = DefaultDict{Int,Vector{Float64}}(Vector{Float64})
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
                w = 1
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

function LoopErasedMC(g::Graph, d::Vector{Float64}, q::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}; R::Int)
    N, _ = mysize(g)
    weights = zeros(Float64, R, N)
    ans = zeros(Float64, R, N)
    fa = zeros(Int, R, N)
    sons = Matrix{Vector{Int}}(undef, R, N)
    inv_q = inv.(q)
    for i in (1:R*N)
        sons[i] = Int[]
    end
    # deg = zeros(Float64, R, N)
    for r in (1:R)
        next = zeros(Int, N)
        inforest = falses(N)
        for src in 1:N
            u = src
            while inforest[u] == false
                if rand(Float64) * (q[u] + d[u]) < q[u]
                    inforest[u] = true
                    next[u] = 0
                    ans[r, u] += inv_q[u]
                    weights[r, u] = q[u]
                else
                    next[u] = sample(g.adjs[u], g.normalized_weights[u])
                    u = next[u]
                end
            end
            u = src
            while inforest[u] == false
                inforest[u] = true
                weights[r, u] = A[u, next[u]]
                fa[r, u] = next[u]
                # deg[r, next[u]] += inv_q[u]
                push!(sons[r, next[u]], u)
                u = next[u]
            end
        end
    end
    return weights, ans, sons, fa
end

function AnotherLE(g::Graph, d::Vector{Float64}, q::Vector{Float64}; R::Int)
    N, _ = mysize(g)
    ans = zeros(Float64, N)
    for _ in (1:R)
        next = zeros(Int, N)
        root = zeros(Int, N)
        q_sum = DefaultDict{Int,Float64}(0)
        for src in 1:N
            u = src
            while root[u] == 0
                if rand(Float64) * (q[u] + d[u]) < q[u]
                    root[u] = u
                    next[u] = 0
                else
                    next[u] = sample(g.adjs[u], g.normalized_weights[u])
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
            q_sum[root[u]] += q[u]
        end
        for u in 1:N
            ans[u] += 1 / q_sum[root[u]]
        end
    end
    return ans / R
end

function VarianceReductionLE(g::Graph, d::Vector{Float64}, q::Vector{Float64}; R::Int, alpha::Float64)
    avg_q, avg_d = mean(q), mean(d)
    inv_q = inv.(q)
    # alpha = avg_q / (avg_q + avg_d)
    N, _ = mysize(g)
    ans = zeros(Float64, N)
    for _ in (1:R)
        next = zeros(Int, N)
        root = zeros(Int, N)
        roots = Int[]
        for src in 1:N
            u = src
            while root[u] == 0
                if rand(Float64) * (q[u] + d[u]) < q[u]
                    root[u] = u
                    push!(roots, u)
                    next[u] = 0
                else
                    next[u] = sample(g.adjs[u], g.normalized_weights[u])
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
        for u in roots
            tmp = 0
            len = length(g.adjs[u])
            # tmp = sum(g.weights[u][i] for i in 1:len if root[g.adjs[u][i]] != u)
            for i in 1:len
                v, w = g.adjs[u][i], g.weights[u][i]
                if root[v] != u
                    tmp += w
                end
            end
            ans[u] += inv_q[u] - alpha * (1 + inv_q[u] * tmp)
        end
    end
    return fill(alpha, N) + (ans / R)
end

function getdelta(ans::Matrix{Float64}, A::SparseMatrixCSC{Float64,Int64}, q::Vector{Float64}, ratios::Vector{Float64}, fa::Matrix{Int}, removed::BitVector)
    R, N = size(ans)
    delta = -ans
    # delta = zeros(Float64, R, N)
    for r in 1:R
        for u in 1:N
            s = fa[r, u]
            if s == 0 || removed[s] == true
                continue
            end
            delta[r, s] += ratios[r] / (q[u] + A[u, s])
        end
    end
    rows, vals = rowvals(A), nonzeros(A)
    for j in 1:N
        for i in nzrange(A, j)
            row, val = rows[i], vals[i]
            if row >= j
                break
            end
            if removed[row] == true || removed[j] == true
                continue
            end
            for r in 1:R
                delta[r, row] -= (ans[r, j] * val) / (q[j] + val)
                delta[r, j] -= (ans[r, row] * val) / (q[row] + val)
            end
        end
    end
    return delta
end

function std_greedy_set(g::Graph, alpha::Float64, K::Int)
    d, sp_A = DiagAdj(g)
    N = size(sp_A, 1)
    q = (alpha / (1 - alpha)) * d
    L = diagm(q + d) - sp_A
    S = Int[]
    t = RBTree{Int}()
    for i in 1:N
        push!(t, i)
    end
    for k in 1:K
        println("k = $k")
        inv_lap = submat(L, S) |> inv
        sqinv_lap = inv_lap * inv_lap
        margin = diag(sqinv_lap) ./ diag(inv_lap)
        arg = argmax(margin)
        push!(S, t[arg])
        delete!(t, t[arg])
    end
    return S
end

function std_diag(g::Graph, alpha::Float64, S::Vector{Int})
    d, sp_A = DiagAdj(g)
    q = (alpha / (1 - alpha)) * d
    return submat(diagm(q + d) - sp_A, S) |> inv |> diag
end

function approx_diag(g::Graph, alpha::Float64, S::Vector{Int}, R::Int)
    d, sp_A = DiagAdj(g)
    N = size(sp_A, 1)
    q = (alpha / (1 - alpha)) * d
    weights, ans, sons, fa = LoopErasedMC(g, d, q, sp_A; R=R)
    ratios = ones(Float64, R)
    removed = falses(N)
    for s in S
        println("s = $s")
        removed[s] = true
        s_adj = zip(g.adjs[s], g.weights[s])
        for (v, w) in s_adj
            q[v] += w
        end
        for r in ProgressBar(1:R)
            ratio = 1 / weights[r, s]
            ans[r, :] *= ratio
            ratios[r] *= ratio
            for (v, w) in s_adj
                ans[r, v] *= (q[v] - w) / q[v]
            end
            for v in sons[r, s]
                ans[r, v] += ratios[r] / q[v]
            end
            ans[r, s] = 0
        end
    end
    return subvec(vec(sum(ans; dims=1)) / sum(ratios), S)
end

function std_margin(g::Graph, alpha::Float64, S::Vector{Int})
    d, sp_A = DiagAdj(g)
    q = (alpha / (1 - alpha)) * d
    inv_lap = submat(diagm(q + d) - Matrix(sp_A), S) |> inv
    sqinv_lap = inv_lap * inv_lap
    margin = diag(sqinv_lap) ./ diag(inv_lap)
    return margin
end

function approx_margin(g::Graph, alpha::Float64, S::Vector{Int}, R::Int)
    d, sp_A = DiagAdj(g)
    N = size(sp_A, 1)
    q = (alpha / (1 - alpha)) * d
    weights, ans, sons, fa = LoopErasedMC(g, d, q, sp_A; R=R)
    ratios = ones(Float64, R)
    removed = falses(N)
    for s in S
        println("s = $s")
        removed[s] = true
        s_adj = zip(g.adjs[s], g.weights[s])
        for (v, w) in s_adj
            q[v] += w
        end
        for r in ProgressBar(1:R)
            ratio = 1 / weights[r, s]
            ans[r, :] *= ratio
            ratios[r] *= ratio
            for (v, w) in s_adj
                ans[r, v] *= (q[v] - w) / q[v]
            end
            for v in sons[r, s]
                ans[r, v] += ratios[r] / q[v]
            end
            ans[r, s] = 0
        end
        # writeratio("ans_sum.csv", ans_sum, vec(sum(ans; dims=2)))
    end
    ans_sum = vec(sum(ans; dims=2))
    delta = getdelta(ans, sp_A, q, ratios, fa, removed)
    margin = zeros(Float64, N)
    for u in ProgressBar(1:N)
        margin[u] = sum(ans_sum) / sum(ratios) - sum((ans_sum + delta[:, u]) ./ weights[:, u]) / sum(ratios ./ weights[:, u])
    end
    return margin
end

# Random.seed!(19260817)
const g = Graph("data/Euroroads.txt"; weight=false)
const alpha = 0.05
const R = 1000
const N = mysize(g)[1]
const S = Int[]

std = std_diag(g, alpha, S)
test = approx_diag(g, alpha, S, R)
writeratio("diag.csv", std, test)