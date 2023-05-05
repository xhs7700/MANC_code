using DataStructures
using ProgressBars
using SparseArrays
using LinearAlgebra
using Statistics
using Base

my_display(x) = display(x), println()

struct Graph
    nodes::Set{Int}
    edges::Set{Tuple{Int,Int}}
    adjs::DefaultDict{Int,AbstractSet}
    function Graph(file_path::AbstractString)
        println("Reading graph from $file_path...")
        nodes = Set{Int}()
        edges = Set{Tuple{Int,Int}}()
        adjs = DefaultDict{Int,AbstractSet}(Set{Int})
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

function DiagAdj(g::Graph; aug::Bool)
    N, M = mysize(g)
    A_I, A_J, A_V = Int[], Int[], Float64[]
    sizehint!(A_I, 2 * M), sizehint!(A_J, 2 * M), sizehint!(A_V, 2 * M)
    d = aug ? ones(Float64, N) : zeros(Float64, N)
    for (u, v) in ProgressBar(g.edges)
        push!(A_I, u, v), push!(A_J, v, u), push!(A_V, 1, 1)
        d[u], d[v] = d[u] + 1, d[v] + 1
    end
    return d, sparse(A_I, A_J, A_V)
end

function MonteCarlo(g::Graph, d::Vector, src::Int, R::Int, L::Int)
    cnt = 0
    for _ in (1:R)
        u = src
        for i in 1:L
            if rand(1:d[u]) == 1
                break
            end
            # @show u g.adjs[u]
            u = rand(g.adjs[u])
            if u == src
                cnt += i + 1
            end
        end
    end
    return inv(d[src]) * inv(d[src]) * (1 + cnt / R)
end

function MonteCarloTest(g::Graph, d::Vector, U::Matrix; R::Int, p_term::Float64)
    N, _ = mysize(g)
    ans = zeros(Float64, (N, N))
    targetsets = Vector{Set}(undef, N)
    for src in ProgressBar(1:N)
        vis = Set{Int}()
        for _ in 1:R
            u = src
            load = 1
            ans[u, u] += 1
            while rand(Float64) > p_term
                v = rand(g.adjs[u])
                push!(vis, v)
                load *= U[u, v] * (d[u] - 1) / (1 - p_term)
                ans[src, v] += load
                u = v
            end
        end
        targetsets[src] = vis
    end
    return ans / R, targetsets
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
d, sp_A = DiagAdj(g; aug=true)

invd = inv.(d)
sqrtinvd = sqrt.(invd)
# cbrtinvd = invd .* sqrtinvd

# P = diagm(inv.(d)) * Matrix(sp_A)
# invL = inv(diagm(d) - Matrix(sp_A))
# sqinvL = invL * invL

sqrtinvD = diagm(sqrtinvd)
P = sqrtinvD * sp_A * sqrtinvD
N = size(P, 1)
invP = inv(I - P)
sqinvP = invP * invP

B1, targetsets1 = MonteCarloTest(g, d, P; R=500, p_term=0.05)
B2, targetsets2 = MonteCarloTest(g, d, P; R=500, p_term=0.05)

@show descrip(getratios(targetsets1, targetsets2))

test_mat = B1 * B2'

ratios = []
open("tmp.csv", "w") do io
    write(io, "std, test, ratio\n")
    for i in ProgressBar(1:N)
        std = sqinvP[i, i]
        test = test_mat[i, i]
        ratio = abs(std - test) / std
        append!(ratios, ratio)
        write(io, "$std, $test, $ratio\n")
    end
end
@show descrip(ratios)

