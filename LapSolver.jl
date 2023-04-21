using LinearAlgebra
using Laplacians
using ProgressBars
using DataStructures
using Statistics
using SparseArrays
using TOML
using Base.Threads
using Base.Filesystem
using Caching

struct Comb
    N::Int
    K::Int
end

function Base.iterate(C::Comb, state=one(UInt128) << C.K - 1)
    state < one(UInt128) << C.N || return nothing
    x = state & (-state)
    y = state + x
    new_state = (((state & (~y)) ÷ x) >> 1) | y
    return ([i + 1 for i in 0:C.N-1 if (one(UInt128) << i) & state != zero(UInt128)], new_state)
    # return (bitstring(state)[end-C.N+1:end], new_state)
end

Base.length(C::Comb) = CombNum(C.N, C.K)
Base.firstindex(::Comb) = 1

@cache function CombNum(N::Int, K::Int)::Int
    return K == 0 ? 1 : N == K ? 1 : CombNum(N - 1, K - 1) + CombNum(N - 1, K)
end

my_display(x) = display(x), println()

my_dot(x::Vector{Float64}) = dot(x, x)

function max_diff(A::VecOrMat{Float64}, B::VecOrMat{Float64})
    ans = @. abs(A - B)
    return reduce(max, ans; init=0)
end

function max_diff_ratio(std::VecOrMat{Float64}, test::VecOrMat{Float64})
    ans = @. abs(std - test) / std
    return reduce(max, ans; init=0)
end

function stat_diff_ratio(std::VecOrMat{Float64}, test::VecOrMat{Float64})
    ans = @. abs(std - test) / std
    return (max=reduce(max, ans; init=0), min=reduce(min, ans; init=1000), mean=mean(ans))
end

function ReadGraph(d::AbstractDict)
    dir_name, file_name, graph_name = d["dir_name"], d["file_name"], d["name"]
    N, M = d["LCC_n"], d["LCC_m"]
    file_path = joinpath(dir_name, "$file_name.txt")
    println("Reading graph $graph_name from $file_path...")
    println("N = $N M = $M")
    A_I, A_J, A_V = Int[], Int[], Float64[]
    sizehint!(A_I, 2 * M), sizehint!(A_J, 2 * M), sizehint!(A_V, 2 * M)
    d = zeros(Float64, N)
    for line in ProgressBar(eachline(file_path))
        if line[begin] in "#%"
            continue
        end
        line = strip(line, ('\t', ' '))
        e = tuple(map(x -> parse(Int, x), split(line, ('\t', ' ')))..., 1)
        u, v, w = e
        append!(A_I, u), append!(A_J, v), append!(A_V, w)
        d[u] += w
        if u != v
            append!(A_I, v), append!(A_J, u), append!(A_V, w)
            d[v] += w
        end
    end
    return d, sparse(A_I, A_J, A_V)
end

function LapDecomp(L::SparseMatrixCSC{Float64,Int64})
    N = size(L, 1)
    B_I, B_J, B_V = Int[], Int[], Float64[]
    w = Float64[]
    rows, vals = rowvals(L), nonzeros(L)
    tot = 0
    for j in 1:N
        for i in nzrange(L, j)
            row, val = rows[i], vals[i]
            if row >= j
                break
            end
            tot += 1
            push!(B_I, row, j), push!(B_J, tot, tot), push!(B_V, 1.0, -1.0)
            append!(w, -val)
        end
    end
    B = sparse(B_I, B_J, B_V)
    x = L * ones(Float64, N)
    return B, spdiagm(sqrt.(w)), spdiagm(sqrt.(x))
end

function ExactAGC(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, S::Vector{Int}, d_sum::Float64)
    N = size(A, 1)
    mask = trues(N)
    for x in S
        mask[x] = 0
    end
    mask_d, mask_A = d[mask], A[mask, mask]
    L = spdiagm(mask_d) - mask_A
    inv_L = inv(Matrix(L))
    x = dot(mask_d, inv_L, mask_d)
    return x / d_sum
end

function ApproxAGC(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, S::Vector{Int}, d_sum::Float64)
    N = size(A, 1)
    mask = trues(N)
    for x in S
        mask[x] = 0
    end
    d, A = d[mask], A[mask, mask]
    L = spdiagm(d) - A
    sol = approxchol_sddm(L)
    x = sol(d)
    return d' * x / d_sum
end

function AGCSeqs(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, S::Vector{Int}, step::Int=1, approx::Bool=false)
    GetAGC = approx ? ApproxAGC : ExactAGC
    T, ans = Int[], Float64[]
    d_sum = sum(d)
    for (i, u) in enumerate(ProgressBar(S))
        push!(T, u)
        if i % step == 0
            push!(ans, GetAGC(d, A, T, d_sum))
        end
    end
    return ans
end

function ExactH(d::Vector{Float64}, L::SparseMatrixCSC{Float64,Int64}, d_sum::Float64)
    N = size(L, 1)
    J = fill(1 / N, (N, N))
    pinv_L = inv(Matrix(L) + J) - J
    vecpi = d ./ d_sum
    Lpi = pinv_L * vecpi
    piLpi = dot(vecpi, Lpi)
    return map(u -> pinv_L[u, u] - 2 * Lpi[u] + piLpi, ProgressBar(1:N))
end

function ExactHK(d::Vector{Float64}, L::SparseMatrixCSC{Float64,Int64}, d_sum::Float64)
    H = ExactH(d, L, d_sum)
    return H, dot(d, H)
end

ExactInitialDelta(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, L::SparseMatrixCSC{Float64,Int64}, d_sum::Float64) = -ExactH(d, L, d_sum)

function ApproxH(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, L::SparseMatrixCSC{Float64,Int64}, d_sum::Float64, c::Int=25)
    N = size(L, 1)
    println("Preparing solver...")
    sol = approxchol_lap(A)
    println("Solver ready.")
    B, W, _ = LapDecomp(L)
    M = size(W, 1)
    m = min(N, ceil(Int, c * log(N)))
    Z = Matrix{Float64}(undef, m, N)
    mapvec = Vector{Float64}(undef, M)
    val = sqrt(1 / m)
    @threads for i in ProgressBar(1:m)
        for j in 1:M
            mapvec[j] = rand(Bool) ? val : -val
        end
        Z[i, :] = sol(B * W * mapvec)
    end
    H = Vector{Float64}(undef, N)
    iter = N >= 500000 ? ProgressBar(1:N) : (1:N)
    Pi = d ./ d_sum
    Z_Pi = Z * Pi
    @threads for u in iter
        H[u] = my_dot(Z[:, u] - Z_Pi)
    end
    return H
end

function ApproxHK(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, L::SparseMatrixCSC{Float64,Int64}, d_sum::Float64, c::Int=25)
    H = ApproxH(d, A, L, d_sum, c)
    return H, dot(d, H)
end

ApproxInitialDelta(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, L::SparseMatrixCSC{Float64,Int64}, d_sum::Float64, c::Int=25) = -ApproxH(d, A, L, d_sum, c)

function ExactDelta(d::Vector{Float64}, L::SparseMatrixCSC{Float64,Int64})
    N = size(L, 1)
    inv_L = inv(Matrix(L))
    sol_d = inv_L * d
    return map(u -> sol_d[u]^2 / inv_L[u, u], 1:N)
end

function ApproxDelta(d::Vector{Float64}, L::SparseMatrixCSC{Float64,Int64}, c::Int=25)
    N = size(L, 1)
    println("Preparing solver...")
    sol = approxchol_sddm(L)
    println("Solver ready.")
    sol_d = sol(d)
    B, W, X = LapDecomp(L)
    M = size(W, 1)
    m = min(N, ceil(Int, c * log(N)))
    val = sqrt(1 / m)
    mapvec1, mapvec2 = Vector{Float64}(undef, M), Vector{Float64}(undef, N)
    Z1, Z2 = Matrix{Float64}(undef, m, N), Matrix{Float64}(undef, m, N)
    @threads for i in ProgressBar(1:m)
        for j in 1:M
            mapvec1[j] = rand(Bool) ? val : -val
        end
        for j in 1:N
            mapvec2[j] = rand(Bool) ? val : -val
        end
        Z1[i, :] = sol(B * W * mapvec1)
        Z2[i, :] = sol(X * mapvec2)
    end
    delta = Vector{Float64}(undef, N)
    iter = N >= 500000 ? ProgressBar(1:N) : (1:N)
    @threads for u in iter
        delta[u] = sol_d[u]^2 / (my_dot(Z1[:, u]) + my_dot(Z2[:, u]))
    end
    return delta
end

function SelectNode(margin::Vector{Float64})
    N = size(margin, 1)
    max_margin = fill(-Inf, nthreads())
    max_margin_arg = zeros(Int, nthreads())
    iter = N >= 500000 ? ProgressBar(1:N) : (1:N)
    @threads for u in iter
        t = threadid()
        if margin[u] > max_margin[t]
            max_margin[t] = margin[u]
            max_margin_arg[t] = u
        end
    end
    return max_margin_arg[argmax(max_margin)]
end

function AbsorbSet(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, K::Int; approx::Bool, c::Int=25)
    N, d_sum = size(A, 1), sum(d)
    L = spdiagm(d) - A
    t = RBTree{Int}()
    S = Int[]
    for i in 1:N
        push!(t, i)
    end
    println("Selecting initial node...")
    delta = approx ? ApproxInitialDelta(d, A, L, d_sum, c) : ExactInitialDelta(d, A, L, d_sum)
    u = SelectNode(delta)
    mask = trues(N)
    mask[u] = 0
    d, L = d[mask], L[mask, mask]
    N -= 1
    push!(S, t[u])
    delete!(t, t[u])
    for k in 2:K
        println("k = $k")
        margin = approx ? ApproxDelta(d, L, c) : ExactDelta(d, L)
        u = SelectNode(margin)
        mask = trues(N)
        mask[u] = 0
        d, L = d[mask], L[mask, mask]
        N -= 1
        push!(S, t[u])
        delete!(t, t[u])
    end
    return S
end

function OptimumSet(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, K::Int)
    N, d_sum = size(A, 1), sum(d)
    min_agc = fill(Inf, nthreads())
    min_agc_arg = Vector{Vector{Int}}(undef, nthreads())

    S_list = Vector{Int}[]

    for S in ProgressBar(Comb(N, K))
        push!(S_list, S)
    end

    @threads for S in ProgressBar(S_list)
        t = threadid()
        agc = ExactAGC(d, A, S, d_sum)
        if agc < min_agc[t]
            min_agc[t] = agc
            min_agc_arg[t] = S
        end
    end
    return min_agc_arg[argmin(min_agc)]
end

function RandomSet(N::Int, K::Int)
    V, S = Set(1:N), Int[]
    for _ in 1:K
        u = rand(V)
        delete!(V, u)
        push!(S, u)
    end
    return S
end

DegreeSet(d::Vector{Float64}, K::Int) = copy(partialsortperm(d, 1:K; rev=true))

function RankSet(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, K::Int)
    d_sum = sum(d)
    L = spdiagm(d) - A
    margin = ApproxInitialDelta(d, A, L, d_sum)
    return copy(partialsortperm(margin, 1:K; rev=true))
end

function PageRankSet(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, K::Int)
    alpha = 0.15
    N = size(A, 1)
    D = spdiagm(d)
    L = D - (1 - alpha) * A
    ans = alpha * D * (inv(Matrix(L)) * fill(1 / N, N))
    return copy(partialsortperm(ans, 1:K, rev=true))
end

function CompareEffect(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, K::Int, step::Int, approx::Bool, inc::Bool, overwrite::Bool)
    println("CompareEffect: Computing different algorithms...")
    agcseq = (overwrite == false && isfile(output_path)) ? TOML.parsefile(output_path) : Dict{AbstractString,Dict{AbstractString,Vector{Float64}}}()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if !haskey(agcseq, graph_name)
                agcseq[graph_name] = Dict{AbstractString,Vector{Float64}}()
            end
            println("graph_name = $graph_name")
            d, sp_A = ReadGraph(tot_d[graph_index])

            if !(inc && haskey(agcseq[graph_name], "Approx"))
                println("Computing absorb set...")
                S = AbsorbSet(d, sp_A, K; approx=true)
                println("Computing AGC on absorb set...")
                agcseq[graph_name]["Approx"] = AGCSeqs(d, sp_A, S, step, approx)
            end

            if !(inc && haskey(agcseq[graph_name], "Exact"))
                println("Computing exact set...")
                S = AbsorbSet(d, sp_A, K; approx=false)
                println("Computing AGC on exact set...")
                agcseq[graph_name]["Exact"] = AGCSeqs(d, sp_A, S, step, approx)
            end

            if !(inc && haskey(agcseq[graph_name], "Top-Absorb"))
                println("Computing rank set...")
                S = RankSet(d, sp_A, K)
                println("Computing AGC on rank set...")
                agcseq[graph_name]["Top-Absorb"] = AGCSeqs(d, sp_A, S, step, approx)
            end

            if !(inc && haskey(agcseq[graph_name], "Top-Degree"))
                println("Computing degree set...")
                S = DegreeSet(d, K)
                println("Computing AGC on degree set...")
                agcseq[graph_name]["Top-Degree"] = AGCSeqs(d, sp_A, S, step, approx)
            end

            if !(inc && haskey(agcseq[graph_name], "Top-PageRank"))
                println("Computing pagerank set...")
                S = PageRankSet(d, sp_A, K)
                println("Computing AGC on pagerank set...")
                agcseq[graph_name]["Top-PageRank"] = AGCSeqs(d, sp_A, S, step, approx)
            end
        end
    finally
        open(io -> TOML.print(io, agcseq), output_path, "w")
    end
end

function CompareOptimumEffect(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, K::Int, inc::Bool, overwrite::Bool)
    println("CompareOptimumEffect: Computing different algorithms...")
    agcseq = (overwrite == false && isfile(output_path)) ? TOML.parsefile(output_path) : Dict{AbstractString,Dict{AbstractString,Vector{Float64}}}()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if !haskey(agcseq, graph_name)
                agcseq[graph_name] = Dict{AbstractString,Vector{Float64}}()
            end
            println("graph_name = $graph_name")
            d, sp_A = ReadGraph(tot_d[graph_index])
            d_sum = sum(d)

            if !(inc && haskey(agcseq[graph_name], "Approx"))
                println("Computing absorb set...")
                S = AbsorbSet(d, sp_A, K; approx=true)
                println("Computing AGC on absorb set...")
                agcseq[graph_name]["Approx"] = AGCSeqs(d, sp_A, S)
            end

            if !(inc && haskey(agcseq[graph_name], "Exact"))
                println("Computing exact set...")
                S = AbsorbSet(d, sp_A, K; approx=false)
                println("Computing AGC on exact set...")
                agcseq[graph_name]["Exact"] = AGCSeqs(d, sp_A, S)
            end

            if !(inc && haskey(agcseq[graph_name], "Optimum"))
                println("Computing optimum set...")
                agcseq[graph_name]["Optimum"] = Float64[]
                for k in 1:K
                    println("Computing AGC on optimum set with capacity $k...")
                    S = OptimumSet(d, sp_A, k)
                    push!(agcseq[graph_name]["Optimum"], ExactAGC(d, sp_A, S, d_sum))
                end
            end
        end
    finally
        open(io -> TOML.print(io, agcseq), output_path, "w")
    end
end

function TestHK(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, c_List::Vector{Int}, inc::Bool, overwrite::Bool)
    println("Executing tests on HK...")
    data = (overwrite == false && isfile(output_path)) ? TOML.parsefile(output_path) : Dict()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if !haskey(data, graph_name)
                data[graph_name] = Dict()
            end
            println("graph_name = $graph_name")
            d, sp_A = ReadGraph(tot_d[graph_index])
            L = spdiagm(d) - sp_A
            d_sum = sum(d)
            if !(inc && haskey(data[graph_name], "Exact"))
                println("Computing ExactHK...")
                BLAS.set_num_threads(4)
                stats = @timed ExactHK(d, L, d_sum)
                data[graph_name]["Exact"] = Dict("Kemeny" => stats.value[2], "time" => stats.time)
            end
            std_kem = data[graph_name]["Exact"]["Kemeny"]
            if !haskey(data[graph_name], "Approx")
                data[graph_name]["Approx"] = Dict()
            end
            BLAS.set_num_threads(16)
            for c in c_List
                if !(inc && haskey(data[graph_name]["Approx"], string(c)))
                    println("Computing Approx$c...")
                    stats = @timed ApproxHK(d, sp_A, L, d_sum, c)
                    kem = stats.value[2]
                    err = abs(std_kem - kem) / std_kem
                    data[graph_name]["Approx"][string(c)] = Dict("Kemeny" => kem, "error" => err, "time" => stats.time)
                end
            end
        end
    finally
        open(io -> TOML.print(io, data), output_path, "w")
    end
end

function ModelHK(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, inc::Bool, overwrite::Bool)
    println("Computing Kemeny constant of model networks...")
    data = (overwrite == false && isfile(output_path)) ? TOML.parsefile(output_path) : Dict()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if inc && haskey(data, graph_name)
                continue
            end
            println("graph_name = $graph_name")
            d, sp_A = ReadGraph(tot_d[graph_index])
            L = spdiagm(d) - sp_A
            d_sum = sum(d)
            stats = @timed ApproxHK(d, sp_A, L, d_sum)
            kem = stats.value[2]
            data[graph_name] = Dict("Kemeny" => kem, "time" => stats.time)
        end
    finally
        open(io -> TOML.print(io, data), output_path, "w")
    end
end

function HKTimer(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, c_List::Vector{Int}, inc::Bool, overwrite::Bool)
    println("Computing running time...")
    run_times = (overwrite == false && isfile(output_path)) ? TOML.parsefile(output_path) : Dict()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if !haskey(run_times, graph_name)
                run_times[graph_name] = Dict()
            end
            println("graph_name = $graph_name")
            d, sp_A = ReadGraph(tot_d[graph_index])
            L = spdiagm(d) - sp_A
            d_sum = sum(d)
            for c in c_List
                if !(inc && haskey(run_times[graph_name], string(c)))
                    stats = @timed ApproxHK(d, sp_A, L, d_sum, c)
                    run_times[graph_name][string(c)] = stats.time
                end
            end
        end
    finally
        open(io -> TOML.print(io, run_times), output_path, "w")
    end
end

function AGCExactTimer(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, K::Int, inc::Bool, overwrite::Bool)
    println("Computing running time...")
    run_times = (overwrite == false && isfile(output_path)) ? TOML.parsefile(output_path) : Dict()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if inc && haskey(run_times, graph_name)
                continue
            end
            println("graph_name = $graph_name")
            d, sp_A = ReadGraph(tot_d[graph_index])
            stats = @timed AbsorbSet(d, sp_A, K; approx=false)
            run_times[graph_name] = stats.time
        end
    finally
        open(io -> TOML.print(io, run_times), output_path, "w")
    end
end

function AGCApproxTimer(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, c_List::Vector{Int}, K::Int, inc::Bool, overwrite::Bool)
    println("Computing running time...")
    run_times = (overwrite == false && isfile(output_path)) ? TOML.parsefile(output_path) : Dict()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if !haskey(run_times, graph_name)
                run_times[graph_name] = Dict()
            end
            println("graph_name = $graph_name")
            d, sp_A = ReadGraph(tot_d[graph_index])
            for c in c_List
                if !(inc && haskey(run_times[graph_name], string(c)))
                    stats = @timed AbsorbSet(d, sp_A, K; approx=true, c=c)
                    run_times[graph_name][string(c)] = stats.time
                end
            end
        end
    finally
        open(io -> TOML.print(io, run_times), output_path, "w")
    end
end

const tot_d = TOML.parsefile("graphs.toml")

BLAS.set_num_threads(16)

ModelHK(tot_d;
    graph_indices=[
        "SmallHanoiExt",
        "Pseudofractal",
        "Koch",
        "CayleyTree",
        "HanoiExt",
    ],
    output_path="outputs/HK_model.toml",
    inc=true, overwrite=false
)