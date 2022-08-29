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

function ComputeMapMat(N::Int, M::Int)
    m = Matrix{Float64}(undef, N, M)
    val = 1 / sqrt(N)
    for i in 1:N*M
        m[i] = rand(Bool) ? val : -val
    end
    return m
end

function ComputeExactMANC(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, S::Vector{Int}, d_sum::Float64)
    N = size(A, 1)
    mask = trues(N)
    for x in S
        mask[x] = 0
    end
    d, A = d[mask], A[mask, mask]
    L = spdiagm(d) - A
    inv_L = inv(Matrix(L))
    x = dot(d, inv_L, d)
    return x / d_sum
end

function ComputeApproxMANC(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, S::Vector{Int}, d_sum::Float64)
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

function ComputeMANCSeries(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, S::Vector{Int}, approx::Bool=false)
    ComputeMANC = approx ? ComputeApproxMANC : ComputeExactMANC
    T, ans = Int[], Float64[]
    d_sum = sum(d)
    for u in ProgressBar(S)
        push!(T, u)
        push!(ans, ComputeMANC(d, A, T, d_sum))
    end
    return ans
end

function BaseVector(N::Int, x::Int)
    v = zeros(Float64, N)
    v[x] = one(Float64)
    return v
end

function ComputeExactInitialMargin(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, L::SparseMatrixCSC{Float64,Int64}, d_sum::Float64)
    N = size(L, 1)
    pinv_L = pinv(Matrix(L))
    Pi = d ./ d_sum
    e = map(u -> BaseVector(N, u) - Pi, 1:N)
    return map(u -> -dot(e[u], pinv_L, e[u]), 1:N)
end

function ComputeApproxInitialMargin(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, L::SparseMatrixCSC{Float64,Int64}, d_sum::Float64, c::Int=50)
    N = size(L, 1)
    sol = approxchol_lap(A)
    B, W, _ = LapDecomp(L)
    M = size(W, 1)
    m = min(N, ceil(Int, c * log(N)))
    Q = ComputeMapMat(m, M)
    Q = Q * W * B'
    Z = Matrix{Float64}(undef, m, N)
    @threads for i in ProgressBar(1:m)
        Z[i, :] = sol(Q[i, :])
    end
    margin = Vector{Float64}(undef, N)
    iter = N >= 500000 ? ProgressBar(1:N) : (1:N)
    Pi = d ./ d_sum
    Z_Pi = Z * Pi
    @threads for u in iter
        margin[u] = -my_dot(Z[:, u] - Z_Pi)
    end
    return margin
end

function ComputeExactMargin(d::Vector{Float64}, L::SparseMatrixCSC{Float64,Int64})
    N = size(L, 1)
    inv_L = inv(Matrix(L))
    sol_d = inv_L * d
    return map(u -> sol_d[u]^2 / inv_L[u, u], 1:N)
end

function ComputeApproxMargin(d::Vector{Float64}, L::SparseMatrixCSC{Float64,Int64}, c::Int=50)
    N = size(L, 1)
    sol = approxchol_sddm(L)
    sol_d = sol(d)
    B, W, X = LapDecomp(L)
    M = size(W, 1)
    m = min(N, ceil(Int, c * log(N)))
    Q, R = ComputeMapMat(m, M), ComputeMapMat(m, N)
    Q = Q * W * B'
    R = R * X
    Z1, Z2 = Matrix{Float64}(undef, m, N), Matrix{Float64}(undef, m, N)
    @threads for i in ProgressBar(1:m)
        Z1[i, :] = sol(Q[i, :])
        Z2[i, :] = sol(R[i, :])
    end
    margin = Vector{Float64}(undef, N)
    iter = N >= 500000 ? ProgressBar(1:N) : (1:N)
    @threads for u in iter
        margin[u] = sol_d[u]^2 / (my_dot(Z1[:, u]) + my_dot(Z2[:, u]))
    end
    return margin
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

function ComputeAbsorbSet(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, K::Int; approx::Bool=true)
    N, d_sum = size(A, 1), sum(d)
    L = spdiagm(d) - A
    t = RBTree{Int}()
    S = Int[]
    for i in 1:N
        push!(t, i)
    end
    println("Selecting initial node...")
    ComputeMargin = approx ? ComputeApproxInitialMargin : ComputeExactInitialMargin
    margin = ComputeMargin(d, A, L, d_sum)
    u = SelectNode(margin)
    mask = trues(N)
    mask[u] = 0
    d, L = d[mask], L[mask, mask]
    N -= 1
    push!(S, t[u])
    delete!(t, t[u])
    for k in 2:K
        println("k = $k")
        ComputeMargin = approx ? ComputeApproxMargin : ComputeExactMargin
        margin = ComputeMargin(d, L)
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

function ComputeOptimumSet(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, K::Int)
    N, d_sum = size(A, 1), sum(d)
    # opt_vec = argmin(S -> ComputeExactMANC(d, A, S, d_sum), Comb(N, K))
    min_manc = fill(Inf, nthreads())
    min_manc_arg = [collect(1:K) for _ in 1:nthreads()]
    for S in ProgressBar(Comb(N, K))
        t = threadid()
        manc = ComputeExactMANC(d, A, S, d_sum)
        if manc < min_manc[t]
            min_manc[t] = manc
            min_manc_arg[t] = S
        end
    end
    opt_vec = min_manc_arg[argmin(min_manc)]
    opt_set = Set(opt_vec)
    ans = Int[]
    while !isempty(opt_set)
        node = argmin(u -> ComputeExactMANC(d, A, vcat(ans, u), d_sum), opt_set)
        delete!(opt_set, node)
        append!(ans, node)
    end
    return ans
end

function ComputeRandomSet(N::Int, K::Int)
    V, S = Set(1:N), Int[]
    for _ in 1:K
        u = rand(V)
        delete!(V, u)
        push!(S, u)
    end
    return S
end

ComputeDegreeSet(d::Vector{Float64}, K::Int) = copy(partialsortperm(d, 1:K; rev=true))

function ComputeRankSet(d::Vector{Float64}, A::SparseMatrixCSC{Float64,Int64}, K::Int)
    d_sum = sum(d)
    L = spdiagm(d) - A
    margin = ComputeApproxInitialMargin(d, A, L, d_sum)
    return copy(partialsortperm(margin, 1:K; rev=true))
end

function CompareEffect(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, K::Int, approx::Bool, inc::Bool)
    println("CompareEffect: Computing different algorithms...")
    mancs = (inc && isfile(output_path)) ? TOML.parsefile(output_path) : Dict{AbstractString,Dict{AbstractString,Vector{Float64}}}()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if haskey(mancs, graph_name)
                continue
            end
            println("graph_name = $graph_name")
            manc = Dict{AbstractString,Vector{Float64}}()
            d, sp_A = ReadGraph(tot_d[graph_index])
            N = size(sp_A, 1)

            println("Computing absorb set...")
            S = ComputeAbsorbSet(d, sp_A, K; approx=true)
            println("Computing MANC on absorb set...")
            manc["Approx"] = ComputeMANCSeries(d, sp_A, S, approx)

            println("Computing exact set...")
            S = ComputeAbsorbSet(d, sp_A, K; approx=false)
            println("Computing MANC on exact set...")
            manc["Exact"] = ComputeMANCSeries(d, sp_A, S, approx)

            println("Computing rank set...")
            S = ComputeRankSet(d, sp_A, K)
            println("Computing MANC on rank set...")
            manc["Top-Absorb"] = ComputeMANCSeries(d, sp_A, S, approx)

            println("Computing degree set...")
            S = ComputeDegreeSet(d, K)
            println("Computing MANC on degree set...")
            manc["Top-Degree"] = ComputeMANCSeries(d, sp_A, S, approx)

            mancs[graph_name] = manc
        end
    finally
        open(io -> TOML.print(io, mancs), output_path, "w")
    end
end

function CompareOptimumEffect(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, K::Int, inc::Bool)
    println("CompareOptimumEffect: Computing different algorithms...")
    mancs = (inc && isfile(output_path)) ? TOML.parsefile(output_path) : Dict{AbstractString,Dict{AbstractString,Vector{Float64}}}()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if haskey(mancs, graph_name)
                continue
            end
            println("graph_name = $graph_name")
            manc = Dict{AbstractString,Vector{Float64}}()
            d, sp_A = ReadGraph(tot_d[graph_index])
            N = size(sp_A, 1)

            println("Computing absorb set...")
            S = ComputeAbsorbSet(d, sp_A, K; approx=true)
            println("Computing MANC on absorb set...")
            manc["Approx"] = ComputeMANCSeries(d, sp_A, S)

            println("Computing exact set...")
            S = ComputeAbsorbSet(d, sp_A, K; approx=false)
            println("Computing MANC on exact set...")
            manc["Exact"] = ComputeMANCSeries(d, sp_A, S)

            println("Computing rank set...")
            S = ComputeRankSet(d, sp_A, K)
            println("Computing MANC on rank set...")
            manc["Top-Absorb"] = ComputeMANCSeries(d, sp_A, S)

            println("Computing degree set...")
            S = ComputeDegreeSet(d, K)
            println("Computing MANC on degree set...")
            manc["Top-Degree"] = ComputeMANCSeries(d, sp_A, S)

            println("Computing optimum set...")
            S = ComputeOptimumSet(d, sp_A, K)
            println("Computing MANC on optimum set...")
            manc["Optimum"] = ComputeMANCSeries(d, sp_A, S)

            mancs[graph_name] = manc
        end
    finally
        open(io -> TOML.print(io, mancs), output_path, "w")
    end
end

function ComputeHistogramData(ratios::Vector{Float64}, limits::Vector{Float64})
    N = length(ratios)
    num_limits = length(limits)
    cnt = zeros(Int, num_limits)
    for ratio in ratios
        for (i, limit) in enumerate(limits)
            if ratio < limit
                cnt[i] += 1
                break
            end
        end
    end
    return cnt ./ N
end

function ComputeMarginError(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, coeffs::Vector{Int}, limits::Vector{Float64}, inc::Bool)
    println("Computing margin error...")
    append!(limits, 1.01)
    sort!(limits)
    errors = (inc && isfile(output_path)) ? TOML.parsefile(output_path) : Dict{AbstractString,Dict{AbstractString,Any}}()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if haskey(errors, graph_name)
                continue
            end
            println("graph_name = $graph_name")
            d, sp_A = ReadGraph(tot_d[graph_index])
            N = size(sp_A, 1)
            d_sum = sum(d)
            L = spdiagm(d) - sp_A
            margin = ComputeExactInitialMargin(d, sp_A, L, d_sum)
            u = SelectNode(margin)
            mask = trues(N)
            mask[u] = 0
            d, L = d[mask], L[mask, mask]
            exact_margin = ComputeExactMargin(d, L)
            single_error = Dict{AbstractString,Any}()
            for coeff in coeffs
                println("coeff = $coeff")
                approx_margin = ComputeApproxMargin(d, L, coeff)
                ratios = @. abs(approx_margin - exact_margin) / exact_margin
                cnt = ComputeHistogramData(ratios, limits)
                single_error["$coeff"] = Dict("distribution" => cnt, "mean" => mean(ratios))
            end
            errors[graph_name] = single_error
        end
    finally
        open(io -> TOML.print(io, errors), output_path, "w")
    end
end

function ComputeRunningTime(tot_d::AbstractDict; graph_indices::Vector{String}, output_path::AbstractString, K::Int, approx::Bool, inc::Bool)
    println("Computing running time...")
    run_times = (inc && isfile(output_path)) ? TOML.parsefile(output_path) : Dict{AbstractString,Float64}()
    try
        for graph_index in graph_indices
            graph_name = tot_d[graph_index]["name"]
            if haskey(run_times, graph_name)
                continue
            end
            println("graph_name = $graph_name")
            d, sp_A = ReadGraph(tot_d[graph_index])
            stats = @timed ComputeAbsorbSet(d, sp_A, K; approx=approx)
            run_times[graph_name] = stats.time
        end
    finally
        open(io -> TOML.print(io, run_times), output_path, "w")
    end
end

const tot_d = TOML.parsefile("graphs.toml")

# CompareOptimumEffect(tot_d;
#     graph_indices=[
#         "Zachary_karate_club",
#         "Zebra",
#         "Contiguous_USA",
#         "Les_Miserables",
#     ],
#     output_path="outputs/compare_effects_optimum.toml",
#     K=5, inc=true
# )


BLAS.set_num_threads(32)

# ComputeMarginError(tot_d;
#     graph_indices=[
#         "Hamsterster_households",
#         "Euroroads",
#         "Hamsterster_friends",
#         "ego-Facebook",
#         "CA-GrQc",
#         "US_power_grid",
#     ],
#     output_path="outputs/margin_errors.toml",
#     coeffs=[20, 50, 100, 200],
#     limits=[0.1, 0.2, 0.3, 0.4, 0.5],
#     inc=true
# )

# CompareEffect(tot_d;
#     graph_indices=[
#         "Euroroads",
#         "Hamsterster_friends",
#         "ego-Facebook",
#         "CA-GrQc",
#         "US_power_grid",
#     ],
#     output_path="outputs/compare_effects_exact.toml",
#     K=50, approx=false, inc=true
# )

ComputeRunningTime(tot_d;
    graph_indices=[
        "Zachary_karate_club",
        "Zebra",
        # "Contiguous_USA",
        # "Les_Miserables",
        # "Jazz_musicians",
        # "Euroroads",
        # "Hamsterster_friends",
        # "ego-Facebook",
        # "CA-GrQc",
        # "US_power_grid",
        # "Reactome",
        # "CA-HepTh",
        # "Sister_cities",
    ],
    output_path="outputs/running_time_exact.toml",
    K=10, approx=false, inc=true
)

# ComputeRunningTime(tot_d;
#     graph_indices=[
#         "Zebra",
#         "Zachary_karate_club",
#         "Contiguous_USA",
#         "Les_Miserables",
#         "Jazz_musicians",
#         "Euroroads",
#         "Hamsterster_friends",
#         "ego-Facebook",
#         "CA-GrQc",
#         "US_power_grid",
#         "Reactome",
#         "CA-HepTh",
#         "Sister_cities",
#         "CA-HepPh",
#         "CAIDA",
#         "loc-Gowalla",
#         "com-Amazon",
#         "Dogster_friends",
#         "roadNet-PA",
#         "roadNet-CA",
#     ],
#     output_path="outputs/running_time_approx.toml",
#     K=10, approx=true, inc=true
# )