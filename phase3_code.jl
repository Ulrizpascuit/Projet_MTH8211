# %%
using Pkg
Pkg.activate("Phase3_projet")
Pkg.add(["LinearAlgebra", "SparseArrays", "Krylov", "BenchmarkTools", "SuiteSparseMatrixCollection", "MatrixMarket"])
using LinearAlgebra, SparseArrays, Krylov, BenchmarkTools, SuiteSparseMatrixCollection, MatrixMarket, Random, Plots

# %%
function badly_conditioned_rectangular_matrix(m, n, kappa)
    U, _ = qr(randn(m, n))
    V, _ = qr(randn(n, n))
    s = range(1.0, 1.0/kappa, length=n)
    S = Diagonal(s)
    A = U * S * V'
    return Matrix(A)
end

function badly_conditioned_underdetermined_matrix(m, n, kappa)
    U, _ = qr(randn(n, m))
    V, _ = qr(randn(m, m))
    s = range(1.0, 1.0 / kappa, length=m)
    S = Diagonal(s)
    A_tall = U * S * V'
    return Matrix(A_tall')[1:m, 1:n]
end

function lsrn_lsqr(A, b; gamma=2.0, tol=1e-10, itmax=2000)
    m, n = size(A)
    s = ceil(Int, gamma * n) 
    G = randn(s, m)
    Ã = G * A
    Ũ, Σ̃, Ṽ = svd(Ã; full=false)
    r = sum(Σ̃.> 1e-12)
    Σinv = Diagonal(1.0 ./ Σ̃[1:r])
    V_r = Ṽ[:,1:r]
    N = V_r * Σinv
    AN = A * N
    ŷ, histo = lsqr(AN, b; atol=tol, btol=tol, itmax=itmax, history=true)

    x̂ = N * ŷ
    return x̂, histo
end

function lsrn_lsqr_underdetermined(A, b; gamma=2.0, tol=1e-10, itmax=2000)
    m, n = size(A)
    s = ceil(Int, gamma * m)
    G = randn(n, s)
    Ã = A * G
    Ũ, Σ̃, Ṽ = svd(Ã; full=false)
    r = sum(Σ̃ .> 1e-12)
    U_r = Ũ[:,1:r]
    Σ_r = Σ̃[1:r]
    Σinv = Diagonal(1.0 ./ Σ_r)
    M = U_r * Σinv      
    Mt = M'
    At_pre = Mt * A    
    bt_pre = Mt * b     

    x̂, histo = lsqr(At_pre, bt_pre; atol=tol, btol=tol, itmax=itmax, history=true)

    return x̂, histo
end

# %% [markdown]
# # Conditionnement après préconditionnement

# %%
function precondition_over(A; γ=2.0)
    m, n = size(A)
    s = ceil(Int, γ * n)
    G = randn(s, m)
    Ã = G * A
    Ũ, Σ̃, Ṽ = svd(Ã; full=false)
    r = sum(Σ̃ .> 1e-12)
    Σinv = Diagonal(1.0 ./ Σ̃[1:r])
    V_r = Ṽ[:, 1:r]
    N = V_r * Σinv
    AN = A * N
    return AN
end

function precondition_under(A; γ=2.0)
    m, n = size(A)
    s = ceil(Int, γ * m)
    G = randn(n, s)
    Ã = A * G
    Ũ, Σ̃, Ṽ = svd(Ã; full=false)
    r = sum(Σ̃ .> 1e-12)
    U_r = Ũ[:, 1:r]
    Σinv = Diagonal(1.0 ./ Σ̃[1:r])
    M = U_r * Σinv
    Mt = M'
    At_pre = Mt * A
    return At_pre
end

# %%
kappas = 10 .^ range(1, 7, length=9)
dims = [(100, 100), (1000, 100), (10000, 100)]
dims_under = [(100, 100), (100, 1000), (100, 10000)]

results = Dict()

for (m,n) in dims
    ys = Float64[]
    for κ in kappas
        A = badly_conditioned_rectangular_matrix(m, n, κ)
        AN = precondition_over(A)
        push!(ys, cond(AN))
    end
    results[("over", m, n)] = ys
end

for (m,n) in dims_under
    ys = Float64[]
    for κ in kappas
        A = badly_conditioned_underdetermined_matrix(m, n, κ)
        AN = precondition_under(A)
        push!(ys, cond(AN))
    end
    results[("under", m, n)] = ys
end

plt = plot(xscale=:log10,
    xlabel="Cond(A)", ylabel="Cond(A preconditionnée)",
    title="Amélioration du conditionnement")

for (m,n) in dims
    plot!(plt, kappas, results[("over", m, n)],
        lw=2, label="Surdéterminé m=$m, n=$n")
end

for (m,n) in dims_under
    plot!(plt, kappas, results[("under", m, n)],
        lw=2, linestyle=:dash, label="Sous-déterminé m=$m, n=$n")
end
savefig(plt, "Fig/cond_improvement.png")
savefig(plt, "Fig/cond_improvement.pdf")

# %% [markdown]
# # Système sur-déterminé

# %% [markdown]
# ### ColorMap de Gain en focntion du conditionnement allant de 1 à 1000000 et ratio m/n allant de 1 à 1000 (avec m0=n0=1000)

# %%
ms = Int.(round.(vec(10 .^ range(3, 6, length=4))))
n = 1000
kappa_vals = vec(10 .^ range(0, 6, length=7))
gain = zeros(length(kappa_vals),length(ms))
for (j, m) in enumerate(ms)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_rectangular_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("m           : $(m)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =heatmap(ms, kappa_vals, gain;
    xscale=:log10,
    yscale=:log10,
    xlabel="Ratio m/n",
    ylabel="Conditionnement",
    color=:viridis,
    title="Gain",
    clim=(-1, 1)
)
savefig(plt,"Fig/Gain_VS_Condi-ratio_surdet_1000.png")
savefig(plt,"Fig/Gain_VS_Condi-ratio_surdet_1000.pdf")

# %% [markdown]
# ### ColorMap de Gain en focntion du conditionnement allant de 1 à 1000000 et ratio m/n allant de 1 à 1000 (avec m0=n0=100)

# %%
ms = Int.(round.(vec(10 .^ range(2, 5, length=4))))
n = 100
kappa_vals = vec(10 .^ range(0, 6, length=7))
gain = zeros(length(kappa_vals),length(ms))
for (j, m) in enumerate(ms)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_rectangular_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("m           : $(m)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =heatmap(ms, kappa_vals, gain;
    xscale=:log10,
    yscale=:log10,
    xlabel="Ratio m/n",
    ylabel="Conditionnement",
    color=:viridis,
    title="Gain",
    clim=(-1, 1)
)
savefig(plt,"Fig/Gain_VS_Condi-ratio_surdet_100.png")
savefig(plt,"Fig/Gain_VS_Condi-ratio_surdet_100.pdf")

# %% [markdown]
# ### ColorMap de Gain en focntion du conditionnement allant de 1 à 1000000 et ratio m/n allant de 1 à 1000 (avec m0=n0=10)

# %%
ms = Int.(round.(vec(10 .^ range(1, 4, length=4))))
n = 10
kappa_vals = vec(10 .^ range(0, 6, length=7))
gain = zeros(length(kappa_vals),length(ms))
for (j, m) in enumerate(ms)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_rectangular_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("m           : $(m)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =heatmap(ms, kappa_vals, gain;
    xscale=:log10,
    yscale=:log10,
    xlabel="Ratio m/n",
    ylabel="Conditionnement",
    color=:viridis,
    title="Gain",
    clim=(-1, 1)
)
savefig(plt,"Fig/Gain_VS_Condi-ratio_surdet_10.png")
savefig(plt,"Fig/Gain_VS_Condi-ratio_surdet_10.pdf")

# %% [markdown]
# ### Gain vs ratio m/n

# %%
ms = Int.(round.(vec(10 .^ range(2, 5, length=500))))
n = 100
kappa_vals = 10
gain = zeros(length(kappa_vals),length(ms))
for (j, m) in enumerate(ms)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_rectangular_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("m           : $(m)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =plot(ms, vec(gain);
    xscale=:log10,
    xlabel="Ratio m/n",
    ylabel="Gain",
    color=:viridis,
    title="Gain en fonction du ratio m/n",
)
savefig(plt,"Fig/Gain_VS_ratio_surdet_100.png")
savefig(plt,"Fig/Gain_VS_ratio_surdet_100.pdf")

# %% [markdown]
# ### Gain vs conditionnement

# %%
ms = 10000
n = 100
kappa_vals = vec(10 .^ range(0, 2, length=200))
gain = zeros(length(kappa_vals),length(ms))
for (j, m) in enumerate(ms)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_rectangular_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("m           : $(m)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =plot(kappa_vals, vec(gain);
    xscale=:log10,
    xlabel="Conditionnement",
    ylabel="Gain",
    color=:viridis,
    title="Gain en fonction du conditionnement",
)
savefig(plt,"Fig/Gain_VS_condi_surdet_100-10000.png")
savefig(plt,"Fig/Gain_VS_condi_surdet_100-10000.pdf")


# %% [markdown]
# # Système sous-déterminé

# %% [markdown]
# ### ColorMap de Gain en focntion du conditionnement allant de 1 à 1000000 et ratio n/m allant de 1 à 1000 (avec m0=n0=10)

# %%
m = 10
ns = Int.(round.(vec(10 .^ range(1, 4, length=4))))
kappa_vals = vec(10 .^ range(0, 6, length=7))
gain = zeros(length(kappa_vals),length(ns))
for (j, n) in enumerate(ns)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_underdetermined_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr_underdetermined(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("n           : $(n)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =heatmap(ns, kappa_vals, gain;
    xscale=:log10,
    yscale=:log10,
    xlabel="Nombre de colonnes",
    ylabel="Conditionnement",
    color=:viridis,
    title="Niter/Niter_LSRN",
    clim=(-1, 1)
)
savefig(plt,"Fig/Gain_VS_Condi-ratio_sousdet_10.png")
savefig(plt,"Fig/Gain_VS_Condi-ratio_sousdet_10.pdf")

# %% [markdown]
# ### ColorMap de Gain en focntion du conditionnement allant de 1 à 1000000 et ratio n/m allant de 1 à 1000 (avec m0=n0=100)

# %%
m = 100
ns = Int.(round.(vec(10 .^ range(2, 5, length=4))))
kappa_vals = vec(10 .^ range(0, 6, length=7))
gain = zeros(length(kappa_vals),length(ns))
for (j, n) in enumerate(ns)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_underdetermined_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr_underdetermined(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("n           : $(n)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =heatmap(ns, kappa_vals, gain;
    xscale=:log10,
    yscale=:log10,
    xlabel="Nombre de colonnes",
    ylabel="Conditionnement",
    color=:viridis,
    title="Niter/Niter_LSRN",
    clim=(-1, 1)
)
savefig(plt,"Fig/Gain_VS_Condi-ratio_sousdet_100.png")
savefig(plt,"Fig/Gain_VS_Condi-ratio_sousdet_100.pdf")

# %% [markdown]
# ### ColorMap de Gain en focntion du conditionnement allant de 1 à 1000000 et ratio n/m allant de 1 à 1000 (avec m0=n0=1000)

# %%
m = 1000
ns = Int.(round.(vec(10 .^ range(3, 6, length=4))))
kappa_vals = vec(10 .^ range(0, 6, length=7))
gain = zeros(length(kappa_vals),length(ns))
for (j, n) in enumerate(ns)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_underdetermined_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr_underdetermined(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("n           : $(n)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =heatmap(ns, kappa_vals, gain;
    xscale=:log10,
    yscale=:log10,
    xlabel="Nombre de colonnes",
    ylabel="Conditionnement",
    color=:viridis,
    title="Niter/Niter_LSRN",
    clim=(-1, 1)
)
savefig(plt,"Fig/Gain_VS_Condi-ratio_sousdet_1000.png")
savefig(plt,"Fig/Gain_VS_Condi-ratio_sousdet_1000.pdf")

# %% [markdown]
# ### Gain vs ratio m/n

# %%
m = 100
ns = Int.(round.(vec(10 .^ range(2, 5, length=500))))
kappa_vals = 10
gain = zeros(length(kappa_vals),length(ns))
for (j, n) in enumerate(ns)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_underdetermined_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr_underdetermined(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("n           : $(n)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =plot(ns, vec(gain);
    xscale=:log10,
    xlabel="Ratio n/m",
    ylabel="Gain",
    color=:viridis,
    title="Gain en fonction du ratio n/m",
)
savefig(plt,"Fig/Gain_VS_ratio_sousdet_100.png")
savefig(plt,"Fig/Gain_VS_ratio_sousdet_100.pdf")

# %% [markdown]
# ### Gain vs conditionnement

# %%
m = 100
ns = 100000
kappa_vals = vec(10 .^ range(0, 2, length=200))
gain = zeros(length(kappa_vals),length(ns))
for (j, n) in enumerate(ns)
    for (i, kappa) in enumerate(kappa_vals)
        A = badly_conditioned_underdetermined_matrix(m, n, kappa)
        x = randn(n)
        b = A * x
        res1, hist1 = lsqr(A, b; atol=1e-10, btol=1e-10, itmax=2000, history=true)
        x_prec, hist2 = lsrn_lsqr_underdetermined(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        rationiter = hist2.niter/hist1.niter
        println("n           : $(n)")
        println("Kappa           : $(kappa)")
        println("  ")
        gain[i,j] = max(1 - rationiter,-1)
    end
end
plt =plot(kappa_vals, vec(gain);
    xscale=:log10,
    xlabel="Conditionnement",
    ylabel="Gain",
    color=:viridis,
    title="Gain en conditionnement",
)
savefig(plt,"Fig/Gain_VS_condi_sousdet_100-10000.png")
savefig(plt,"Fig/Gain_VS_condi_sousdet_100-10000.pdf")

# %% [markdown]
# # Parallélisme

# %%
ms = Int.([1000 10000 100000 1000000])
n = 1000
kappa = 70
threads_list = [1, 2, 4, 8, 16, 32]
times = zeros(length(threads_list),length(ms))
for (j,m) in enumerate(ms)
    for (i, t) in enumerate(threads_list)
        BLAS.set_num_threads(t) 
        println("Nombre de threads : $t")
        A = badly_conditioned_rectangular_matrix(m, n, kappa)
        x_true = randn(n)
        b = A * x_true

        lsrn_lsqr(A, b; gamma=2.0, tol=1e-10, itmax=2000)

        t_exec = @elapsed lsrn_lsqr(A, b; gamma=2.0, tol=1e-10, itmax=2000)
        times[i,j] = t_exec
        println("Temps d'exécution : $(round(t_exec, digits=4)) sec")
    end
end
plt = plot(
    threads_list, times[:,1],
    marker=:o, lw=2,
    label="(m,n) = (1000,1000)",
    xlabel="Nombre de threads BLAS",
    ylabel="Temps d'exécution (s)",
    title="Performance en fonction des threads",
    legend=true
)
for i=2:length(ms)
    plot!(plt, threads_list, times[:,i],marker=:o,
        lw=2, label="(m,n) = ($(ms[i]),1000)")
end
savefig(plt,"Fig/Time_VS_Threads.png")
savefig(plt,"Fig/Time_VS_Threads.pdf")

# %%
plt = plot(
    threads_list, times[:,1],
    marker=:o, lw=2,
    label="(m,n) = (1000,1000)",
    xlabel="Nombre de threads BLAS",
    ylabel="Temps d'exécution (s)",
    title="Performance en fonction des threads",
    legend=true
)
for i=2:length(ms)
    plot!(plt, threads_list, times[:,i],marker=:o,
        lw=2, label="(m,n) = ($(ms[i]),1000)")
end
savefig(plt,"Fig/Time_VS_Threads.png")
savefig(plt,"Fig/Time_VS_Threads.pdf")


