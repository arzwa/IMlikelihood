
using Optim, Random, IMLikelihood, Parameters, Plots, StatsBase

rng = Random.seed!(1263)
la = 1.2
lb = 0.5
mab = 0.9
mba = 0.9
na = 20
nb = 20
model = BiModel(la, lb, mab, mba, na, nb)
trees_l = [randtree(rng, model) for _=1:50]
trees = first.(trees_l)
tree = trees[1]
plot(tree)

ss = IMLikelihood.getslices2(tree)

IMLikelihood.lhood_discretized(model, ss, 1e-3, nmin=10)

xs = exp.(range(-2, 1.2, 15)) .* mean([la,lb,mab,mba])        
θs = [la, lb, mab, mba]
labs = ["\$\\lambda_A\$", "\$\\lambda_B\$", "\$m_{AB}\$", "\$m_{BA}\$"]
res = map(1:4) do i
    @info i
    ls = map(xs) do x
        θ = copy(θs); θ[i] = x
        IMLikelihood.lhood_discretized(BiModel(θ..., na, nb), ss, 1e-3, nmin=10)
    end 
    p2 = scatter(xs, ls, ms=2, color=:black, xlabel=labs[i], ylabel="\$\\ell\$")
    vline!([θs[i]])
    vline!([xs[argmax(ls)]], color=:black, ls=:dash)
end 

plot!(res[1], title="\$n_A=n_B=$na\$")
plot(res..., size=(900,200), layout=(1,4), 
    xscale=:log10, bottom_margin=6Plots.mm)


# ---------------
rng = Random.seed!(623)
la = 1.0
lb = 1.0
mab = 1.0
mba = 1.0
na = 20
nb = 20
model = BiModel(la, lb, mab, mba, na, nb)
trees_l = [randtree(rng, model) for _=1:50]
trees = first.(trees_l)
sss = map(t->IMLikelihood.getslices2(t), trees)
ms = exp2.(range(-2, 2, 25)) .* mab
ls = map(ms) do m
    @info m
    mapreduce(ss->IMLikelihood.lhood_discretized(
        BiModel(la, lb, m, m, na, nb), ss, 1e-3, nmin=10), +, sss)
end 

p2 = scatter(ms, ls, ms=2, color=:black, xscale=:log2, 
    xlabel="\$m\$", ylabel="\$\\ell\$",
    title="\$n_A=n_B=$na, m_{AB} = m_{BA} = $mab, \\lambda_A = \\lambda_B = $la, n_{tree} = 50\$", titlefont=9)
vline!([mab])
vline!([ms[argmax(ls)]], color=:black, ls=:dash)

plot!(res[1], title="\$n_A=n_B=$na\$")
plot(res..., size=(230*length(res),200), layout=(1,length(res)), 
    xscale=:log10, bottom_margin=6Plots.mm)


rng = Random.seed!(846)
la = 1.0
lb = 1.0
mab = 1.0
na = 20
nb = 20
mbas = exp2.(-2:0.4:2) .* mab
res = map(mbas) do mba
    @info mba
    model = BiModel(la, lb, mab, mba, na, nb)
    tree, l = randtree(rng, model)
    ss = IMLikelihood.getslices2(tree)
    mleob = x->-IMLikelihood.lhood_discretized(BiModel(la, lb, mab, x, na, nb), ss, 1e-3, nmin=10)
    res = Optim.optimize(mleob, -5, 5)
    #mleob = x->-IMLikelihood.lhood(UniModel(exp.(x)...), ss)
    #res = Optim.optimize(mleob, log.([la, lb, m]), LBFGS())
    est = exp.(res.minimizer)
    @info est
    est
end

scatter(ms, last.(res), color=:black, ms=2, 
    xlabel="\$m\$", ylabel="\$\\hat{m}\$")
plot!(x->x, xscale=:log10, yscale=:log10, size=(270,275), 
    title="\$\\lambda_A = \\lambda_B = $la, n_A=n_B=$na\$")
