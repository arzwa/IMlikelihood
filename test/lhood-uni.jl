using Optim, Random, IMLikelihood, Parameters, Plots, StatsBase

rng = Random.seed!(743)
la = 1.2
lb = 0.7
m  = 1.2
na = 4
nb = 4
model = UniModel(la, lb, m, na, nb)
tree, l = randtree(rng, model)
plot(tree)
ss = IMLikelihood.getslices(tree)
th = ss.slices[end]
vline!(th .- ss.slices, color=:gray, ls=:dash, alpha=0.5)
for (k, (h, l)) in enumerate(zip(ss.slices, ss.labels))
    annotate!(th-h, 0.5, text("\$$l\$", 10, :center))
    annotate!(th-h, 8.3, text("\$t_$k\$", 10, :center))
end
plot!(size=(350,250))

P, P_ = IMLikelihood.solve(model, ss, kmax=1)
        
rng = Random.seed!(73)
la = 1.2
lb = 0.7
m  = 1.2
na = 20
nb = 20
model = UniModel(la, lb, m, na, nb)
tree, l = randtree(rng, model)
ss = IMLikelihood.getslices(tree)
@time IMLikelihood.lhood(model, ss,)
@time IMLikelihood.lhood_discretized(model, ss, 1e-2, nmin=20)

P, P_ = IMLikelihood.solve(model, ss)

model = UniModel(la, lb, m, 10,10)
ns = map(1:model.na+model.nb-1) do k
    length(IMLikelihood.constraints(model, k))
end 
scatter(ns)

xs = exp.(range(-2, 2, 50)) .* mean([la,lb,m])
θs = [la, lb, m]
labs = ["\$\\lambda_A\$", "\$\\lambda_B\$", "\$m\$"]
map(1:3) do i
    ls = map(xs) do x
        θ = copy(θs); θ[i] = x
        IMLikelihood.lhood_discretized(UniModel(θ..., na, nb), 
            ss, 1e-2, nmin=20)
    end 
    p2 = scatter(xs, ls, ms=2, color=:black, xlabel=labs[i], ylabel="\$\\ell\$")
    vline!([θs[i]])
    vline!([xs[argmax(ls)]], color=:black, ls=:dash)
end |> x->plot(x..., size=(700,200), layout=(1,3), xscale=:log10, margin=4Plots.mm)

f(x) = -IMLikelihood.lhood(UniModel(exp.(x)..., na, nb), ss)
@time res = Optim.optimize(f, log.(θs), LBFGS(), autodiff=:forward);
exp.(res.minimizer)

f(x) = -IMLikelihood.lhood_discretized(UniModel(exp.(x)..., na, nb), 
    ss, 1e-3, nmin=10)
@time res = Optim.optimize(f, log.(θs), LBFGS(), autodiff=:forward);
exp.(res.minimizer)

# many trees
rng = Random.seed!(332)
la = 1.2
lb = 0.7
m  = 1.2
na = 10
nb = 10
model = UniModel(la, lb, m)
nrep = 50
nts = 10

reps = map(1:nrep) do k
    @info k
    lts = map(1:nts) do _ 
        randtree(rng, model, na, nb)
    end
    trees = first.(lts)
    sss = IMLikelihood.getslices.(trees);
    mleob = x->mapreduce(ss-> -IMLikelihood.lhood(UniModel(exp.(x)...), ss), +, sss)
    #mleob(log.([la, lb, m]))
    res = Optim.optimize(mleob, log.([la, lb, m]))
    est = exp.(res.minimizer)
end

xx = [la, lb, m]
labs = ["\$\\lambda_A\$", "\$\\lambda_B\$", "\$m\$"]
map(1:3) do k
    scatter(getindex.(reps, k), color=:black, ms=2, title=labs[k])
    hline!([xx[k]])
end |> x->plot(x..., layout=(1,3), size=(720,200), xlabel="replicate",margin=4Plots.mm)

xx = [la, lb, m]
labs = ["\$\\lambda_A\$", "\$\\lambda_B\$", "\$m\$"]
map(1:3) do k
    stephist(getindex.(reps, k), color=:lightgray, fill=true, alpha=0.5, bins=25, xlabel=labs[k])
    vline!([xx[k]], xlim=(0.2, 2.5))
end |> x->plot(x..., layout=(1,3), size=(720,200), margin=4Plots.mm)


# single tree many reps across the space
rng = Random.seed!(846)
la = 1.0
lb = 1.0
na = 25
nb = 25
ms = exp.(range(log(la*1e-2), log(la*10), 100))
res = map(ms) do m
    @info m
    model = UniModel(la, lb, m)
    tree, l = randtree(rng, model, na, nb)
    ss = IMLikelihood.getslices(tree)
    mleob = x->-IMLikelihood.lhood(UniModel(la, lb, exp(x)), ss)
    res = Optim.optimize(mleob, -5, 5)
    #mleob = x->-IMLikelihood.lhood(UniModel(exp.(x)...), ss)
    #res = Optim.optimize(mleob, log.([la, lb, m]), LBFGS())
    est = exp.(res.minimizer)
end

scatter(ms, last.(res), color=:black, ms=2, 
    xlabel="\$m\$", ylabel="\$\\hat{m}\$")
plot!(x->x, xscale=:log10, yscale=:log10, size=(270,275), 
    title="\$\\lambda_A = \\lambda_B = $la, n_A=n_B=$na\$")

