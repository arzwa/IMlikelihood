
rng = Random.seed!(73)
la = 1.2
lb = 1.0
mab = 0.2
mba = 0.3
na = 4
nb = 4
model = BiModel(la, lb, mab, mba, na, nb)
tree, l = randtree(rng, model)
plot(tree)

ss = IMLikelihood.getslices2(tree)

IMLikelihood.solve(model, ss)

IMLikelihood.lhood_discretized(model, ss, 1e-3, nmin=10)

xs = exp.(range(-2, 2, 50)) .* mean([la,lb,mab,mba])
θs = [la, lb, mab, mba]
labs = ["\$\\lambda_A\$", "\$\\lambda_B\$", "\$m_{AB}\$", "\$m_{BA}\$"]
map(1:4) do i
    ls = map(xs) do x
        θ = copy(θs); θ[i] = x
        IMLikelihood.lhood_discretized(BiModel(θ..., na, nb), ss, 1e-3, nmin=10)
    end 
    p2 = scatter(xs, ls, ms=2, color=:black, xlabel=labs[i], ylabel="\$\\ell\$")
    vline!([θs[i]])
    vline!([xs[argmax(ls)]], color=:black, ls=:dash)
end |> x->plot(x..., size=(800,200), layout=(1,4), xscale=:log10, margin=4Plots.mm)
