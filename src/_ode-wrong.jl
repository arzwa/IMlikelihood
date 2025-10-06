
function lhood(model, tree)
    p = solve(model, tree)[end,end,1,end]
    p <= 0.0 ? -Inf : log(p)
end

function solve(model::Model{T}, tree) where T
    nv  = NewickTree.nv(tree)
    ths = treeheights(tree)
    hs  = unique(sort(collect(values(ths))))
    hs  = [hs; hs[end]]
    D   = zeros(length(hs)-1, nv, 2, 2)  # time slice x node x pop branch x begin/end
    for n in getleaves(tree)
        idx = name(n) == "A" ? 1 : 2
        D[1,id(n),idx,1] = 1.0
    end    
    for k=1:length(hs)-1
        prob = solve_slice!(D, model, tree, ths, hs[k], hs[k+1], k)
        sol = DE.solve(prob)
        D[k,:,:,2] .= sol[end]
    end
    return D
end

function solve_slice!(D, model, tree, ths, h0, h1, k)  # kth slice from h0 to h1
    @unpack λa, λb, m = model
    active = Int[]
    for u in postwalk(tree) 
        isactive = !isroot(u) && ths[id(u)] <= h0 && h0 < ths[id(parent(u))] 
        if ths[id(u)] == h0 && !isleaf(u)  # time slice beginning where u splits
            v, w = children(u)
            D[k,id(u),1,1] = D[k-1,id(v),1,2]*D[k-1,id(w),1,2]*2λa  # pop A
            D[k,id(u),2,1] = D[k-1,id(v),2,2]*D[k-1,id(w),2,2]*2λb  # pop B
        elseif isactive && k > 1
            D[k,id(u),1,1] = D[k-1,id(u),1,2]
            D[k,id(u),2,1] = D[k-1,id(u),2,2]
        end
        isroot(u) && break
        isactive && push!(active, id(u))
    end
    init = D[k,:,:,1]
    prob = DE.ODEProblem(odesystem!, init, (h0, h1), 
        (model=model, ths=ths, active=active, tree=tree)) 
    return prob
end

function odesystem!(dx, x, problem, t)
    @unpack model, tree, ths, active = problem
    @unpack m, λa, λb = model
    for u in postwalk(tree)
        if id(u) ∉ active
            dx[id(u),:] .= 0.0
        else
            activea = sum(x[active,1]) - x[id(u),1]
            activeb = sum(x[active,2]) - x[id(u),2]
            dx[id(u),1] = -λa*activea*x[id(u),1] + m*x[id(u),2]
            dx[id(u),2] = -(m + λb*activeb)*x[id(u),2]
        end
    end
end


