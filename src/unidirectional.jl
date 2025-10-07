
struct UniModel{T}
    λa :: T
    λb :: T
    m  :: T
    na :: Int
    nb :: Int
end

# XXX redundant with BiModel of course.
function randtree(rng, model::UniModel)
    @unpack na, nb = model
    nodes = [
        [Node(i, n="A", d=0.0) for i=1:na], 
        [Node(i, n="B", d=0.0) for i=na+1:na+nb]]
    k = na+nb
    hs = Dict(i=>0.0 for i=1:na+nb)
    tt = 0.0
    ℓ  = 0.0
    function _randtree(rng, model, nodes)
        @unpack λa, λb, m = model
        na = length(nodes[1])
        nb = length(nodes[2])
        na + nb == 1 && return nodes
        rates = [λa*na*(na-1)/2, λb*nb*(nb-1)/2, nb*m]
        total = sum(rates)
        t = rand(rng, Exponential(1/total))
        ℓ += logpdf(Exponential(1/total), t)
        tt += t
        event = sample(rng, 1:3, Weights(rates))
        ℓ += log(rates[event]/total)
        if event == 1 || event == 2  # coalescence
            (n, h) = event == 1 ? (na, 1) : (nb, 2)
            ℓ += log(2/n*(n-1))
            k += 1
            hs[k] = tt
            i, j = sample(rng, 1:n, 2, replace=false)
            nk = Node(k, d=0.0)
            ni = nodes[h][i]
            nj = nodes[h][j]
            NewickTree.setdistance!(ni, tt-hs[id(ni)])    
            NewickTree.setdistance!(nj, tt-hs[id(nj)])    
            push!(nk, ni)
            push!(nk, nj)
            deleteat!(nodes[h], sort([i,j]))
            push!(nodes[h], nk)
        else  # migration
            ℓ += log(1/nb)
            i = sample(rng, 1:nb)
            push!(nodes[1], nodes[2][i])
            deleteat!(nodes[2], i)
        end
        _randtree(rng, model, nodes)
    end
    _randtree(rng, model, nodes)[1][1], ℓ
end


function _parentlabel(a, b)
    a == b == "AB" && return "A"
    a == b && return a
    (a == "A" && b == "B") && return "AB"
    (a == "B" && b == "A") && return "AB"
    return min(a, b)
end

struct Slices{T,V}
    slices :: Vector{T}
    labels :: Vector{V}
end

Base.length(ss::Slices) = length(ss.slices)

# Obtain time slices and associated labelings
function getslices(tree::Node{T,V}) where {T,V}
    labels = Dict{T,Tuple{Float64,String}}()
    slices = Tuple{Float64,String}[]
    for u in postwalk(tree)
        if isleaf(u)
            labels[id(u)] = (0.0, name(u))
        else
            v, w = children(u)
            h, lv = labels[id(v)]
            _, lw = labels[id(w)]
            lu = _parentlabel(lv, lw)
            hu = h + distance(v)
            labels[id(u)] = (hu, lu)
            push!(slices, (hu, lu))
        end
    end
    sort!(slices)
    return Slices(first.(slices), last.(slices))
end

function lhood(model::UniModel, slices; kwargs...) 
    P, P_ = solve(model, slices; kwargs...)
    p = P[2,1,1]
    p <= 0.0 ? -Inf : log(p) 
end

function lhood_discretized(model::UniModel, slices, dt; kwargs...) 
    P, P_ = solve_discretized(model, slices, dt; kwargs...)
    p = P[2,1,1]
    p <= 0.0 ? -Inf : log(p) 
end

function solve(model::UniModel{T}, ss::Slices; kmax=length(ss), kwargs...) where T
    @unpack slices, labels = ss
    @unpack na, nb = model
    # initial state
    P = zeros(T, na+1, nb+1, nb+1)
    # XXX the P matrix is oversized, since the first dimension (# lineages
    # in A) cannote be zero... In general, due to the constraints on the
    # sums across dimensions, a 3D array is not an ideal representation,
    # but is transparent, i.e. P[x+1,y+1,z+1] = p(x,y,z) in our math.
    # notation.
    P[na+1, 1, nb+1] = 1.0 
    for k=1:kmax
        t0 = k == 1 ? 0.0 : slices[k-1]
        P_ = solve_slice(model, P, t0, slices[k], k; kwargs...) 
        P  = initial_state(P_, labels[k], k+1, model) 
        k == kmax && return P, P_
    end
    #return P
end

function initial_state(P::Array{T,3}, label, k, model; dt=1.0) where T
    @unpack na, nb, λa, λb = model
    λa *= dt
    λb *= dt
    _P = zeros(T, size(P))
    n = na + nb
    maxx = min(n-k+1, na)
    minx = max(1, na-k+1)
    for x=minx:maxx
        maxy = min(n-k+1-x,nb)
        for y=0:maxy
            z = n - k + 1 - x - y
            if label == "A" && x < na
                _P[x+1,y+1,z+1] = P[x+2,y+1,z+1]*λa*x*(x+1)/2
                # if x == na (maxx?), it is not possible that we are
                # dealing with a coalescence 
            elseif label == "B"
                y < nb && (_P[x+1,y+1,z+1]  = P[x+1,y+2,z+1]*λa*y*(y+1)/2)
                z < nb && (_P[x+1,y+1,z+1] += P[x+1,y+1,z+2]*λb*z*(z+1)/2)
            elseif label == "AB" && y < nb  # AB
                _P[x+1,y+1,z+1] = P[x+1,y+2,z+1]*λa*x*(y+1)
            end
        end
    end
    return _P
end

function solve_slice(model, P, h0, h1, k; kwargs...)
    na, nb, _ = size(P)
    na -= 1; nb -= 1 
    prob = DE.ODEProblem(odesystem!, P, (h0, h1), 
        (model=model, n=na+nb, na=na, nb=nb, k=k)) 
    DE.solve(prob, DE.Tsit5(); 
        save_everystep=false, kwargs...)[end]
end

function odesystem!(dP, P, problem, t)
    @unpack model, n, na, nb, k = problem
    @unpack m, λa, λb = model
    maxx = min(n-k+1, na)
    minx = max(1, na-k+1)
    for x=minx:maxx
        maxy = min(n-k+1-x,nb)
        for y=0:maxy
            z = n - k + 1 - x - y
            dP[x+1,y+1,z+1] = -((x*(x-1)/2 + y*(y-1)/2 + x*y)*λa + 
                λb*z*(z-1)/2 + z*m)*P[x+1,y+1,z+1] + (
                    (z >= nb || y == 0) ? 0.0 : (z+1)*m*P[x+1,y,z+2])
        end
    end
end

function solve_discretized(model::UniModel{T}, ss::Slices, dt; 
        nmin=10, kmax=length(ss)) where T
    @unpack slices, labels = ss
    @unpack na, nb = model
    # initial state
    P = zeros(T, na+1, nb+1, nb+1)
    P[na+1, 1, nb+1] = 1.0 
    for k=1:kmax
        t0 = k == 1 ? 0.0 : slices[k-1]
        P_, dt_ = solve_slice_discretized(model, P, t0, slices[k], k, dt, nmin) 
        P  = initial_state(P_, labels[k], k+1, model, dt=dt_) 
        k == kmax && return P, P_
    end
end

function solve_slice_discretized(model::UniModel, P, t0, t1, k, dt, nmin)
    @unpack m, λa, λb, na, nb = model
    n = na + nb
    maxx = min(n-k+1, na)
    minx = max(1, na-k+1)
    nstep = floor(Int, (t1 - t0) ÷ dt)
    if nstep < nmin
        dt = (t1 - t0)/nmin
        nstep = nmin-1
    end
    laststep = (t1 - t0) - nstep*dt
    P_ = copy(P)
    for i=1:nstep
        for x=minx:maxx
            maxy = min(n-k+1-x,nb)
            for y=0:maxy
                z = n - k + 1 - x - y
                P_[x+1,y+1,z+1] = (1 - ((x*(x-1)/2 + y*(y-1)/2 + x*y)*λa*dt + 
                    λb*dt*z*(z-1)/2 + z*m*dt))*P[x+1,y+1,z+1] + (
                        (z >= nb || y == 0) ? 0.0 : (z+1)*m*dt*P[x+1,y,z+2])
            end
        end
        P = copy(P_)
    end
    return P, laststep
end

function constraints(model::UniModel, k)
    @unpack na, nb = model
    n = na+nb
    maxx = min(n-k+1, na)
    minx = max(1, na-k+1)
    xx = NTuple{3,Int}[]
    for x=minx:maxx
        maxy = min(n-k+1-x,nb)
        for y=0:maxy
            z = n - k + 1 - x - y
            push!(xx, (x, y, z))
        end
    end
    return xx
end


