
struct BiModel{T}
    λa  :: T
    λb  :: T
    mab :: T  # A -> B backward in time
    mba :: T  # B -> A backward in time
    na  :: Int
    nb  :: Int
end

function randtree(rng, model::BiModel)
    @unpack na, nb = model
    nodes = [
        [Node(i, n="A", d=0.0) for i=1:na], 
        [Node(i, n="B", d=0.0) for i=na+1:na+nb]]
    k = na+nb
    hs = Dict(i=>0.0 for i=1:na+nb)
    tt = 0.0
    ℓ  = 0.0
    function _randtree(rng, model, nodes)
        @unpack λa, λb, mab, mba = model
        na = length(nodes[1])
        nb = length(nodes[2])
        na + nb == 1 && return nodes
        rates = [λa*na*(na-1)/2, λb*nb*(nb-1)/2, na*mab, nb*mba]
        total = sum(rates)
        t = rand(rng, Exponential(1/total))
        ℓ += logpdf(Exponential(1/total), t)
        tt += t
        event = sample(rng, 1:4, Weights(rates))
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
            (n, src, tgt) = event == 3 ? (na, 1, 2) : (nb, 2, 1)
            ℓ += log(1/n)
            i = sample(rng, 1:n)
            push!(nodes[tgt], nodes[src][i])
            deleteat!(nodes[src], i)
        end
        _randtree(rng, model, nodes)
    end
    vcat(_randtree(rng, model, nodes)...)[1], ℓ
end

function _labels(a, b)
    eventlabel = if a != b && a == "AB" 
        "$b|$a"
    elseif a != b && b == "AB"
        "$a|$b"
    elseif a != b
        "A|B"
    else
        "$a|$b"
    end
    nodelabel = if a == b 
        a
    elseif a != b && a == "AB"
        b
    elseif a != b && b == "AB"
        a
    else
        "AB"
    end
    return eventlabel, nodelabel
end

# Obtain time slices and associated labelings
function getslices2(tree::Node{T,V}) where {T,V}
    labels = Dict{T,Tuple{Float64,String}}()
    slices = Tuple{Float64,String}[]
    na = nb = 0
    for u in postwalk(tree)
        if isleaf(u)
            labels[id(u)] = (0.0, name(u))
            name(u) == "A" ? (na += 1) : (nb += 1)
        else
            v, w = children(u)
            h, lv = labels[id(v)]
            _, lw = labels[id(w)]
            el, nl = _labels(lv, lw)
            hu = h + distance(v)
            labels[id(u)] = (hu, nl)
            push!(slices, (hu, el))
        end
    end
    sort!(slices)
    return Slices(first.(slices), last.(slices))
end

function lhood_discretized(model::BiModel, slices, dt; kwargs...) 
    P, P_ = solve_discretized(model, slices, dt; kwargs...) 
    p = logsumexp(P)
end

function solve_discretized(model::BiModel{T}, ss::Slices, dt; 
        nmin=10, kmax=length(ss)) where T
    @unpack slices, labels = ss
    @unpack na, nb = model
    # initial state
    P = fill(-Inf, na+2, na+2, nb+2, nb+2)
    P[na+1, 1, nb+1, 1] = 0.0 
    for k=1:kmax
        t0 = k == 1 ? 0.0 : slices[k-1]
        P_, dt_ = solve_slice_discretized(
            model, P, t0, slices[k], k, dt, nmin) 
        P  = initial_state_(P_, labels[k], k+1, model, dt=dt_) 
        k == kmax && return P, P_
    end
end

function solve_slice_discretized(model, P, t0, t1, k, dt, nmin)
    @unpack mab, mba, λa, λb, na, nb = model
    n = na + nb
    maxx = min(n-k+1, na)
    nstep = floor(Int, (t1 - t0) ÷ dt)
    if nstep < nmin
        dt = (t1 - t0)/nmin
        nstep = nmin-1
    end
    laststep = (t1 - t0) - nstep*dt
    P_ = copy(P)
    for i=1:nstep
        for x1=0:maxx
            # x1 + x2 >= na - k + 1
            minx2 = max(0, na-k+1-x1)
            for x2=minx2:maxx-x1
                maxy = min(n-k+1-x1-x2, nb)
                for y1=0:maxy
                    y2 = n-k+1-x1-x2-y1
                    # x1 : A lineages in A
                    # x2 : A lineages in B
                    # y1 : B lineages in B
                    # y2 : B lineages in A
                    xa = x1 + y2  # lineages in A
                    xb = x2 + y1  # lineages in B
                    r = 1 - (
                            λa*dt*xa*(xa-1)/2 + 
                            λb*dt*xb*(xb-1)/2 + 
                            mab*dt*xa + mba*dt*xb)
                    if r < 0.0 
                        @error "rates too large"
                        r = 0.0
                    end
                    p = P[x1+1,x2+1,y1+1,y2+1] + log(r)
                    if x1 > 0 && x2 < na
                        p = logaddexp(p, P[x1,x2+2,y1+1,y2+1] + log((x2+1)*mba*dt))
                    end
                    if x1 < na && x2 > 0
                        p = logaddexp(p, P[x1+2,x2,y1+1,y2+1] + log((x1+1)*mab*dt))
                    end
                    if y1 > 0 && y2 < nb
                        p = logaddexp(p, P[x1+1,x2+1,y1,y2+2] + log((y2+1)*mab*dt))
                    end
                    if y1 < nb && y2 > 0
                        p = logaddexp(p, P[x1+1,x2+1,y1+2,y2] + log((y1+1)*mba*dt))
                    end
                    P_[x1+1,x2+1,y1+1,y2+1] = p
                end
            end
        end
        P = copy(P_)
    end
    return P, laststep
end

function initial_state_(P::Array{T,4}, label, k, model::BiModel{T};
        dt=1.0) where T
    @unpack λa, λb, na, nb = model
    λa *= dt
    λb *= dt
    _P = fill(-Inf, size(P))
    n = na + nb
    maxx = min(n-k+1, na)
    for x1=0:maxx
        # x1 + x2 >= na - k + 1
        minx2 = max(0, na-k+1-x1)
        for x2=minx2:maxx-x1
            maxy = min(n-k+1-x1-x2, nb)
            for y1=0:maxy
                y2 = n-k+1-x1-x2-y1
                # XXX Assumes P is oversized by one in each dimension, with
                # a zero at na+2 in the first two and nb+2 in the last two
                # dimensions
                if label == "A|A" || label == "A|AB" || label == "AB|AB"
                    # coalescence of 11 or of 22
                    _P[x1+1,x2+1,y1+1,y2+1] = logaddexp(
                        _P[x1+1,x2+1,y1+1,y2+1], 
                        P[x1+2,x2+1,y1+1,y2+1] + log(λa*x1*(x1+1)/2))
                    _P[x1+1,x2+1,y1+1,y2+1] = logaddexp( 
                         _P[x1+1,x2+1,y1+1,y2+1], 
                         P[x1+1,x2+2,y1+1,y2+1] + log(λb*x2*(x2+1)/2))
                end
                if label == "B|B" || label == "B|AB" || label == "AB|AB"
                    # coal of 33 or 44
                    _P[x1+1,x2+1,y1+1,y2+1] = logaddexp( 
                        _P[x1+1,x2+1,y1+1,y2+1],
                        P[x1+1,x2+1,y1+2,y2+1] + log(λb*y1*(y1+1)/2))
                    _P[x1+1,x2+1,y1+1,y2+1] = logaddexp(
                        _P[x1+1,x2+1,y1+1,y2+1],
                        P[x1+1,x2+1,y1+1,y2+2] + log(λa*y2*(y2+1)/2))
                end
                if label == "A|B" || label == "A|AB" || label == "B|AB"
                    # coal of 14->1 or 23->3
                    _P[x1+1,x2+1,y1+1,y2+1] = logaddexp( 
                        _P[x1+1,x2+1,y1+1,y2+1],
                        P[x1+1,x2+1,y1+1,y2+2] + log(λa*(y2+1)*x1/2))
                    _P[x1+1,x2+1,y1+1,y2+1] = logaddexp( 
                        _P[x1+1,x2+1,y1+1,y2+1],
                        P[x1+1,x2+2,y1+1,y2+1] + log(λb*(x2+1)*y1/2))
                end
                #elseif label == "A|AB"
                #    # coal of 11 or 14 or 23 or 22
                #elseif label == "B|AB"
                #    # coal of 33 or 23 or 14 or 44
                if label == "AB|AB"
                    # coal of 22 or 23 or 32 or 33
                    #      or 11 or 14 or 41 or 44
                    _P[x1+1,x2+1,y1+1,y2+1] = logaddexp( 
                        _P[x1+1,x2+1,y1+1,y2+1],
                        2P[x1+1,x2+1,y1+1,y2+2] + log(λa*(y2+1)*x1/2))
                    _P[x1+1,x2+1,y1+1,y2+1] = logaddexp(
                        _P[x1+1,x2+1,y1+1,y2+1],
                        2P[x1+1,x2+2,y1+1,y2+1] + log(λb*(x2+1)*y1/2))
                end
            end
        end
    end
    return _P
end

function lhood(model::BiModel, slices; kwargs...) 
    p = solve(model, slices; kwargs...) |> sum
    p <= 0.0 ? -Inf : log(p) 
end

function solve(model::BiModel{T}, ss::Slices; kwargs...) where T
    @unpack na, nb, slices, labels = ss
    P = zeros(T, na+2, na+2, nb+2, nb+2)
    P[na+1,1,nb+1,1] = 1.0
    for k=1:3#length(ss)
        t0 = k == 1 ? 0.0 : slices[k-1]
        P_ = solve_slice(model, P, t0, slices[k], k; kwargs...)
        P = initial_state(P_, labels[k], k+1, model, ss)
    end
    return P
end

function solve_slice(model::BiModel, P, h0, h1, k; kwargs...)
    na, _, nb, _ = size(P)
    na -= 2; nb -= 2 
    prob = DE.ODEProblem(odesystem_bi!, P, (h0, h1), 
        (model=model, n=na+nb, na=na, nb=nb, k=k)) 
    DE.solve(prob, DE.Tsit5(); 
        save_everystep=false, kwargs...)[end]
end

function odesystem_bi!(dP, P, problem, t)
    @unpack n, k, na, nb, model = problem
    @unpack λa, λb, mab, mba = model
    maxx = min(n-k+1, na)
    for x1=0:maxx
        # x1 + x2 >= na - k + 1
        minx2 = max(0, na-k+1-x1)
        for x2=minx2:maxx-x1
            maxy = min(n-k+1-x1-x2, nb)
            for y1=0:maxy
                y2 = n-k+1-x1-x2-y1
                # x1 : A lineages in A
                # x2 : A lineages in B
                # y1 : B lineages in B
                # y2 : B lineages in A
                xa = x1 + y2  # lineages in A
                xb = x2 + y1  # lineages in B
                p = -P[x1+1,x2+1,y1+1,y2+1]*(
                        λa*xa*(xa-1)/2 + λb*xb*(xb-1)/2 + mab*xa + mba*xb)
                if x1 > 0 && x2 < na
                    p += P[x1,x2+2,y1+1,y2+1]*(x2+1)*mba
                end
                if x1 < na && x2 > 0
                    p += P[x1+2,x2,y1+1,y2+1]*(x1+1)*mab
                end
                if y1 > 0 && y2 < nb
                    p += P[x1+1,x2+1,y1,y2+2]*(y2+1)*mab
                end
                if y1 < nb && y2 > 0
                    p += P[x1+1,x2+1,y1+2,y2]*(y1+2)*mba
                end
                dP[x1+1,x2+1,y1+1,y2+1] = p
            end
        end
    end
end

# complicated!
function initial_state(P::Array{T,4}, label, k, model::BiModel{T};
        dt=1.0) where T
    @unpack λa, λb, na, nb = model
    λa *= dt
    λb *= dt
    _P = zeros(T, size(P))
    n = na + nb
    maxx = min(n-k+1, na)
    for x1=0:maxx
        # x1 + x2 >= na - k + 1
        minx2 = max(0, na-k+1-x1)
        for x2=minx2:maxx-x1
            maxy = min(n-k+1-x1-x2, nb)
            for y1=0:maxy
                y2 = n-k+1-x1-x2-y1
                # XXX Assumes P is oversized by one in each dimension, with
                # a zero at na+2 in the first two and nb+2 in the last two
                # dimensions
                if label == "A|A" || label == "A|AB" || label == "AB|AB"
                    # coalescence of 11 or of 22
                    _P[x1+1,x2+1,y1+1,y2+1] += 
                        P[x1+2,x2+1,y1+1,y2+1] * λa*x1*(x1+1)/2
                    _P[x1+1,x2+1,y1+1,y2+1] += 
                        P[x1+1,x2+2,y1+1,y2+1] * λb*x2*(x2+1)/2
                end
                if label == "B|B" || label == "B|AB" || label == "AB|AB"
                    # coal of 33 or 44
                    _P[x1+1,x2+1,y1+1,y2+1] += 
                        P[x1+1,x2+1,y1+2,y2+1] * λb*y1*(y1+1)/2
                    _P[x1+1,x2+1,y1+1,y2+1] += 
                        P[x1+1,x2+1,y1+1,y2+2] * λa*y2*(y2+1)/2
                end
                if label == "A|B" || label == "A|AB" || label == "B|AB"
                    # coal of 14->1 or 23->3
                    _P[x1+1,x2+1,y1+1,y2+1] += 
                        P[x1+1,x2+1,y1+1,y2+2] * λa*(y2+1)*x1/2
                    _P[x1+1,x2+1,y1+1,y2+1] += 
                        P[x1+1,x2+2,y1+1,y2+1] * λb*(x2+1)*y1/2
                end
                #elseif label == "A|AB"
                #    # coal of 11 or 14 or 23 or 22
                #elseif label == "B|AB"
                #    # coal of 33 or 23 or 14 or 44
                if label == "AB|AB"
                    # coal of 22 or 23 or 32 or 33
                    #      or 11 or 14 or 41 or 44
                    _P[x1+1,x2+1,y1+1,y2+1] += 
                        2P[x1+1,x2+1,y1+1,y2+2] * λa*(y2+1)*x1/2
                    _P[x1+1,x2+1,y1+1,y2+1] += 
                        2P[x1+1,x2+2,y1+1,y2+1] * λb*(x2+1)*y1/2
                end
            end
        end
    end
    return _P
end

function constraints(na, nb, k)
    xx = NTuple{4,Int}[]
    n = na + nb
    maxx = min(n-k+1, na)
    for x1=0:maxx
        # x1 + x2 >= na - k + 1
        minx2 = max(0, na-k+1-x1)
        for x2=minx2:maxx-x1
            maxy = min(n-k+1-x1-x2, nb)
            for y1=0:maxy
                y2 = n-k+1-x1-x2-y1
                push!(xx, (x1, x2, y1, y2))
            end
        end
    end
    return xx
end

