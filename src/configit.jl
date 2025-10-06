
# How to do this reasonably elegant?
struct Configurations
    na :: Int
    nb :: Int
    k  :: Int
end

struct ConfigState
    X :: NTuple{4,Int}
    minx :: Int
    maxx :: Int
    maxy :: Int
end

function Base.iterate(it::Configurations)
    (0, it.na, 0, it.nb),  
end

function Base.iterate(it::Configurations, state)
    @unpack na, nb = it
    @unpack X, minx, maxx, maxy = state
    (x1, x2, y1, y2) = X
    y1 += 1
    if y1 <= maxy
        X = (x1, x2, y1, y2)
        return X, ConfigState(X, minx, maxx, maxy)
    else
        y1 = 0 
        x2 += 1
        if x2 <= maxx
            maxy = min(n-k+1-x1-x2, nb)
            X = (x1, x2, y1, y2)
            return X, ConfigState(X, minx, maxx, maxy)
        else
            x2 =     
    end
end
