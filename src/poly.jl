# ============================================================
# Polynomial Methods
# ============================================================

legendre_point(p::T) where {T<:Integer} = gausslegendre(p + 1)[1]

∂legendre(p::T, x) where {T<:Integer} = last(sf_legendre_Pl_deriv_array(p, x)[2])

function ∂legendre(p::I, x::T) where {I<:Integer,T<:AbstractArray{<:Real,1}}
    Δ = similar(x)
    for i in eachindex(Δ)
        Δ[i] = ∂legendre(p, x[i])
    end

    return Δ
end

function ∂radau(p::TI, x::TU) where {TI<:Integer,TU<:Union{Real,AbstractArray{<:Real,1}}}
    Δ = ∂legendre(p, x)
    Δ_plus = ∂legendre(p+1, x)

    dgl = @. (-1.0)^p * 0.5 * (Δ - Δ_plus)
    dgr = @. 0.5 * (Δ + Δ_plus)

    return dgl, dgr
end

function lagrange_point(sp::T, x) where {T<:AbstractVector{<:Real}}
    l = similar(sp)
    nsp = length(sp)

    for k in 1:nsp
        tmp = 1.0
        for j in 1:nsp
            if j != k
                tmp *= (x - sp[j]) / (sp[k] - sp[j])
            end
        end
        l[k] = tmp
    end

    return l
end

function ∂lagrange(sp::T) where {T<:AbstractVector{<:Real}}
    nsp = length(sp)
    lpdm = similar(sp, nsp, nsp)

    for k = 1:nsp, m = 1:nsp
        lsum = 0.0
        for l in 1:nsp
            tmp = 1.0
            for j = 1:nsp
                if j != k && j != l
                    tmp *= (sp[m] - sp[j]) / (sp[k] - sp[j])
                end
            end
            if l != k
            lsum += tmp / (sp[k] - sp[l])
            end
        end
        lpdm[m, k] = lsum
    end

    return lpdm
end

function standard_lagrange(x)
    ll = lagrange_point(x, -1.0)
    lr = lagrange_point(x, 1.0)
    lpdm = ∂lagrange(x)

    return ll, lr, lpdm
end



"""
Evaluate 2D orthonormal polynomial on simplex at (a, b) of order (i, j)
Translated from Simplex2DP.m

"""
function Simplex2DP(a, b, i, j);

    
    h1 = JacobiP(a,0,0,i); # n, a, b, x
    h2 = JacobiP(b,2*i+1,0,j);
    
    h1 = jacobi(a, i, 0, 0) # x, n, a, b
    h2 = jacobi(b, j, 2*i+1, 0)
    
    P = sqrt(2.0)*h1.*h2.*(1-b).^i;
    return P
end


function JacobiP(x,alpha,beta,N)
  
    xp = copy(x); dims = size(xp);
    if length(dims)==1
        xp = xp'
    end
    
    PL = zeros(N+1, length(xp))
    
    gamma0 = 2^(alpha+beta+1)/(alpha+beta+1)*gamma(alpha+1)*gamma(beta+1)/gamma(alpha+beta+1);
    PL[1,:] .= 1.0 / sqrt(gamma0);

    if (N==0)
        P = PL'
        return P
    end

    gamma1 = (alpha+1)*(beta+1)/(alpha+beta+3)*gamma0;
    PL[2,:] .= ((alpha+beta+2)*xp/2 + (alpha-beta)/2)/sqrt(gamma1);
    if (N==1)
        P=PL[N+1,:]'
        return P
    end
    
    aold = 2/(2+alpha+beta)*sqrt((alpha+1)*(beta+1)/(alpha+beta+3));
    
    # forward recurrence using the symmetry of the recurrence.
    for i=1:N-1
      h1 = 2*i+alpha+beta;
      anew = 2/(h1+2)*sqrt( (i+1)*(i+1+alpha+beta)*(i+1+alpha)*
          (i+1+beta)/(h1+1)/(h1+3));
      bnew = - (alpha^2-beta^2)/h1/(h1+2);
      @. PL[i+2,:] = 1/anew*( -aold*PL[i,:] + (xp-bnew)*PL[i+1,:])
      aold =anew;
    end
    
    P = PL[N+1, :]'
    return P
end

function rs_ab(r, s)
    Np = length(r); a = zeros(Np);
    for n=1:Np
        if s[n] != 1
            a[n] = 2*(1+r[n])/(1-s[n])-1;
        else
            a[n] = -1;
        end
    end
    b = s;

    return a, b
end
