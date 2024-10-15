module RobustWithLessShifts
using JuMP, LinearAlgebra
using MosekTools, SCS

export  Standard_LQR_SDP, Quadratically_Stable_LQR_SDP, 
        Quadratically_Stable_State_Data_Conforming_LQR_SDP, 
        Quadratically_Stable_State_Input_Data_Conforming_LQR_SDP

function Standard_LQR_SDP(A, B, Q, R, W, V)
    model = Model(Mosek.Optimizer);
    rₓ = size(A)[1]
    rᵤ = size(B)[2]
    @variable(model, Σ[i=1:rₓ, j=1:rₓ], PSD)
    @variable(model, Z₀[i=1:rᵤ, j=1:rᵤ], PSD)
    @variable(model, L[i=1:rᵤ, j=1:rₓ])
    @objective(model, Min, tr(Q*Σ) + tr(R*Z₀))
    @constraint(model, Σ-1e-5*I >= 0, PSDCone())
    @constraint(model, [Z₀ L; L' Σ] >= 0, PSDCone())
    # Lyupanov
    @constraint(model, [(Σ-W-B*V*B') A*Σ+B*L; (A*Σ+B*L)' Σ]>=0, PSDCone());

    set_silent(model)
    optimize!(model);
    
    
    L, Σ = value.(L), value.(Σ)
    K_LMI = L * inv(Σ)
    println("Lyupanov condition error: (L∞ norm) is ", maximum(abs.(Σ-W-B*V*B'-(A+B*K_LMI)*Σ*(A+B*K_LMI)')))
    return K_LMI
end

function Quadratically_Stable_LQR_SDP(ABs, Q, R, W, V)
    model = Model(Mosek.Optimizer);
    rₓ = size(Q)[1]
    rᵤ = size(R)[2]
    @variable(model, Σ[i=1:rₓ, j=1:rₓ], PSD)
    @variable(model, Z₀[i=1:rᵤ, j=1:rᵤ], PSD)
    @variable(model, L[i=1:rᵤ, j=1:rₓ])
    @objective(model, Min, tr(Q*Σ) + tr(R*Z₀))
    @constraint(model, Σ-1e-5*I >= 0, PSDCone())
    @constraint(model, [Z₀ L; L' Σ] >= 0, PSDCone())
    # Lyupanov
    for AB in ABs
        A, B = AB
        @constraint(model, [(Σ-W-B*V*B') A*Σ+B*L; (A*Σ+B*L)' Σ]>=0, PSDCone());
    end

    set_silent(model)
    optimize!(model);
    
    L, Σ = value.(L), value.(Σ)
    K_LMI = L * inv(Σ)
    return K_LMI
end

function Quadratically_Stable_State_Data_Conforming_LQR_SDP(ABs, Q, R, W, V, X, γ_prime)
    N = size(X)[2]
    Σ_data = 1/(N) * X * X'
    model = Model(Mosek.Optimizer);
    rₓ = size(Q)[1]
    rᵤ = size(R)[2]

    @variable(model, Σ[i=1:rₓ, j=1:rₓ], PSD)
    @variable(model, Z_F[i=1:rₓ, j=1:rₓ], PSD)
    @variable(model, Z₀[i=1:rᵤ, j=1:rᵤ], PSD)
    @variable(model, L[i=1:rᵤ, j=1:rₓ])

    @objective(model, Min, tr(Q*Σ) + tr(R*Z₀) + γ_prime * tr(Z_F))

    @constraint(model, Σ - (1e-6I) >= 0, PSDCone())
    @constraint(model, [Z₀ L; L' Σ] >= 0, PSDCone())
    # Lyupanov
    #@constraint(model, Σ - (Â*Σ*Â' + B̂*L*Â' + Â*L'*B̂' +  B̂*Z₀*B̂' + W + B̂*V*B̂') >= 0, PSDCone());
    #@constraint(model, Σ - (Â*Σ*Â' + B̂*L*Â' + Â*L'*B̂' +  B̂*Z₀*B̂' + W + B̂*V*B̂') <= 0, PSDCone());
    @constraint(model, [(Σ-W-B*V*B') A*Σ+B*L; (A*Σ+B*L)' Σ]>=0, PSDCone());
    @constraint(model, [Z_F (Σ-Σ_data); (Σ-Σ_data) I] >= 0, PSDCone());
    #set_attribute(model, "INTPNT_CO_TOL_DFEAS", 1e-10)
    set_silent(model)
    optimize!(model);

    L, Σ = value.(L), value.(Σ)
    K_star = L * inv(Σ)
    return K_star
end

function Quadratically_Stable_State_Input_Data_Conforming_LQR_SDP(ABs, Q, R, W, V, X, U, γ)
    N = size(X)[2]
    rₓ = size(Q)[1]
    rᵤ = size(R)[2]
    D = [X;U]
    Σ_data = 1/(N) * X * X'
    Γ_data = 1/(N) * D * D'
    H_data = Γ_data[1:rₓ,rₓ+1:end]
    Υ = cat(V, zeros(rₓ,rₓ); dims=(1,2))
    model = Model(Mosek.Optimizer);
    

    @variable(model, Σ[i=1:rₓ, j=1:rₓ], PSD)
    @variable(model, Z₀[i=1:rᵤ, j=1:rᵤ], PSD)
    @variable(model, L[i=1:rᵤ, j=1:rₓ])
    @variable(model, Z₁[i=1:rₓ+rᵤ, j=1:rₓ+rᵤ])
    @variable(model, Z₂[i=1:rᵤ, j=1:rᵤ])
    @variable(model, Z₃[i=1:rₓ, j=1:rₓ])

    @objective(model, Min, tr(Q*Σ) + tr(R*Z₀) 
    + γ * (tr(inv(Γ_data) * Z₁) + tr(inv(V) * Z₂) + tr(Σ_data * Z₃)))

    @constraint(model, Σ - (1e-6I) >= 0, PSDCone())
    @constraint(model, [Z₀ L; L' Σ] >= 0, PSDCone())
    # Lyupanov
    for AB in ABs
        A, B = AB
        @constraint(model, [(Σ-W-B*V*B') A*Σ+B*L; (A*Σ+B*L)' Σ]>=0, PSDCone());
    end
    # The Z's
    @constraint(model, [Z₃ I; I Σ] >= 0, PSDCone())
    @constraint(model, [Z₂ (L-H_data'*inv(Σ_data)*Σ); (L-H_data'*inv(Σ_data)*Σ)' Σ] >= 0, PSDCone())
    @constraint(model, [(Z₁-Υ) cat(Σ_data, L; dims=1); cat(Σ_data, L; dims=1)' Σ] >= 0, PSDCone())
    #set_attribute(model, "INTPNT_CO_TOL_DFEAS", 1e-10)
    set_silent(model)
    optimize!(model);

    L, Σ = value.(L), value.(Σ)
    K_star = L * inv(Σ)
    return K_star
end

end # module DataConformingControl