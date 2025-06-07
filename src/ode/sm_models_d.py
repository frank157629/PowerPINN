import torch

def pll_rom_ode(t, x, params):
    """
    x[:,0] = δ, x[:,1] = ω,  params is a dict containing k_p, k_i, L_g, ...
    """
    # unpack state
    δ = x[0]
    ω = x[1]

    # unpack parameters
    k_p   = params['k_p']
    k_i   = params['k_i']
    L_g   = params['L_g']
    r_Lg  = params['r_Lg']
    i_d_c = params['i_d_c']
    i_q_c = params['i_q_c']
    v_g   = params['v_g']
    ω_g   = params['omega_g']

    # compute constant M, T_m
    M   = 1.0 - k_p * L_g * i_d_c
    T_m = k_i * (r_Lg * i_q_c + L_g * i_d_c * ω_g)

    # compute T_e and D
    T_e = k_i * v_g * np.sin(δ)
    D   = k_p * v_g * np.cos(δ) - k_i * L_g * i_d_c

    # ODE right-hand sides
    ddelta_dt = ω
    domega_dt = (T_m - T_e - D * ω) / M

    return [ddelta_dt, domega_dt]