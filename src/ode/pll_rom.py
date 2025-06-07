import torch

def pll_rom(t, x, params):
    """
    Reduced‐order PLL (2 states: δ and ω).

    Inputs:
      t      (torch.Tensor): time placeholder, shape (batch,1).
      x      (torch.Tensor): state tensor, shape (batch,2), [δ, ω].
      params (dict of torch.Tensor): {k_p, k_i, L_g, r_Lg, i_d_c, i_q_c, v_g, omega_g}.

    Outputs:
      dxdt (torch.Tensor): shape (batch,2), [dδ/dt, dω/dt].
    """
    δ = x[:, 0:1]      # δ has shape (batch,1)
    ω = x[:, 1:2]      # ω has shape (batch,1)

    k_p   = params['k_p']
    k_i   = params['k_i']
    L_g   = params['L_g']
    r_Lg  = params['r_Lg']
    i_d_c = params['i_d_c']
    i_q_c = params['i_q_c']
    v_g   = params['v_g']
    ω_g   = params['omega_g']

    M   = 1.0 - k_p * L_g * i_d_c
    T_m = k_i * (r_Lg * i_q_c + L_g * i_d_c * ω_g)

    T_e = k_i * v_g * torch.sin(δ)
    D   = k_p * v_g * torch.cos(δ) - k_i * L_g * i_d_c

    ddelta_dt = ω
    domega_dt = (T_m - T_e - D * ω) / M

    dxdt = torch.cat([ddelta_dt, domega_dt], dim=1)
    return dxdt