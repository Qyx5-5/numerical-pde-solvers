function psi = tssp_step(psi, potential, k2, dt, nonlinear)
%TSSP_STEP One Strang split-step spectral update.

phase_half = exp(-0.5i * dt * (potential + nonlinear(psi)));
psi = phase_half .* psi;
psi_hat = fftn(psi);
psi = ifftn(exp(-0.5i * dt * k2) .* psi_hat);
phase_half = exp(-0.5i * dt * (potential + nonlinear(psi)));
psi = phase_half .* psi;
end
