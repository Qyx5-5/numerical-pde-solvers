function [psi, results] = split_step_solver(psi0, potential, k2, dt, steps, save_every, nonlinear)
%SPLIT_STEP_SOLVER Generic time-splitting spectral evolution.
% nonlinear is a function handle nonlinear(psi) returning the nonlinear potential.

if nargin < 7
    nonlinear = @(psi) zeros(size(psi));
end

psi = psi0;
num_saves = floor(steps / save_every) + 1;
results.time = zeros(1, num_saves);
results.density = cell(1, num_saves);
results.density{1} = abs(psi).^2;
save_idx = 2;

for n = 1:steps
    psi = tssp_step(psi, potential, k2, dt, nonlinear);
    if mod(n, save_every) == 0
        results.time(save_idx) = n * dt;
        results.density{save_idx} = abs(psi).^2;
        save_idx = save_idx + 1;
    end
end
end
