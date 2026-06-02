%RUN_GPE_EXAMPLE Compact 1D Gross-Pitaevskii split-step example.

addpath(fullfile('..', '..', 'tssp'));
config = jsondecode(fileread('default_config.json'));

n = config.grid_points;
L = config.length;
x = linspace(-L / 2, L / 2, n + 1);
x = x(1:end-1);
dx = x(2) - x(1);
k = (2 * pi / L) * [0:n/2-1 -n/2:-1];
k2 = k.^2;

potential = 0.5 * config.trap_strength * x.^2;
psi0 = exp(-0.5 * x.^2);
psi0 = psi0 / sqrt(sum(abs(psi0).^2) * dx);
nonlinear = @(psi) config.g * abs(psi).^2;

[psi, results] = split_step_solver(psi0, potential, k2, config.dt, config.steps, config.save_every, nonlinear);
fprintf('final norm: %.8f\n', sum(abs(psi).^2) * dx);

rmpath(fullfile('..', '..', 'tssp'));
