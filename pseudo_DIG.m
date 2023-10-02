function [xopt, xout, theta_opt, Theta_opt, mu, sigma] = pseudo_DIG(m, p, K, ...
    nodetype, n, lambda1, lambda2, phi, filename, ...
    tol, alg, printevery, add, ini_theta, ini_Theta)
% m: p-dimensional vector, dimensions of sufficient statistics
% p: number of nodes per graph
% K: number of classes
% algorithm: unaccelarated proximal gradient descent 'GRA' or accelerated 
%            'AT', 'LLM', 'N07', 'TS'
% phi: input sufficient statistics (unstandardized). 1*K cell array, each
%      cell is n_k*sum(m) matrix
% nodetype: p-dimensional vector. 'd' for categorical, 'e' for exponential,
%           'g' for gaussian, 'p' for poisson, 't' for truncated poisson, 
%           'n' for negative binomial
% add: p-dimensional vector, additional arguments for some types of nodes, 
%      truncation points for 't', over-dispersion parameters for 'n'
% n: K-dimensional vector, per-class sample size
% tol, alg, printevery: tfocs control
% ini_theta, ini_Theta: initialization for warm start

%% default initial values if not pre-assigned
if nargin == 13
    ini_theta = zeros(K, sum(m));
    for i = 1:p
        if nodetype(i) == 'e'
            index = getindex(m, i);
            ini_theta(:,index) = -1.5;
        end
        if nodetype(i) == 't'
            index = getindex(m, i);
            ini_theta(:,index) = 1;
        end
        if nodetype(i) == 'n'
            index = getindex(m, i);
            ini_theta(:,index) = -30;
        end
    end
    ini_Theta = zeros(K, sum(m), sum(m)); 
    for i = 1:p
        if nodetype(i) == 'g'
            index = getindex(m, i);
            ini_Theta(:,index,index) = -1; 
        end
    end
end

ini = par2vec(m, p, K, nodetype, ini_theta, ini_Theta);

%% standardize
c1 = clock; 
mu_k = zeros(K, sum(m));
sigma2_k = zeros(K, sum(m)); 
for k = 1:K
    for i = 1:sum(m)
        mu_k(k,i) = mean(phi{k}(:,i)); 
        sigma2_k(k,i) = var(phi{k}(:,i));
    end
end
N = sum(n);
mu = n*mu_k/N; 
sigma2 = n*sigma2_k/N;
sigma = sqrt(sigma2); 

for k = 1:K
    for i = 1:sum(m)
        phi{k}(:,i) = (phi{k}(:,i) - mu(i))./sigma(i);
    end
end


%% tfocs
pseudo_lkh = @(x)mylkh(m, n, p, K, nodetype, phi, mu, sigma, add, x);
group_lasso = @(varargin)myprox(m, p, K, nodetype, lambda1, lambda2, n, varargin{:}); 
opts.alg = alg;  % unaccelarated proximal gradient descent, 'GRA'or 'AT', 'LLM', 'N07', 'TS'
opts.maxIts = 9999; 
opts.printEvery = printevery; % print result every 10 iterations
opts.tol = tol;  % tolerance 1e-4 or 1e-5

[xopt, xout] = tfocs(pseudo_lkh, {}, group_lasso, ini, opts);
[theta_opt, Theta_opt] = vec2par(xopt, p, m, K, nodetype); 
c2 = clock; 
time = etime(c2, c1); 
save(filename, 'xopt', 'xout', 'theta_opt', 'Theta_opt', 'mu', 'sigma', 'time'); 



