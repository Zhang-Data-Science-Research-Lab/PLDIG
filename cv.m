function cv_err = cv(nfold, alpha, lambda, nodetype, m, p, K, n, ...
    phi, filepath, tol, alg, printevery, add)
% a function to do nfold cross validation
% phi is an 1*K cell array of sufficient statitics. phi is unstanardized
% and permuted. 
% alpha and lambda are two sequences
% n is a K-dimensional vector
cv_err = zeros(length(alpha), length(lambda), nfold); 
for fold = 1:nfold
    % construct training data and validation data
    t_phi = cell(1, K); 
    v_phi = cell(1, K); 
    t_n = zeros(1, K); 
    v_n = zeros(1, K); 
    for k = 1:K
        a = repmat(ceil(n(k)/nfold), 1, rem(n(k), nfold)); 
        a = [a, repmat(floor(n(k)/nfold), 1, nfold-rem(n(k), nfold))]; 
        lower = [1, cumsum(a)+1];
        upper = cumsum(a); 
        t_phi{k} = phi{k}([1:(lower(fold)-1), (upper(fold)+1):n(k)],:); 
        v_phi{k} = phi{k}(lower(fold):upper(fold),:); 
        t_n(k) = size(t_phi{k}, 1); 
        v_n(k) = size(v_phi{k}, 1); 
    end
    
    
    % cv error
    for i = 1:length(alpha)
        % initialize theta and Theta
        ini_theta = zeros(K, sum(m));
        for r = 1:p
            if nodetype(r) == 'e'
                index = getindex(m, r);
                ini_theta(:,index) = -0.5;
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
        for r = 1:p
            if nodetype(r) == 'g'
                index = getindex(m, r);
                ini_Theta(:,index,index) = -1; 
            end
        end
        for j = 1:length(lambda)
            lambda1 = (1-alpha(i))*lambda(j);
            lambda2 = alpha(i)*lambda(j);
            filename = [filepath, '/out.mat'];
            [xopt, xout, theta_opt, Theta_opt, mu, sigma] = pseudo_DIG(m, p, K, ...
                nodetype, t_n, lambda1, lambda2, t_phi, ...
                filename, tol, alg, printevery, add, ini_theta, ini_Theta); 
            x = par2vec(m, p, K, nodetype, theta_opt, Theta_opt); 
            % standardize v_phi
            v_phi_std = v_phi;
            for r = 1:sum(m)
                for k = 1:K
                    v_phi_std{k}(:,r) = (v_phi{k}(:,r) - mu(r))./sigma(r); 
                end
            end
            cv_err(i,j,fold) = mylkh(m, v_n, p, K, nodetype, v_phi_std, mu, sigma, add, x); 
            ini_theta = theta_opt; 
            ini_Theta = Theta_opt; % warm start solution path
        end
    end
end

filename = [filepath, '/cv.mat']; 
save(filename, 'cv_err'); 



