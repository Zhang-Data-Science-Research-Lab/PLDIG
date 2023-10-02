function [g, prox] = myprox(m, p, K, nodetype, lambda1, lambda2, n, x, t)
% The proximal operator of hierarchical group lasso
% w is a p*p weight matrix
% In order to properly determine which computation TFOCS is requesting, it 
% is necessary to test both nargin (the number of input arguments) and 
% nargout (the number of output arguments). 

[theta, Theta] = vec2par(x, p, m, K, nodetype); 
w = sqrt(m'*m)/sqrt(K); 
N = sum(n);
    
if nargin == 9
    % Calculate the penalty function value
    % Proximal operator
    u = theta;
    U = Theta;
    for r = 1:(p-1)
        [r_lower, r_upper] = getindex(m, r);
        for s = (r+1):p
            [s_lower, s_upper] = getindex(m, s);
            bigsoft = double(0);
            for k = 1:K
                par = Theta(k,r_lower:r_upper,s_lower:s_upper);
                eta = w(r,s)*sqrt(K)*n(k)/N; 
                bigsoft = bigsoft + soft(norm(par(:), 2), lambda1*eta*t)^2;
            end
            if sqrt(bigsoft) <= lambda2*w(r,s)*t
                U(:,r_lower:r_upper,s_lower:s_upper) = 0;
                U(:,s_lower:s_upper,r_lower:r_upper) = 0;
            else
                for k = 1:K
                    par = Theta(k,r_lower:r_upper,s_lower:s_upper);
                    eta = w(r,s)*sqrt(K)*n(k)/N;
                    a = soft(norm(par(:), 2), lambda1*eta*t);
                    if a == 0
                        U(k,r_lower:r_upper,s_lower:s_upper) = 0;
                        U(k,s_lower:s_upper,r_lower:r_upper) = 0;
                    else
                        b = 1 - lambda2*w(r,s)*t/sqrt(bigsoft);
                        c = 1 / norm(par(:), 2);
                        U(k,r_lower:r_upper,s_lower:s_upper) = a*b*c*...
                            Theta(k,r_lower:r_upper,s_lower:s_upper);
                        U(k,s_lower:s_upper,r_lower:r_upper) = a*b*c*...
                            Theta(k,s_lower:s_upper,r_lower:r_upper);
                    end   
                end
            end
        end
    end
    prox = par2vec(m, p, K, nodetype, u, U);
elseif nargout == 2
    error('The function is not differentiable. ');
end

g = double(0); 
for r = 1:(p-1)
    [r_lower, r_upper] = getindex(m, r);
    for s = (r+1):p
        [s_lower, s_upper] = getindex(m, s);
        big = double(0);
        for k = 1:K
            par = Theta(k,r_lower:r_upper,s_lower:s_upper);
            eta = w(r,s)*sqrt(K)*n(k)/N; 
            g = g + lambda1*eta*norm(par(:), 2);
            big = big + norm(par(:), 2)^2;
        end
        g = g + lambda2*w(r,s)*sqrt(big); 
    end
end


