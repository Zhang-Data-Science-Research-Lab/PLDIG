function [f, g] = mylkh(m, n, p, K, nodetype, phi, mu, sigma, add, x)
% Calculate the likelihood function value f and gradient g at current value 
% of model parameters x. 
% phi: input sufficient statistics (have been standardized by mu and sigma). 
%      It is a 1*K cell array. Each cell is a n_k*sum(m) data matrix. 
% mu, sigma: used to standardize the sufficient statistics. They are
%            sum(m)-dimensioal vectors. 
% add: additional parameters, for example, negative binomial r, truncated 
%      poisson truncation points


N = sum(n); 
[theta, Theta] = vec2par(x, p, m, K, nodetype); 

% Calculate the function value f
f = double(0); 
% Calculate the gradient g in terms of theta and Theta
theta_g = zeros(size(theta));
Theta_g = zeros(size(Theta));
% Calculate the sufficient statistics of all node conditional
% distributions. They are n_k*m_r matrices. 
for k = 1:K
    for i = 1:p
        [i_lower, i_upper] = getindex(m, i);
        suff = repmat(theta(k,i_lower:i_upper), n(k), 1) + ...
            phi{k}(:,[1:(i_lower-1), (i_upper+1):end])*...
            reshape(Theta(k,[1:(i_lower-1), (i_upper+1):end],i_lower:i_upper), sum(m)-m(i), m(i)); % n_k*m_r
        
        if nodetype(i) == 'd'
            eta = suff * diag(1 ./ sigma(i_lower:i_upper)); % n_k*m_r
            y = zeros(n(k), m(i)); % n_k*m_r
            for j = 1:m(i)
                y(:,j) = mu(i_lower+j-1) + phi{k}(:,i_lower+j-1) .* sigma(i_lower+j-1); 
            end
            f = f + trace(-eta*y')+sum(log(1+sum(exp(eta), 2))); 
            grad = exp(eta)./repmat(1+sum(exp(eta), 2), 1, m(i)); % n_k*m_r
            for j = 1:m(i)
                grad(:,j) = grad(:,j) ./ sigma(i_lower+j-1) - mu(i_lower+j-1)/sigma(i_lower+j-1); 
            end
            theta_g(k,i_lower:i_upper) = ...
                sum(grad-phi{k}(:,i_lower:i_upper), 1); 
            temp = phi{k}(:,[1:(i_lower-1), (i_upper+1):end])'*...
                (grad-phi{k}(:,i_lower:i_upper)); 
            Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower:i_upper) = ...
                reshape(Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower:i_upper), sum(m)-m(i), m(i)) + temp;
            Theta_g(k,i_lower:i_upper,[1:(i_lower-1), (i_upper+1):end]) = ...
                reshape(Theta_g(k,i_lower:i_upper,[1:(i_lower-1), (i_upper+1):end]), m(i), sum(m)-m(i)) + temp';
        end
        
        if nodetype(i) == 'g'
            y = phi{k}(:,i_lower:i_upper); % n_k*1
            if Theta(k,i_lower,i_lower) >= 0
                f = Inf;
            else
                f = f - sum(log(normpdf(y, -0.5*suff/Theta(k,i_lower,i_lower), ...
                    sqrt(-0.5/Theta(k,i_lower,i_lower)))));
            end
            grad = -0.5.*suff./Theta(k,i_lower,i_lower); % n_k*1
            theta_g(k,i_lower) = sum(grad-phi{k}(:,i_lower));
            temp = phi{k}(:,[1:(i_lower-1), (i_upper+1):end])'*...
                (grad-phi{k}(:,i_lower)); 
            Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower) = ...
                reshape(Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower), sum(m)-1, 1) + temp;
            Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]) = ...
                reshape(Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]), 1, sum(m)-1) + temp';
            Theta_g(k,i_lower,i_lower) = -sum(y.^2)+...
                0.25*sum(suff.^2)/Theta(k,i_lower,i_lower)^2-...
                0.5*n(k)/Theta(k,i_lower,i_lower);
        end
        
        if nodetype(i) == 'p'
            eta = suff ./ sigma(i_lower); % n_k*1
            y = mu(i_lower) + phi{k}(:,i_lower).*sigma(i_lower); % n_k*1
            f = f - sum(log(poisspdf(round(y), exp(eta))));
            grad = exp(eta); 
            grad = grad ./ sigma(i_lower) - mu(i_lower)/sigma(i_lower); % n_k*1
            theta_g(k,i_lower) = sum(grad-phi{k}(:,i_lower)); 
            temp = phi{k}(:,[1:(i_lower-1), (i_upper+1):end])'*...
                (grad-phi{k}(:,i_lower));
            Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower) = ...
                reshape(Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower), sum(m)-1, 1) + temp;
            Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]) = ...
                reshape(Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]), 1, sum(m)-1) + temp';
        end
        
        if nodetype(i) == 't'  % truncated poisson
            eta = suff ./ sigma(i_lower); % n_k*1
            y = mu(i_lower) + phi{k}(:,i_lower).*sigma(i_lower); % n_k*1
            prob = zeros(1+add(i), n(k)); 
            for r = 1:n(k)
                for j = 0:add(i)
                    prob(j+1,r) = exp(j*eta(r)) / factorial(j); 
                end
            end
            f = f - sum(log(exp(y.*eta) ./ factorial(round(y)) ./ sum(prob)')); 
            grad = (0:add(i))*prob ./ sum(prob); 
            grad = grad'; 
            grad = grad ./ sigma(i_lower) - mu(i_lower)/sigma(i_lower); % n_k*1
            theta_g(k,i_lower) = sum(grad-phi{k}(:,i_lower)); 
            temp = phi{k}(:,[1:(i_lower-1), (i_upper+1):end])'*...
                (grad-phi{k}(:,i_lower)); 
            Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower) = ...
                reshape(Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower), sum(m)-1, 1) + temp; 
            Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]) = ...
                reshape(Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]), 1, sum(m)-1) + temp'; 
        end
        
        if nodetype(i) == 'n' % negative binomial
            eta = suff ./ sigma(i_lower); % n_k*1
            y = mu(i_lower) + phi{k}(:,i_lower).*sigma(i_lower); % n_k*1
            if any(eta >= 0)
                f = Inf;
            else
                f = f - sum(log(pdf('Negative Binomial', round(y), add(i), 1-exp(eta)))); 
            end
            grad = (add(i)*exp(eta)) ./ (1-exp(eta)); 
            grad = grad ./ sigma(i_lower) - mu(i_lower)/sigma(i_lower); % n_k*1
            theta_g(k,i_lower) = sum(grad-phi{k}(:,i_lower)); 
            temp = phi{k}(:,[1:(i_lower-1), (i_upper+1):end])'*...
                (grad-phi{k}(:,i_lower));
            Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower) = ...
                reshape(Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower), sum(m)-1, 1) + temp;
            Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]) = ...
                reshape(Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]), 1, sum(m)-1) + temp';
        end
        
        if nodetype(i) == 'e'
            eta = suff ./ sigma(i_lower); % n_k*1
            y = mu(i_lower) + phi{k}(:,i_lower).*sigma(i_lower); % n_k*1
            if any(eta >= 0)
                f = Inf;
            else
                f = f - sum(log(exppdf(y, -1./eta)));
            end
            grad = -1./eta; 
            grad = grad ./ sigma(i_lower) - mu(i_lower)/sigma(i_lower); % n_k*1
            theta_g(k,i_lower) = sum(grad-phi{k}(:,i_lower)); 
            temp = phi{k}(:,[1:(i_lower-1), (i_upper+1):end])'*...
                (grad-phi{k}(:,i_lower));
            Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower) = ...
                reshape(Theta_g(k,[1:(i_lower-1), (i_upper+1):end],i_lower), sum(m)-1, 1) + temp;
            Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]) = ...
                reshape(Theta_g(k,i_lower,[1:(i_lower-1), (i_upper+1):end]), 1, sum(m)-1) + temp';
        end
    end
end

f = f/N;
theta_g = theta_g./N;
Theta_g = Theta_g./N;
g = par2vec(m, p, K, nodetype, theta_g, Theta_g); 




