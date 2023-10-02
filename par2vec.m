function x = par2vec(m, p, K, nodetype, theta, Theta)
% change parameter to vector
% theta is a K*M array
% Theta is a K*M*M array

% change theta to a stacked column vector
x = theta(:);
% Theta (i<j)
for k = 1:K
    for i = 1:(p-1)
        [i_lower,i_upper] = getindex(m, i);
        for j = (i+1):p
            [j_lower,j_upper] = getindex(m, j);
            par = Theta(k,i_lower:i_upper,j_lower:j_upper);
            x = [x ; par(:)];
        end
    end
end
% Theta (ii)
for k = 1:K
    for i = 1:p
        if nodetype(i) == 'g'
            i_lower = getindex(m, i);
            x = [x ; Theta(k,i_lower,i_lower)];
        end
    end
end

