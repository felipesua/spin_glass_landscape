

%% Parameters
clc,clear
n = 300;
I = eye(n);
M = spherefactory(n);

syms x [n 1] matrix

%A = randn(n,n,n,n);
A = (rand(n,n,n)>.5)-.5; 
%%
x0 = randn(n,1); x0 = x0/norm(x0);
etas = 2./[100 150 200];
iter = 5e2;
freq = 50;

f_vals = zeros(length(etas),iter);
grads = zeros(length(etas),iter);
sharps = zeros(length(etas),iter/10);
% %%
for j = 1:length(etas)
    eta = etas(j);
    x_ = x0;

    i_hess = 1; 
    for i = 1:iter
        i
        egrad  = grad_spinGlass(A,x_);
        grad = egrad - x_ * (x_' * egrad);
        
        
        f_vals(j,i) = objective_f(A,x_);
        grads(j,i) = norm(grad);
        
        if rem(i,freq)==1
            i_hess
            ehess = hess_spinGlass(A,x_);
            hess = (I-x_*x_') * ehess * (I-x_*x_') - dot(x_,egrad) * (I-x_*x_');
            sharps(j,i_hess) = max(eig(hess));
            i_hess=i_hess+1;
        end
        x_ = M.exp(x_, -grad, eta);	
    end
end

%%
subplot(131), plot(f_vals', 'LineWidth',2), grid on
legend('\eta = ' + string(etas))
subplot(132), plot(grads', 'LineWidth',2), grid on
subplot(133), plot(sharps', 'LineWidth',2'), grid on

yline(2./etas,'--', 'Linewidth',2)




 
%%

function r = objective_f(A,x)
    prodx = x;
    dims = size(A);
    p = length(dims);
    for i=1:(p-1)
        prodx = kron(x,prodx);
    end
    r = A(:)' * prodx;
end

function [out] = evalf(f,a)
    a = num2cell(a);
    out = f(a{:});
end

function egrad = grad_spinGlass(A,x0)
    dims = size(A);
    p = length(dims);
    A = cyclic_symmetrize(A);

    prodx = 1;
    for i =  1:(p-1)
        prodx = kron(x0,prodx);
    end
    prodx = reshape(kron(ones(dims(1),1),prodx), dims);
    A = A .* prodx; % faster than reshaping
    % A = reshape(prodx .* A(:), dims);
    egrad = sum(A, 1:(p-1));
    egrad = egrad(:);
end

function ehess = hess_spinGlass(A,x0)
    dims = size(A);
    p = length(dims);
    A = transpose_symmetrize(A);
    A = cyclic_symmetrize(A);

    prodx = 1;
    for i =  1:(p-2)
        prodx = kron(x0,prodx);
    end
    prodx = kron(ones(dims(1),1),prodx);
    prodx = kron(ones(dims(1),1),prodx);
    prodx = reshape(prodx, dims);

    A = A .* prodx;

    ehess = reshape(sum(A, 1:(p-2)), dims(1), dims(1));
    ehess = .5*(ehess + ehess');
end

function A = cyclic_symmetrize(A)
    % Symmetrize A by adding all cyclic shifts (in the order of dimensions) of
    % A
    p = length(size(A));
    A_ = A;
    for i = 1:(p-1)
        A = A + shiftdim(A_,i);
    end
end

function A = transpose_symmetrize(A)
    % Symmetrize A by adding all permutations of A consisting of transpositions
    p = length(size(A));
    permutations = kron(1:p,ones(p-1,1));

    % Generate all permutations
    for i = 1:(p-1)
        permutations(i,2) = i+1;
        permutations(i,i+1) = 2;
    end
    A_ = A;
    % Add up all transpositions
    for i = 2:(p-1)
        A = A + permute(A_,permutations(i,:));
    end
end


function A = partitioned_A_2(n,k)
A = zeros(n,n);
idx = 1:k;
for i = 1:(n/k)
    A(idx+(i-1)*k,idx+(i-1)*k) = randn(k,k);
end
end



function A = partitioned_A_3(n,k)
A = zeros(n,n,n);
idx = 1:k;
for i = 1:(n/k)
    A(idx+(i-1)*k,idx+(i-1)*k,idx+(i-1)*k) = randn(k,k,k);
end
end
 
