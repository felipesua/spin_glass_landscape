% % run once
% clc, clear
% cd 'C:\Users\Felipe\Downloads\NN Landscape'
% cd './Manopt_7.0/manopt/'
% addpath('./checkinstall/')
% % basicexample

%% Parameters
clc,clear
n = 100;
I = eye(n);
M = spherefactory(n);

syms x [n 1] matrix
%A = (rand(n,n,n)>.5)-.5;
% A = partitioned_A_3(n,25);

x0 = rand(n,1); x0 = x0/norm(x0);

[egrad,ehess] = derivative_closedform(A,x0); % first run - may take a while
%%
x0 = randn(n,1); x0 = x0/norm(x0);
etas = 2./[100 200 300 400 500]*10;
iter = 1e2;

f_vals = zeros(length(etas),iter);
grads = zeros(length(etas),iter);
sharps = zeros(length(etas),iter);
% %%
for j = 1:length(etas)
    eta = etas(j);
    x_ = x0;
    for i = 1:iter
        [egrad,ehess] = derivative_closedform(A,x_);
        grad = egrad - x_ * (x_' * egrad);
        hess = (I-x_*x_') * ehess * (I-x_*x_') - dot(x_,egrad) * (I-x_*x_');
        
        f_vals(j,i) = objective_f(A,x_,3);
        grads(j,i) = norm(grad);
        sharps(j,i) = max(eig(hess));
        
        x_ = M.exp(x_, -grad, eta);	
    end
end
%  %%
subplot(131), plot(f_vals', 'LineWidth',2), grid on
legend('\eta = ' + string(etas))
subplot(132), plot(grads', 'LineWidth',2), grid on
subplot(133), plot(sharps', 'LineWidth',2'), grid on






%% tests
clc
n = 5;
A = randn(n,n,n,n);
x0 = randn(n,1); x0 = x0/norm(x0);
objective_f(A,x0,4)
[egrad,ehess] = derivative_closedform(A,x0);

%%
clc
A = partitioned_A_3(10,5);
A
%%

function r = objective_f(A,x,k)
prodx = x;
for i=1:(k-1)
    prodx = kron(x,prodx);
end
r = A(:)' * prodx;
end

function [out] = evalf(f,a)
    a = num2cell(a);
    out = f(a{:});
end

function [egrad,ehess] = derivative_closedform(A,x0)
% Symmetrize A
dims = size(A);
n = length(dims);

A_ = A;
idx = circshift(1:n,1);
for i = 1:(n-1)
    A = A + permute(A_,idx);
    idx = circshift(idx,1);
end
A_ = A;

% Contract A with x0^n 
% g = A* x^n
% H = 2A*x^(n-1)
prodx = x0;
for i =  1:(n-3)
    prodx = kron(x0,prodx);
end
% ehess
A = reshape( kron( ones(dims(1),1) , kron(ones(dims(1),1),prodx)) .* A_(:), dims);
ehess = sum(A);
for i = 1:n-3
    ehess = sum(ehess);
end
ehess = reshape(ehess,dims(1),dims(1));
ehess = ehess + ehess';

% egrad
prodx = kron(x0,prodx);
A = reshape(kron(ones(dims(1),1), prodx) .* A_(:), dims);
egrad = sum(A);
for i = 1:n-2
    egrad = sum(egrad);
end
egrad = reshape(egrad,[],1);
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

% %% sanity check - Via gradient and Hessian commands 
% f2 = symmatrix2sym(f);
% g = matlabFunction(gradient(f2));
% H = matlabFunction(hessian(f2));
% egrad2 = evalf(g,x0);
% ehess2 = evalf(H,x0);


% function r = derivative_autodiff(Df,x,x0)
%     r = double(symmatrix2sym( subs(Df,x,x0) ));
% end
