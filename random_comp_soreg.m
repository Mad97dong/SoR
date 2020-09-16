% Using random data 
% Compare adding SoReg regularizer and not-adding SoReg Regularizer
clear;
nmf = zeros(1,30);
reg1 = zeros(1,30); reg2 = zeros(1,30); reg3 = zeros(1,30); reg4 = zeros(1,30); reg5 = zeros(1,30); reg6 = zeros(1,30);
reg7 = zeros(1,30); reg8 = zeros(1,30);
mf1 = zeros(1,30); mf2 = zeros(1,30); mf3 = zeros(1,30); mf4 = zeros(1,30); mf5 = zeros(1,30); mf6 = zeros(1,30);
mf7 = zeros(1,30); mf8 = zeros(1,30);

F = 100;
N = 150;
K = 5;

%V = abs(randn(F,K))*abs(randn(K,N)) + 3*abs(randn(F,N));
V = dlmread("/Users/dongdong/Desktop/cai/V_2.txt");
%outlink = randi(2,F) - 1;
%outlink = tril(outlink,-1) + triu(outlink',0);
%outlink = outlink - diag(diag(outlink));
outlink = dlmread( "/Users/dongdong/Desktop/cai/G_2.txt");

S = similarity(V, 1);
%beta = logspace(-3,4,8);

% beta = 2; using b=2 for beta divergence
D = binornd(1,0.5,F,N); % mask for prediction 50% for training

for i = 1:30
    
    % Initialization
    W_ini = (abs(randn(F,K)) + ones(F,K));
    H_ini = (abs(randn(K,N)) + ones(K,N));

    tol = 1e-8; % scale of data
    maxiter = 1e6;

    % NMF without regularizer
    %==========================================================================
    W = W_ini; H = H_ini;
    cost_nmf = zeros(1,maxiter);
    rmse_nmf = zeros(1,maxiter);

    cost_nmf(1) = betadiv(D.*V, D.*(W*H), 2);
    rmse_nmf(1) = sqrt(norm(D.*(W*H) - D.*V, 'fro')^2/length(find(D.*V)));
    for iter = 2:maxiter
        V1 = D.* (W*H);

        W = W .* ((D.*V) * H') ./ ( (D.*(W*H)) * H' );
        H = H .* (W' * (D.*V)) ./ ( W' * (D.*(W*H)) );

        V2 = D .* (W*H);

        cost_nmf(iter) = betadiv(D.*V, D.*(W*H), 2);
        obj = betadiv(D.*V, D.*(W*H), 2);
        rmse_nmf(iter) = sqrt(norm(D.*(W*H) - D.*V, 'fro')^2/length(find(D.*V)));
        obj_rmse = sqrt(norm(D.*(W*H) - D.*V, 'fro')^2/length(find(D.*V)));

        % stopping criterion
        if norm(V2-V1, 'fro')^2 / norm(V1, 'fro')^2 < tol
            break;
        end

    end

    %{
    figure;
    loglog(cost_nmf')
    legend('NMF')
    title(['objective loss'])
    %}
    W_nmf = W; H_nmf = H;

    %NMF with SoReg regularizer
    %==========================================================================
    W_reg = zeros(F,K,8); H_reg = zeros(K,N,8);
    j = 1;
    temp = S.*outlink;
    A = 2*repmat(sum(temp,2),1,K);
    for beta = logspace(-3,4,8)
        
        W = W_ini; H = H_ini;
        cost_reg = zeros(1, maxiter);
        rmse_reg = zeros(1, maxiter);
        
        %temp = S.*outlink;
        cost_cur = beta/2 * temp(:)' * vecnorm(repelem(W,F,1) - repmat(W,F,1),2,2).^2;
        cost_reg(1) = betadiv(D.*V, D.*(W*H), 2) + cost_cur;
        rmse_reg(1) = sqrt(norm(D.*(W*H) - D.*V, 'fro')^2/length(find(D.*V)));
        %A = 2*repmat(sum(temp,2),1,K);
        %B = 2*(temp)*W;
        
        for iter = 2:maxiter
            B = 2*(temp)*W;
            
            V1 = D .* (W*H);

            W = W .* ((D.*V)*H' + beta*B) ./ (beta * A .* W + (D.*(W*H))*H'); 
            H = H .* (W' * (D.*V)) ./ (W' * (D .* (W*H)));

            V2 = D .* (W*H);

            cost_cur = beta/2 * temp(:)' * vecnorm(repelem(W,F,1) - repmat(W,F,1),2,2).^2;
            obj = betadiv(D.*V, D.*(W*H),2) + cost_cur;
            cost_reg(iter) = obj;
            obj_rmse = sqrt(norm(D.*(W*H) - D.*V, 'fro')^2/length(find(D.*V)));
            rmse_reg(iter) = obj_rmse;

            % stopping criterion
            if norm(V2 - V1, 'fro')^2 / norm(V1, 'fro')^2 < tol
                break
            end
        end
        
        W_reg(:,:,j) = W; H_reg(:,:,j) = H;
        j = j+1;
       
    end
    
    %{
    W = W_ini; H = H_ini;
    cost_reg = zeros(1, maxiter);
    rmse_reg = zeros(1, maxiter);

    temp = S.*outlink;
    cost_cur = beta/2 * temp(:)' * vecnorm(repelem(W,F,1) - repmat(W,F,1),2,2).^2;
    cost_reg(1) = betadiv(D.*V, D.*(W*H), 2) + cost_cur;
    rmse_reg(1) = sqrt(norm(D.*(W*H) - D.*V, 'fro')^2/length(find(D.*V)));
    A = 2*repmat(sum(S.*outlink,2),1,K);
    B = 2*(S.*outlink)*W;

    for iter = 2:maxiter
        V1 = D .* (W*H);

        W = W .* ((D.*V)*H' + beta*B) ./ (beta * A .* W + (D.*(W*H))*H'); 
        H = H .* (W' * (D.*V)) ./ (W' * (D .* (W*H)));

        V2 = D .* (W*H);

        cost_cur = beta/2 * temp(:)' * vecnorm(repelem(W,F,1) - repmat(W,F,1),2,2).^2;
        obj = betadiv(D.*V, D.*(W*H),2) + cost_cur;
        cost_reg(iter) = obj;
        obj_rmse = sqrt(norm(D.*(W*H) - D.*V, 'fro')^2/length(find(D.*V)));
        rmse_reg(iter) = obj_rmse;

        % stopping criterion
        if norm(V2 - V1, 'fro')^2 / norm(V1, 'fro')^2 < tol
            break
        end

    end

    %{
    figure;
    loglog(cost_reg')
    legend('NMF-Soreg')
    title(['objective loss'])
    %}
    W_reg = W; H_reg = H;
    %}
    
    % MF with SoReg regularizer
    %==========================================================================
    W_mf = zeros(F,K,8); H_mf = zeros(K,N,8);
    j = 1;
    
    for beta = logspace(-3,4,8)
        
        W = W_ini; H = H_ini;
        cost_mf = zeros(1,maxiter);
        rmse_mf = zeros(1,maxiter);

        %temp = S.*outlink;
        cost_cur = beta/2 * temp(:)' * vecnorm(repelem(W,F,1) - repmat(W,F,1),2,2).^2;
        cost_mf(1) =betadiv(D.*V, D.*(W*H), 2) + cost_cur;
        rmse_mf(1) = sqrt(norm(D.*(W*H) - D.*V, 'fro')^2/length(find(D.*V)));
        %A = 2*repmat(sum(S.*outlink,2),1,K);
        %B = 2*(S.*outlink)*W;
        

        step_size = 0.001;
        
        if beta > 1e1
            step_size = step_size / 10;
        end
        
        if beta > 1e2
            step_size = step_size / 10;
        end
        
        if beta > 1e3
            step_size = step_size / 10;
        end
        

        for iter = 2:maxiter
            B = 2*temp*W;
            
            V1 = D .* (W*H);

            delta_1 = (D.*(W*H - V)) * H';
            out = [temp, zeros(F,F^2)]';
            out = out(:);
            result = reshape(out(1:(length(out)-F^2)),[],F)';
            delta_3 = beta * result * (repelem(W,F,1) - repmat(W,F,1));
            delta_4 = delta_3;
            delta_W = delta_1 + delta_3 + delta_4;
            W = W - step_size * delta_W;

            H = H - step_size * W'*(D.*(W*H - V));

            V2= D.* (W*H);

            cost_cur = beta/2 * temp(:)' * vecnorm(repelem(W,F,1) - repmat(W,F,1),2,2).^2;
            obj = betadiv(D.*V, D.*(W*H), 2) + cost_cur;
            cost_mf(iter) = obj;
            obj_rmse = sqrt(norm(D.*(W*H) - D.*V, 'fro')^2/length(find(D.*V)));
            rmse_mf(iter) = obj_rmse;

            if norm(V2-V1, 'fro')^2 / norm(V1, 'fro')^2 < tol
                break
            end
        end
        
        W_mf(:,:,j) = W; H_mf(:,:,j) = H;
        j = j+1;
    end
    
    
    %{
    figure;
    loglog(cost_mf')
    legend('MF_soreg')
    title(['objective loss'])
    %}
    %W_mf = W; H_mf = H;    
    
    
    
    figure;
    loglog([cost_nmf' cost_reg' cost_mf'])
    legend('NMF', 'reg', 'MF')
    title(['objective loss'])
    
    
    figure;
    loglog([rmse_nmf' rmse_reg' rmse_mf'])
    legend('NMF', 'reg', 'MF')
    title(['RMSE']) 

    RMSE_nmf = sqrt(norm(abs(D-1).*(W_nmf*H_nmf - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("Iteration %d\n", i);
    fprintf("NMF RMSE is %f\n", RMSE_nmf);

    RMSE_reg1 = sqrt(norm(abs(D-1).*(W_reg(:,:,1)*H_reg(:,:,1) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("NMF_REG1 RMSE is %f\n", RMSE_reg1);
    RMSE_reg2 = sqrt(norm(abs(D-1).*(W_reg(:,:,2)*H_reg(:,:,2) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("NMF_REG2 RMSE is %f\n", RMSE_reg2);
    RMSE_reg3 = sqrt(norm(abs(D-1).*(W_reg(:,:,3)*H_reg(:,:,3) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("NMF_REG3 RMSE is %f\n", RMSE_reg3);
    RMSE_reg4 = sqrt(norm(abs(D-1).*(W_reg(:,:,4)*H_reg(:,:,4) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("NMF_REG4 RMSE is %f\n", RMSE_reg4);
    RMSE_reg5 = sqrt(norm(abs(D-1).*(W_reg(:,:,5)*H_reg(:,:,5) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("NMF_REG5 RMSE is %f\n", RMSE_reg5);
    RMSE_reg6 = sqrt(norm(abs(D-1).*(W_reg(:,:,6)*H_reg(:,:,6) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("NMF_REG6 RMSE is %f\n", RMSE_reg6);
    RMSE_reg7 = sqrt(norm(abs(D-1).*(W_reg(:,:,7)*H_reg(:,:,7) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("NMF_REG7 RMSE is %f\n", RMSE_reg7);
    RMSE_reg8 = sqrt(norm(abs(D-1).*(W_reg(:,:,8)*H_reg(:,:,8) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("NMF_REG8 RMSE is %f\n", RMSE_reg8);
    
    RMSE_mf1 = sqrt(norm(abs(D-1).*(W_mf(:,:,1)*H_mf(:,:,1) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("MF_REG1 RMSE is %f\n", RMSE_mf1);
    RMSE_mf2 = sqrt(norm(abs(D-1).*(W_mf(:,:,2)*H_mf(:,:,2) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("MF_REG2 RMSE is %f\n", RMSE_mf2);
    RMSE_mf3 = sqrt(norm(abs(D-1).*(W_mf(:,:,3)*H_mf(:,:,3) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("MF_REG3 RMSE is %f\n", RMSE_mf3);
    RMSE_mf4 = sqrt(norm(abs(D-1).*(W_mf(:,:,4)*H_mf(:,:,4) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("MF_REG4 RMSE is %f\n", RMSE_mf4);
    RMSE_mf5 = sqrt(norm(abs(D-1).*(W_mf(:,:,5)*H_mf(:,:,5) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("MF_REG5 RMSE is %f\n", RMSE_mf5);
    RMSE_mf6 = sqrt(norm(abs(D-1).*(W_mf(:,:,6)*H_mf(:,:,6) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("MF_REG6 RMSE is %f\n", RMSE_mf6);
    RMSE_mf7 = sqrt(norm(abs(D-1).*(W_mf(:,:,7)*H_mf(:,:,7) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("MF_REG7 RMSE is %f\n", RMSE_mf7);
    RMSE_mf8 = sqrt(norm(abs(D-1).*(W_mf(:,:,8)*H_mf(:,:,8) - V), 'fro')^2/length(find(abs(D-1).*V)));
    fprintf("MF_REG8 RMSE is %f\n", RMSE_mf8);
     
    nmf(i) = RMSE_nmf; 
    reg1(i) = RMSE_reg1; reg2(i) = RMSE_reg2; reg3(i) = RMSE_reg3; reg4(i) = RMSE_reg4;
    reg5(i) = RMSE_reg5; reg6(i) = RMSE_reg6; reg7(i) = RMSE_reg7; reg8(i) = RMSE_reg8;
    mf1(i) = RMSE_mf1; mf2(i) = RMSE_mf2; mf3(i) = RMSE_mf3; mf4(i) = RMSE_mf4;
    mf5(i) = RMSE_mf5; mf6(i) = RMSE_mf6; mf7(i) = RMSE_mf7; mf8(i) = RMSE_mf8;
end
