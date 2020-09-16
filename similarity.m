function S = similarity(R,t)

%similarity function
% R: the orginal matrix
% type: which type of similarity function (1.VSS / 2.PCC)


switch t
    case 1
        M = size(R,1);
        S = zeros(M);
        for i = 1:M
            for f = 1:M
                numerator = R(i,:)*R(f,:)';
                common_index = find(R(i,:) .* R(f,:));
                denominator = norm(R(i,common_index)) * norm(R(f, common_index));
                
                S(i,f) = numerator / denominator;
            end
        end
        S(isnan(S)) = 0;
    case 2
        M = size(R,1);
        S = zeros(M);
        R(R==0) = nan;
        R = R - mean(R, 2, 'omitnan');
        R(isnan(R)) = 0;
        for i = 1:M
            for f = 1:M
                numerator = R(i,:) * R(f,:)';
                common_index = find(R(i,:).*R(f,:));
                denominator = norm(R(i, common_index)) * norm(R(f, common_index));
                
                S(i,f) = (numerator / denominator + 1)/2;
            end
        end
        S(isnan(S)) = 0;
end



        