function C = cmc(D, varargin)
% CMC computes the Cumulative Match Characteristic.
%   C = CMC(D) computes the CMC of given distance matrix D between each pair
%   of gallery and probe person. Both gallery and probe persons are
%   assumed to be unique, i.e., the i-th gallery person matches only to
%   the i-th probe person. D(i, j) should be the distance between
%   gallery instance i and probe instance j.
%
%   C = CMC(D, G, P) computes the CMC with G and P are id labels of each
%   person in gallery and probe sets. D is M*N where M and N are the lengths of
%   vector G and P, respectively. This function will first randomly select
%   exactly one instance among for each gallery identity and then calculate
%   the CMC of this sub-matrix. This procedure will be repeated 100 times
%   and the final result is the mean of them.

switch nargin
    case 1
        assert(ismatrix(D));
        assert(size(D, 1) == size(D, 2));

    case 3
        G = varargin{1};
        P = varargin{2};
        assert(isvector(G));
        assert(isvector(P));
        assert(length(G) == size(D, 1));
        assert(length(P) == size(D, 2));

    otherwise
        error('Invalid inputs');
end

if nargin == 1
    C = cmc_core(D, 1:size(D, 1), 1:size(D, 1));
else
    gtypes = union(G, []);
    ngtypes = length(gtypes);
    ntimes = 100;

    C = zeros(ngtypes, ntimes);
    for t = 1:ntimes
        subdist = zeros(ngtypes, size(D, 2));
        for i = 1:ngtypes
            j = find(G == gtypes(i));
            k = j(randi(length(j)));
            subdist(i, :) = D(k, :);
        end
        C(:, t) = cmc_core(subdist, gtypes, P);
    end
    C = mean(C, 2);
end

end


function C = cmc_core(D, G, P)
m = size(D, 1);
n = size(D, 2);
[~, order] = sort(D);
match = (G(order) == repmat(P, [m, 1]));
C = cumsum(sum(match, 2) / n);
end