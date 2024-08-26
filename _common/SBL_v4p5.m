function [ Ilocs, report ] = SBL_v4p5( A , Y, options, peak_sep_ind_bins )
%
% function [ gamma, report ] = SBL_v4( A , Y, options )
% The idea behind SBL is to find a diagonal replica 'covariance' Gamma.
% Minimizing (YY^T / AGA^T + penality) should lead to the correct
% replica selection (up to a bogus scale factor/amplitude).
%
% Inputs
%
% A - Multiple frequency augmented dictionary < n , m , f >
%     n: number of sensors
%     m: number of replicas
%     f: number of frequencies
%   Note: if f==1, A = < n , m >
% Y - Multiple snapshot multiple frequency observations < n , L , f >
%     n: number of sensors
%     L: number of snapshots
%     f: number of frequencies
%   Note: if f==1, Y = < n , L >
% options - see SBLset.m 
%
%
% Outputs
%
% gamma <m , 1> - vector containing source power
%                 1: surfaces found by minimum error norm
% report - various report options
%
%--------------------------------------------------------------------------
% Version 1.0:
% Code originally written by P. Gerstoft.
%
% Version 2.23
% Edited to include multiple frequency support: 5/16/16
%
% Version 3.1
% Different convergance norm and code update
% A and Y have now one more dimensions
% Posterior unbiased mean
% handles single snapshot
%
% Version 3.32
% Robust version of SBL with Phi parameter in options structure
% Santosh Nannuru & Kay L Gemba
% NoiseLab/SIO/UCSD gemba@ucsd.edu & snannuru@ucsd.edu
%
% Version 4.5
% Stopping criteria updated
% Y. Park & P. Gerstoft

%%
options.SBL_v = '4.5';

%% slicing
if options.tic == 1
    tic
end

activeIndRepN = options.activeIndRepN;

%% Initialize variables
Nfreq       = size(A,3);        % number of frequencies
Nsensor     = size(A,1);        % number of sensors
Ntheta      = size(A,2);        % number of dictionary entries
Nsnapshot   = size(Y,2);        % number of snapshots in the data covariance
Nsource     = options.Nsource;  % number of sources

% noise power initialization
sigc      = ones(Nfreq,1) * options.noisepower.guess;
%sigc      = options.noisepower.guess;

% posterior
if isfield(options,'x_post_calculation') == 1
x_post    = zeros( Ntheta, Nsnapshot, Nfreq );
end
% minimum (global) gamma
gmin_global = realmax;
% space allocation
errornorm   = zeros(options.convergence.maxiter,1);
activeInd   = zeros(options.convergence.maxiter,Nsource);
sigcItr     = zeros(options.convergence.maxiter,1);

% Sample Covariance Matrix
SCM          = zeros( Nsensor , Nsensor , Nfreq );
maxnoise     = zeros( Nfreq , 1 );
% initialize with CBF output (assume single frequency)
gamma        = zeros(Ntheta, 1);
sigc         = zeros(Nfreq , 1);
for iF = 1:Nfreq
    Af           = squeeze(A(:,:,iF));
    Yf           = squeeze(Y(:,:,iF));
    SCM(:,:,iF)  = (1/Nsnapshot) * (Yf * Yf');
    maxnoise(iF) = real(trace(squeeze(SCM(:,:,iF))))/Nsensor;

    gamma_init   = sum(abs(Af' * Yf).^2, 2) / Nsnapshot ./ sum(abs(Af(:,1)).^2)^2; % Note: normalize with norm(Af(:,iColumn))^4, Below Eq. (24) [Initialization] --*

    % -- Compute the initial noise variance
    if Nsource==1
        [~,indx] = max(gamma_init);
    else
        [~,indx] = SBLpeaks_1D(gamma_init,Nsource);
    end
    Am       = Af(:,indx);        % only active replicas
    Aplus    = pinv(Am);
    P_N      = eye(Nsensor) - Am*Aplus;
    sigc(iF) = real(trace(P_N*SCM(:,:,iF)))/(Nsensor - Nsource);  % noise covariance estimate

    % -- Compute initial estimates of gamma
    gamma_temp =  max(  max(gamma_init)*options.gamma_range , gamma_init -  sigc(iF)); % Note: norm(Af(:,iColumn)) = sqrt( Nsensor ) Below Eq. (24) [Initialization] --*
    gamma      = gamma + gamma_temp;
end

gamma_num    = zeros(Ntheta,Nfreq);
gamma_denum  = zeros(Ntheta,Nfreq);

%% Main Loop
disp(['SBL version ', options.SBL_v ,' initialized.']);

for j1 = 1 : options.convergence.maxiter
        
    % for error analysis
    gammaOld    = gamma;
    Itheta      = find(gamma>max(gamma)*options.gamma_range);
    gammaSmall  = gamma(Itheta);

    %% gamma update
    % this is a sum over frequency and can be done by multiple processors
    % --> multi-frequency SBL should be almost as fast as single frequency if num(proc) = Nfreq
    for iF = 1 : Nfreq
        Af = squeeze(A(:,Itheta,iF));
        Yf = squeeze(Y(:,:,iF));

        % SigmaY inverse
        SigmaYinv = eye(Nsensor) / (sigc(iF) * eye(Nsensor) + ...
           Af * bsxfun(@times, gammaSmall, Af') );

        B =  SigmaYinv*Af; % \Sigma^-1 a_m , m=1,..,M
        
        % Sum over snapshots and normalize
        gamma_num(Itheta,iF)   = ( real( sum ( conj(B).*(SCM(:,:,iF)*B)) ) );

        % positive def quantity, abs takes care of roundoff errors        
        gamma_denum(Itheta,iF) = abs( sum  ( (Af' * SigmaYinv).' .* Af, 1 ) )  ;
    end
    
    % Fixed point Eq. update
     gamma(Itheta)  = gamma(Itheta)   .* ((sum( gamma_num(Itheta,:)    ,2 ) ./...
        sum( gamma_denum(Itheta,:)  ,2 ) ).^(1/options.fixedpoint) ) ;

    
    %% sigma update
    % locate same peaks for all frequencies
    if j1 == 1, IlocsOld = [];
    else,       IlocsOld = Ilocs;
    end

    [~, Ilocs] = SBLpeaks_1D(gamma, Nsource);
    sigmaUp = 1;
    if isempty(setdiff(IlocsOld,Ilocs)), sigmaUp = 0; end %SBL-G skips sigma update when active indices did not change.    
    if sigmaUp
        for iF = 1 : Nfreq
            % only active replicas
            Am       = squeeze(A(:,Ilocs,iF));
            % noise estimate
            sigc(iF) = real(trace( (eye(Nsensor)-Am*pinv(Am)) * squeeze(SCM(:,:,iF)) ) / ( Nsensor - Nsource ) );
            sigc(iF) = min(sigc(iF),maxnoise(iF));          % cannot be larger than signal+noise.
            sigc(iF) = max(sigc(iF),maxnoise(iF)*10^-10);   % snr>100 is unlikely larger than signal.
        end
    end

    % check if the index repeats
    if ~isempty(setdiff(IlocsOld,Ilocs))
        activeIndRep = 0;
    else
        if exist('activeIndRep','var')==0, activeIndRep = 0; end
        activeIndRep = activeIndRep + 1;
    end

    activeInd(j1,:) = Ilocs.';
    sigcItr(j1)     = sigc;
    
    %% Convergence  checks convergance and displays status reports
    % convergence indicator
    errornorm(j1) = norm ( gamma - gammaOld, 1 ) / norm ( gamma, 1 );
        
    % global min error
    if j1 > options.convergence.min_iteration  &&  errornorm(j1) < gmin_global
        gmin_global  = errornorm(j1);
        gamma_min    = gamma;
        iteration_L1 = j1;
    end
    
    % inline convergence code
    if activeIndRep > activeIndRepN || (j1 > options.convergence.min_iteration && ( errornorm(j1) < options.convergence.error  || iteration_L1 + options.convergence.delay <= j1))
        if options.flag == 1
            disp(['Solution converged. Iteration: ',num2str(sprintf('%.4u',j1)),'. Error: ',num2str(sprintf('%1.2e' , errornorm(j1) )),'.'])
        end
        break; % goodbye
    % not convereged
    elseif j1 == options.convergence.maxiter
        if options.flag == 1
            warning(['Solution not converged. Error: ',num2str(sprintf('%1.2e' , errornorm(j1) )),'.'])
        end
    % status report
    elseif j1 ~= options.convergence.maxiter  && options.flag == 1 && mod(j1,options.status_report) == 0 % Iteration reporting
        disp(['Iteration: ',num2str(sprintf('%.4u',j1)),' Dic size: ',num2str(length(Itheta)),'. Error: ',num2str(sprintf('%1.2e' , errornorm(j1) )),'.' ])
    end
    
end

%% Peak finding
[~, Ilocs] = SBLpeaks_1D_sep(gamma, Nsource, peak_sep_ind_bins);

%% Posterior distribution for polarity
% x_post - posterior unbiased mean
if isfield(options,'x_post_calculation') == 1
for iF = 1 : Nfreq
    Af = squeeze (A(:,:,iF));
    x_post(:,:,iF) = repmat(gamma, [1 Nsnapshot] ) .* (Af' / (sigc(iF) * eye(Nsensor) + ...
        Af * (repmat(gamma, [1 Nsensor] ) .* Af')) * squeeze(Y(:,:,iF)));
end
end
%% function return
gamma = gamma_min;

%% Report section
% vectors containing errors
report.results.error    = errornorm(1:iteration_L1);
% Error when minimum was obtained
report.results.iteration_L1 = iteration_L1;
% General info
report.results.final_iteration.iteration  = j1;
report.results.final_iteration.noisepower = sigc;

if options.tic == 1
    report.results.toc = toc;
else
    report.results.toc = 0;
end

% data
report.results.final_iteration.gamma  = gamma;
if isfield(options,'x_post_calculation') == 1
report.results.final_iteration.x_post = x_post;
end

report.options = options;

report.results.activeIndices = activeInd(1:iteration_L1,:);
report.results.sigcItr       = sigcItr(1:iteration_L1);

% debug output parameters (assuming single frequency)
%report.SigmaYinv = SigmaYinv;
%report.SCM = squeeze(SCM);
end

function [pks, locs] = SBLpeaks_1D(gamma, Nsources)
% fast alternative for findpeaks in 1D case
% output variables
pks  = zeros(Nsources,1);
locs = zeros(Nsources,1);

% zero padding on the boundary
gamma_new            = zeros(length(gamma)+2,1);
gamma_new(2:end-1)   = gamma;

[~, Ilocs] = sort(gamma,'descend');

% current number of peaks found
npeaks = 0;
for ii = 1:length(Ilocs)
    % local patch area surrounding the current array entry i.e. (r,c)
    local_patch      = gamma_new(Ilocs(ii):Ilocs(ii)+2);
    % zero the center
    local_patch(2)   = 0;
    if sum(sum(gamma(Ilocs(ii)) > local_patch)) == 3
        npeaks       = npeaks + 1;
        pks(npeaks)  = gamma(Ilocs(ii));
        locs(npeaks) = Ilocs(ii);
        % if found sufficient peaks, break
        if npeaks == Nsources
            break;
        end
    end
end
% if Nsources not found
if npeaks ~= Nsources
    pks(npeaks+1:Nsources)  = [];
    locs(npeaks+1:Nsources) = [];
end
end

function [pks, locs] = SBLpeaks_1D_sep(gamma, Nsources, NIndSep)
% fast alternative for findpeaks in 1D case with an index separation
% output variables
pks  = zeros(Nsources,1);
locs = zeros(Nsources,1);

% zero padding on the boundary
gamma_new                = zeros(length(gamma)+2,1);
gamma_new(2:end-1)       = gamma;

[~, Ilocs]= sort(gamma,'descend');

% current number of peaks found
npeaks = 0;
for K = 1:Nsources
    for ii = 1:length(Ilocs)
        % local patch area surrounding the current array entry i.e. (r,c)
        local_patch      = gamma_new(Ilocs(ii):Ilocs(ii)+2);
        % zero the center
        local_patch(2)   = 0;
        if sum(sum(gamma(Ilocs(ii)) > local_patch)) == 3
            npeaks       = npeaks + 1;
            pks(npeaks)  = gamma(Ilocs(ii));
            locs(npeaks) = Ilocs(ii);
            Ilocs( abs(Ilocs - Ilocs(ii)) <= NIndSep ) = [];
            break;
        end
        if ii == length(Ilocs)
            error('It was not possible to localize peaks with the chosen sep!')
        end
    end
    if isempty(Ilocs)
        error('It was not possible to localize peaks with the chosen sep!')
    end
    if npeaks == Nsources % if found sufficient peaks, break
        break;
    end
end
end