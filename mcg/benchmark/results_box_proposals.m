% ------------------------------------------------------------------------ 
%  Copyright (C)
%  Universitat Politecnica de Catalunya BarcelonaTech (UPC) - Spain
%  University of California Berkeley (UCB) - USA
% 
%  Jordi Pont-Tuset <jordi.pont@upc.edu>
%  Pablo Arbelaez <arbelaez@berkeley.edu>
%  June 2014
% ------------------------------------------------------------------------ 
% This file is part of the MCG package presented in:
%    Arbelaez P, Pont-Tuset J, Barron J, Marques F, Malik J,
%    "Multiscale Combinatorial Grouping,"
%    Computer Vision and Pattern Recognition (CVPR) 2014.
% Please consider citing the paper if you use this code.
% ------------------------------------------------------------------------

%% Some parameters about which result to show
databases = {'pascal2012', 'SBD','COCO'};
gt_sets   = {'val2012', 'val', 'val2014'};
soa_ids   = {'MCG','SCG','CI','GOP','GLS','SeSe','RIGOR','RP','ShSh', 'EB', 'Obj', 'BING', 'QT', 'CPMC'};
          %    1     2    3     4     5     6       7     8     9      10     11     12     13     14
ranked = 1; single = 2;
soa_type  = {ranked, ranked, ranked, single, single, single, single, single, single, single, single, single, ranked, ranked}; 
soa_col   = {'k-', 'r-', 'c-', 'b+', 'gs', 'bo', 'r^', 'm*', 'b>', 'ks', 'r+', 'g^','k--', 'b-'};

% Which soa at each database
soa_which = {[1 2 3 4 5 6 7 8 9 10 11 12 13 14];
             [1 2   4 5 6 7 8   10 11 12 13   ];
             [1 2   4 5 6 7 8   10 11 12 13   ]};

% Overlap levelss 
overlap_levels = [0.5 0.7 0.85];

% Number of candidates of the ranked versions (ranked) of the single versions (single)
nc_ranked = [10:5:100,125:25:1000,1500:500:6000,10000];
nc_single = 1000000;
n_cands   = {nc_ranked, nc_single};

%% Sweep all databases and load pre-computed results
for db_id = 1:length(databases)
   database = databases{db_id};

   % Sweep all soa methods and store the results
   for s_id=1:length(soa_which{db_id})
       soa_id = soa_ids{soa_which{db_id}(s_id)};
       soa_tp = soa_type{soa_which{db_id}(s_id)};
       
       % Load pre-computed results
       soa(db_id).(soa_id) = eval_boxes(soa_id, database, gt_sets{db_id}, n_cands{soa_tp}); %#ok<SAGROW>
   end   
end

%% Plot recall at overlap
for db_id = 1:length(databases)

    % Create figure
    figure;
    title(['Recall @ overlap = [' num2str(overlap_levels) '] on ' databases{db_id} ' ' strrep(gt_sets{db_id},'_','\_')])
    hold on
    grid minor
    grid on
    set(gca,'XScale','log')

    xlabel('Number of candidates')
    ylabel('Recall')
    
    % Compute and store recall
    for s_id=1:length(soa_which{db_id})
        curr_soa = soa(db_id).(soa_ids{soa_which{db_id}(s_id)});
        soa(db_id).(soa_ids{soa_which{db_id}(s_id)}).overlap_levels = overlap_levels; %#ok<SAGROW>
        soa(db_id).(soa_ids{soa_which{db_id}(s_id)}).rec_at_overlap = []; %#ok<SAGROW>

        for ii = 1:length(overlap_levels)
            soa(db_id).(soa_ids{soa_which{db_id}(s_id)}).rec_at_overlap = [soa(db_id).(soa_ids{soa_which{db_id}(s_id)}).rec_at_overlap;
                                                        sum(curr_soa.max_J>overlap_levels(ii),1)./repmat(size(curr_soa.max_J,1),1,length(curr_soa.mean_n_masks))]; %#ok<SAGROW>
        end
    end
    
    % Change the order in which we plot to be consistent with legend
    for ii = 1:length(overlap_levels)
        for s_id=1:length(soa_which{db_id})
            curr_soa = soa(db_id).(soa_ids{soa_which{db_id}(s_id)});
            plot(curr_soa.mean_n_masks, curr_soa.rec_at_overlap(ii,:), soa_col{soa_which{db_id}(s_id)});
        end
    end

    % Legend
    legend(soa_ids(soa_which{db_id}),2) 
end

%% Write values to file
out_dir = '/Users/jpont/Publications/2014_PAMI_UCB/LaTeX-arXiv/data/obj_cands';
for db_id = 1:length(databases)
    for s_id=1:length(soa_which{db_id})
        database = databases{db_id};
        curr_soa = soa(db_id).(soa_ids{soa_which{db_id}(s_id)});
        write_boxes_to_file(curr_soa ,fullfile(out_dir,[database '_' gt_sets{db_id} '_' soa_ids{soa_which{db_id}(s_id)} '_boxes.txt']))
    end
end
