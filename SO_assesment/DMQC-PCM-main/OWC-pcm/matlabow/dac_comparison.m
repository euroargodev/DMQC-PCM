%% comparison between two outputs from RAw and adjusted data

%Check diff between the raw and Adjusted data
%load data from my analysis
float_names={'6902638','6902559'}
%{'3901928','3901936','7900155','7900071','7900155','7900085','3901924'}
 
for i=1:length(float_names)
    flt_name = float_names{i};
% Raw datawith some adjustement due to pressure: create_float_source('wmonum') 
load(['/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_source/default/',flt_name,'.mat'])
PRES_r=PRES;
PTMP_r=PTMP;
SAL_r=SAL;
TEMP_r=TEMP;


% Adjusted data
load(['/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_source/adjusted/',flt_name,'.mat'])

% positive value - float was fresher-they added positive correction
% negative value - float was saltier- they reduced salinity-added negative
% correction
PRES_a=PRES;
PTMP_a=PTMP;
SAL_a=SAL;
TEMP_a=TEMP;

%% 
%s et a depth of the float data for comparison
level = size(TEMP_a,1)-5;
% Checking what correction was applied and if any was
diff_raw_adj = abs(SAL_r(end-level,:)-SAL_a(end-level,:));

% load PCM calulations
load(['/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_calib/argo/cal_',flt_name,'.mat'])

diff_raw_pcm = abs(SAL_r(end-level,:)-cal_SAL(end-level,:));
diff_adj_pcm = abs(SAL_a(end-level,:)-cal_SAL(end-level,:));

% check if any corectins was applied
disp(['Float ',flt_name])

%operator did not applied any corrections
if (nanmax(diff_raw_adj) <= 0.001) % this threshold will give some space for CTM correction whic could be applied
    disp(['No salinity corrections applied by operator.'])
    % compare dm with PCM output- check if no correction was needed
    if (nanmax(diff_raw_pcm) < 0.01)      
        disp(['Correct analysis - no salinity corrections was needed compared to PCM.'])    
    else % PCM corrections were needed by operator didn't apply them
        disp(['Float undercorrected - review needed.'])
    end
% check if applied correction was need and was correct    
elseif (nanmax(diff_raw_adj) >= 0.01)
    disp(['OWC salinity correction applied.'])   
    
    % compare adjusted with PCM output- check if correction was needed    
    if(nanmax(diff_raw_pcm) > 0.01)
       disp(['OWC correction was needed compared to PCM'])      
     if (nanmax(diff_adj_pcm) <= 0.004)
       disp(['Correction applied correctly, difference between PCM is < 0.004'])    
     else %(max(diff_adj_pcm) > 0.004)
       disp(['Correction applied incorrectly, difference between PCM is > 0.004 - review needed.'])      
     end
     
    else % (max(diff_raw_pcm) < 0.01)
       disp(['Float overcorrected - no salinity correction was needed compared to PCM - review needed.'])  
       
    end
   
else %((max(corr_added > 0.001)) && (max(corr_added < 0.01)))
    disp(['Float overcorrected within +/- 0.01.'])
end

%save(['/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_comparison/ctd/',...
 %   flt_name,'_DAC_vs_OWC.mat'],'',)
 
% Comparison float at the deepest level
figure

subplot(3,1,1)

plot(PROFILE_NO,SAL_r(end-level,:),'b')
hold on
plot(PROFILE_NO,SAL_a(end-level,:),'r')
plot(PROFILE_NO,cal_SAL(end-level,:),'g')

plot(PROFILE_NO,SAL_r(end-level,:),'.b')
plot(PROFILE_NO,SAL_a(end-level,:),'.r')
plot(PROFILE_NO,cal_SAL(end-level,:),'.g')

xlabel('Profile number')
ylabel('Salinity [PSU]')
title(['Salinity from the deepest level ',num2str(flt_name)])
legend('raw','d-mode operator','PCM','location', 'northwest')

subplot(3,1,2)
plot(PROFILE_NO,diff_raw_adj,'r')
hold on
plot(PROFILE_NO,diff_raw_pcm ,'g')

plot(PROFILE_NO,diff_raw_adj,'.r')
plot(PROFILE_NO,diff_raw_pcm ,'.g')

xlabel('Profile number')
ylabel('Salinity [PSU]')
title(['Differences from the deepest level ',num2str(flt_name)])
legend('raw - d-mode','raw - pcm','location', 'northwest')

subplot(3,1,3)
plot(PROFILE_NO,diff_adj_pcm,'m')
hold on
plot(PROFILE_NO,diff_adj_pcm,'.m')

xlabel('Profile number')
ylabel('Salinity [PSU]')
title(['Differences between d-mode and pcm at the deepest level ',num2str(flt_name)])


drawnow
set(gcf,'papertype','usletter','paperunits','inches','paperorientation','portrait','paperposition',[.25,.5,8,10]);
%print(['-depsc ','/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_comparison/ctd/',flt_name, '_PCM comparison.eps']);
%print([gcf,'-depsc ','/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_comparison/ctd/',flt_name,'_DAC_vs_OWC.eps']);
print(['/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_comparison/ctd/',flt_name,'_DAC_vs_OWC_deepest'],'-depsc');

% %% Comparison float at around 1000 dbar
% 
% level = 58;
% % Checking what correction was applied and if any was
% diff_raw_adj = abs(SAL_r(level,:)-SAL_a(level,:));
% 
% % load PCM calulations
% load(['/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_calib/ctd/cal_',flt_name,'.mat'])
% 
% diff_raw_pcm = abs(SAL_r(level,:)-cal_SAL(level,:));
% diff_adj_pcm = abs(SAL_a(level,:)-cal_SAL(level,:));
% 
% % check if any corectins was applied
% disp(['Float ',flt_name])
% 
% %operator did not applied any corrections
% if (max(diff_raw_adj) <= 0.001) % this threshold will give some space for CTM correction whic could be applied
%     disp(['No salinity corrections applied by operator.'])
%     % compare dm with PCM output- check if no correction was needed
%     if any(logical(diff_raw_pcm < 0.01))      
%         disp(['Correct analysis - no salinity corrections was needed compared to PCM.'])    
%     else % PCM corrections were needed by operator didn't apply them
%         disp(['Float undercorrected - review needed.'])
%     end
% % check if applied correction was need and was correct    
% elseif (max(diff_raw_adj) >= 0.01)
%     disp(['OWC salinity correction applied.'])   
%     
%     % compare adjusted with PCM output- check if correction was needed    
%     if(max(diff_raw_pcm) > 0.01)
%        disp(['OWC correction was needed compared to PCM'])      
%      if (max(diff_adj_pcm) <= 0.004)
%        disp(['Correction appiled correctly, difference between PCM is < 0.004'])    
%      else %(max(diff_adj_pcm) > 0.004)
%        disp(['Correction appled incorrectly, difference between PCM is > 0.004 - review needed.'])      
%      end
%      
%     else % (max(diff_raw_pcm) < 0.01)
%        disp(['Float overcorrected - no salinity correction was needed compared to PCM - review needed.'])  
%        
%     end
%    
% else %((max(corr_added > 0.001)) && (max(corr_added < 0.01)))
%     disp(['Float overcorrected within +/- 0.01.'])
% end
% 
% % figures
% figure
% 
% subplot(3,1,1)
% 
% plot(PROFILE_NO,SAL_r(level,:),'b')
% hold on
% plot(PROFILE_NO,SAL_a(level,:),'r')
% plot(PROFILE_NO,cal_SAL(level,:),'g')
% 
% plot(PROFILE_NO,SAL_r(level,:),'.b')
% plot(PROFILE_NO,SAL_a(level,:),'.r')
% plot(PROFILE_NO,cal_SAL(level,:),'.g')
% 
% xlabel('Profile number')
% ylabel('Salinity [PSU]')
% title(['Salinity from the deepest level ',num2str(flt_name)])
% legend('raw','d-mode operator','PCM','location', 'northwest')
% 
% subplot(3,1,2)
% plot(PROFILE_NO,diff_raw_adj,'r')
% hold on
% plot(PROFILE_NO,diff_raw_pcm ,'g')
% 
% plot(PROFILE_NO,diff_raw_adj,'.r')
% plot(PROFILE_NO,diff_raw_pcm ,'.g')
% 
% xlabel('Profile number')
% ylabel('Salinity [PSU]')
% title(['Differences from the deepest level ',num2str(flt_name)])
% legend('raw - d-mode','raw - pcm','location', 'northwest')
% 
% subplot(3,1,3)
% plot(PROFILE_NO,diff_adj_pcm,'m')
% hold on
% plot(PROFILE_NO,diff_adj_pcm,'.m')
% 
% xlabel('Profile number')
% ylabel('Salinity [PSU]')
% title(['Differences between d-mode and pcm at the deepest level ',num2str(flt_name)])
% 
% 
% drawnow
% set(gcf,'papertype','usletter','paperunits','inches','paperorientation','portrait','paperposition',[.25,.5,8,10]);
% %print(['-depsc ','/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_comparison/ctd/',flt_name, '_PCM comparison.eps']);
% %print([gcf,'-depsc ','/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_comparison/ctd/',flt_name,'_DAC_vs_OWC.eps']);
% print(['/users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/data/float_comparison/ctd/',flt_name,'_DAC_vs_OWC_1000db'],'-depsc');

end
