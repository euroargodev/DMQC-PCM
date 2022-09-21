
%
% have to edit "ow_config.txt" if this system is moved somewhere else.
% ----------------------------------------------------------------------
%
% script ow_calibration
%
% dir names and float names have to correspond,
% e.g. float_dirs={'sio/';'uw/'}
%      float_names={'R49000139';'R39033'};
%
% these variables have to be set before ow_calibration is called.
%

%diary '/home1/homedir5/perso/agarciaj/EARISE/DMQC-PCM/examples/matlab_output/matlab_output_3901915_CTD_class.txt'
addpath /users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/matlab_codes
addpath /users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/lib/seawater_330_its90
addpath /users/argo/dm_qc/SO_assesment/DMQC-PCM-main/OWC-pcm/matlabow/lib/m_map1.4

float_dirs={'/', '/', '/', '/'}
float_names={'5902065'}

config_files={'ow_config_linux_ctd_argo.txt'};

%%
for i=1:length(float_names)

    flt_dir = float_dirs{i};
    flt_dir = deblank(flt_dir);
    flt_name = float_names{i};
    
  for j=1:length(config_files)
    %tic
  lo_system_configuration = load_configuration( config_files{j} );
  disp([datestr(now) ' Working on ' flt_name])
     
  update_salinity_mapping( flt_dir, flt_name, lo_system_configuration );

  set_calseries( flt_dir, flt_name, lo_system_configuration );

  calculate_piecewisefit( flt_dir, flt_name, lo_system_configuration );

  plot_diagnostics_ow( flt_dir, flt_name, lo_system_configuration );

  end
end
