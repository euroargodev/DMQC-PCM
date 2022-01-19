import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
import seaborn as sns

import numpy as np
import xarray as xr

from PIL import Image, ImageFont, ImageDraw

def plot_spatialdist_ds(ds, float_WMO, float_traj=False):
    """ Plot spatial distribution of the dataset

        Parameters
        ----------
        ds: dataset with lat and long variables
        float_WMO: float reference 
        float_traj: plot float trajectory (default: False)

        Returns
        -------
        Spatial distribution plot
    """
    
    proj=ccrs.PlateCarree()
    subplot_kw = {'projection': proj}
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(
            8, 8), dpi=120, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)

    p1 = ax.scatter(ds['long'], ds['lat'], s=3, transform=proj, label='Argo reference data')
    
    if float_traj:
        selected_float_index = [i for i, isource in enumerate(ds['source'].values) if 'selected_float' in isource]
        p2 = ax.plot(ds['long'].isel(n_profiles = selected_float_index), ds['lat'].isel(n_profiles = selected_float_index), 
                     'ro-', transform=proj, markersize = 3, label = str(float_WMO) + ' float trajectory')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    land_feature = cfeature.NaturalEarthFeature(
            category='physical', name='land', scale='50m', facecolor=[0.9375, 0.9375, 0.859375])
    ax.add_feature(land_feature, edgecolor='black')

    defaults = {'linewidth': .5, 'color': 'gray', 'alpha': 0.5, 'linestyle': '--'}
    gl = ax.gridlines(crs=ax.projection,draw_labels=True, **defaults)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180+1, 4))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 90+1, 4))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'fontsize': 8}
    gl.ylabel_style = {'fontsize': 8}
    gl.xlabels_top = False
    gl.ylabels_right = False
    lon_180 = np.mod((ds['long']+180),360)-180
    ax.set_xlim([lon_180.min()-1, lon_180.max()+1])
    ax.set_ylim([ds['lat'].min()-1, ds['lat'].max()+1])

    plt.draw()

class Plotter:
    '''New class for visualisation of data from pyxpcm

       Parameters
       ----------
           ds: dataset including PCM results
           m: pyxpcm model
           coords_dict: (optional) dictionary with coordinates names (ex: {'latitude': 'lat', 'time': 'time', 'longitude': 'lon'})
           cmap_name: (optional) colormap name (default: 'Accent')

           '''

    def __init__(self, ds, m, coords_dict=None, cmap_name='Accent'):


        self.ds = ds
        self.m = m
        if cmap_name == 'Accent' and self.m.K > 8:
            self.cmap_name = 'tab20'
        else:
            self.cmap_name = cmap_name

        # check if dataset should include PCM variables
        assert ("PCM_LABELS" in self.ds), "Dataset should include PCM_LABELS variable to be plotted. Use pyxpcm.predict function with inplace=True option"

        if coords_dict == None:
            # creates dictionary with coordinates
            coords_list = list(self.ds.coords.keys())
            coords_dict = {}
            for c in coords_list:
                axis_at = self.ds[c].attrs.get('axis')
                if axis_at == 'Y':
                    coords_dict.update({'latitude': c})
                if axis_at == 'X':
                    coords_dict.update({'longitude': c})
                if axis_at == 'T':
                    coords_dict.update({'time': c})
                if axis_at == 'Z':
                    coords_dict.update({'depth': c})

            self.coords_dict = coords_dict

            if 'latitude' not in coords_dict or 'longitude' not in coords_dict:
                raise ValueError(
                    'Coordinates not found in dataset. Please, define coordinates using coord_dict input')

        else:
            self.coords_dict = coords_dict

        # assign a data type
        dims_dict = list(ds.dims.keys())
        dims_dict = [e for e in dims_dict if e not in (
            'quantile', 'pcm_class')]
        if len(dims_dict) > 2:
            self.data_type = 'gridded'
        else:
            self.data_type = 'profiles'

    def pie_classes(self, save_fig=[]):
        """Pie chart of classes
        
           Parameters
           ----------
               save_fig: path to save figure (default: [])
               
           Returns
           ------
               pie chart figure

        """

        # loop in k for counting
        pcm_labels = self.ds['PCM_LABELS']
        kmap = self.m.plot.cmap(name=self.cmap_name)

        for cl in range(self.m.K):
            # get labels
            pcm_labels_k = pcm_labels.where(pcm_labels == cl)
            if cl == 0:
                counts_k = pcm_labels_k.count(...)
                pie_labels = list(['K=%i' % cl])
                table_cn = list([[str(cl), str(counts_k.values)]])
            else:
                counts_k = xr.concat([counts_k, pcm_labels_k.count(...)], "k")
                pie_labels.append('K=%i' % cl)
                table_cn.append([str(cl), str(counts_k[cl].values)])

        table_cn.append(['Total', str(sum([int(row[1]) for row in table_cn]))])

        fig, ax = plt.subplots(ncols=2, figsize=(10, 6))

        cheader = ['$\\bf{K}$', '$\\bf{Number\\ of\\ profiles}$']
        ccolors = plt.cm.BuPu(np.full(len(cheader), 0.1))
        the_table = plt.table(cellText=table_cn, cellLoc='center', loc='center',
                              colLabels=cheader, colColours=ccolors, fontsize=14, colWidths=(0.2, 0.45))

        the_table.auto_set_font_size(False)
        the_table.set_fontsize(12)

        explode = np.ones(self.m.K)*0.05
        kmap_n = [list(kmap(k)[0:3]) for k in range(self.m.K)]
        textprops = {'fontweight': "bold", 'fontsize': 12}

        _, _, autotexts = ax[0].pie(counts_k, labels=pie_labels, autopct='%1.1f%%',
                                    startangle=90, colors=kmap_n, explode=explode, textprops=textprops, pctdistance=0.5)

        #labels in white
        for autotext in autotexts:
            autotext.set_fontweight('normal')
            autotext.set_fontsize(10)

        # draw circle
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        fig = plt.gcf()
        ax[0].add_artist(centre_circle)

        ax[0].axis('equal')
        ax[1].get_xaxis().set_visible(False)
        ax[1].get_yaxis().set_visible(False)
        plt.box(on=None)
        the_table.scale(1, 1.5)
        fig.suptitle('$\\bf{Classes\\ distribution}$', fontsize=14)
        plt.tight_layout()
        #plt.show()
        
        if save_fig!=[]:
            plt.savefig(save_fig)

    @staticmethod
    def cmap_discretize(name, K):
        """Return a discrete colormap from a quantitative or continuous colormap name

            name: name of the colormap, eg 'Paired' or 'jet'
            K: number of colors in the final discrete colormap
        """
        if name in ['Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'Paired', 'Dark2', 'Accent']:
            # Segmented (or quantitative) colormap:
            N_ref = {'Set1': 9, 'Set2': 8, 'Set3': 12, 'Pastel1': 9,
                     'Pastel2': 8, 'Paired': 12, 'Dark2': 8, 'Accent': 8}
            N = N_ref[name]
            cmap = plt.get_cmap(name=name)
            colors_i = np.concatenate(
                (np.linspace(0, 1., N), (0., 0., 0., 0.)), axis=0)
            cmap = cmap(colors_i)  # N x 4
            n = np.arange(0, N)
            new_n = n.copy()
            if K > N:
                for k in range(N, K):
                    r = np.roll(n, -k)[0][np.newaxis]
                    new_n = np.concatenate((new_n, r), axis=0)
            new_cmap = cmap.copy()
            new_cmap = cmap[new_n, :]
            new_cmap = mcolors.LinearSegmentedColormap.from_list(
                name + "_%d" % K, colors=new_cmap, N=K)
        else:
            # Continuous colormap:
            N = K
            cmap = plt.get_cmap(name=name)
            colors_i = np.concatenate(
                (np.linspace(0, 1., N), (0., 0., 0., 0.)))
            colors_rgba = cmap(colors_i)  # N x 4
            indices = np.linspace(0, 1., N + 1)
            cdict = {}
            for ki, key in enumerate(('red', 'green', 'blue')):
                cdict[key] = [(indices[i], colors_rgba[i - 1, ki], colors_rgba[i, ki])
                              for i in np.arange(N + 1)]
            # Return colormap object.
            new_cmap = mcolors.LinearSegmentedColormap(
                cmap.name + "_%d" % N, cdict, N)
            
        return new_cmap

    def vertical_structure(self,
                           q_variable,
                           xlim=None,
                           classdimname='pcm_class',
                           quantdimname='quantile',
                           maxcols=3,
                           cmap=None,
                           ylabel='depth (m)',
                           xlabel='feature',
                           ylim='auto',
                           save_fig=[],
                           **kwargs):
        '''Plot vertical structure of each class

           Parameters
           ----------
               q_variable: quantile variable calculated with pyxpcm.quantile function (inplace=True option)
               xlim: (optional) x axis limits 
               classdimname: (optional) pcm classes dimension name (default = 'pcm_class')
               quantdimname: (optional) pcm quantiles dimension name (default = 'quantiles')
               maxcols: (optional) max number of column (default = 3)
               cmap: (optional) colormap name for quantiles (default = 'brg')
               ylabel: (optional) y axis label (default = 'depth (m)')
               xlabel: (optional) x axis label (default = 'feature')
               ylim: (optional) y axis limits (default = 'auto')
               save_fig: path to save figure (default: [])
               **kwargs

           Returns
           -------
               fig : :class:`matplotlib.pyplot.figure.Figure`
               ax : :class:`matplotlib.axes.Axes` object or array of Axes objects.
                    *ax* can be either a single :class:`matplotlib.axes.Axes` object or an
                    array of Axes objects if more than one subplot was created.  The
                    dimensions of the resulting array can be controlled with the squeeze
                    keyword.

               '''

        # select quantile variable
        da = self.ds[q_variable]

        # da must be 3D with a dimension for: CLASS, QUANTILES and a vertical axis
        # The QUANTILES dimension is called "quantile"
        # The CLASS dimension is identified as the one matching m.K length.
        if classdimname in da.dims:
            CLASS_DIM = classdimname
        elif (np.argwhere(np.array(da.shape) == self.m.K).shape[0] > 1):
            raise ValueError(
                "Can't distinguish the class dimension from the others")
        else:
            CLASS_DIM = da.dims[np.argwhere(
                np.array(da.shape) == self.m.K)[0][0]]
        QUANT_DIM = quantdimname
        VERTICAL_DIM = list(
            set(da.dims) - set([CLASS_DIM]) - set([QUANT_DIM]))[0]

        nQ = len(da[QUANT_DIM])  # Nb of quantiles

        # cmapK = self.m.plot.cmap()  # cmap_discretize(plt.cm.get_cmap(name='Paired'), m.K)
        cmapK = self.cmap_discretize(
            plt.cm.get_cmap(name=self.cmap_name), self.m.K)
        if not cmap:
            cmap = self.cmap_discretize(plt.cm.get_cmap(name='brg'), nQ)

        maxcols = 4
        fig_max_size = 2.5*self.m.K if self.m.K < maxcols else 10
        #fig_max_size = [2.5*self.m.K if self.m.K < maxcols else 10, 6*np.int(self.m.K/maxcols)]
        defaults = {'figsize': (fig_max_size, 8), 'dpi': 80,
                    'facecolor': 'w', 'edgecolor': 'k'}
        # defaults = {'figsize': fig_max_size, 'dpi': 80,
        #            'facecolor': 'w', 'edgecolor': 'k'}
        fig, ax = self.m.plot.subplots(
            maxcols=maxcols, **{**defaults, **kwargs})  # TODO: function in pyxpcm

        if not xlim:
            var_name = q_variable[0:-2]
            xlim = np.array([self.ds[var_name].min(), self.ds[var_name].max()])
        for k in self.m:
            Qk = da.loc[{CLASS_DIM: k}]
            for (iq, q) in zip(np.arange(nQ), Qk[QUANT_DIM]):
                Qkq = Qk.loc[{QUANT_DIM: q}]
                ax[k].plot(Qkq.values.T, da[VERTICAL_DIM], label=(
                    "%0.2f") % (Qkq[QUANT_DIM]), color=cmap(iq), linewidth=1.5)
            ax[k].set_title(("Component: %i") % (k), color=cmapK(k), fontsize=12)
            #ax[k].legend(loc='lower right')
            ax[k].set_xlim(xlim)
            if isinstance(ylim, str):
                ax[k].set_ylim(
                    np.array([da[VERTICAL_DIM].min(), da[VERTICAL_DIM].max()]))
            else:
                ax[k].set_ylim(ylim)
            # ax[k].set_xlabel(Q.units)
            if k == 0 or np.divmod(k, maxcols)[1] == 0:
                ax[k].set_ylabel(ylabel)
            ax[k].grid(True)

        ax[k].legend(bbox_to_anchor=(1.5, 1), loc='upper right')
        plt.subplots_adjust(top=0.90)
        fig.suptitle('$\\bf{Vertical\\ structure\\ of\\ classes}$', fontsize=12)
        fig_size = fig.get_size_inches()
        plt.draw()
        #fig.text((fig_size[0]/2)/fig_size[0], 1-(fig_size[1]-0.5)/fig_size[1], xlabel, va='center', fontsize=10)
        fig.text((fig_size[0]/2)/fig_size[0], 0.05,
                 xlabel, va='center', fontsize=10)
        # plt.tight_layout()
        plt.show()
        
        if save_fig!=[]:
            fig.savefig(save_fig, dpi=300)

    def vertical_structure_comp(self, q_variable,
                                plot_q='all',
                                xlim=None,
                                classdimname='pcm_class',
                                quantdimname='quantile',
                                maxcols=3, cmap=None,
                                ylabel='depth (m)',
                                xlabel='feature',
                                ylim='auto',
                                save_fig=[],
                                **kwargs):
        '''Plot vertical structure of each class

           Parameters
           ----------
               q_variable: quantile variable calculated with pyxpcm.quantile function (inplace=True option)
               plot_q: quantiles to be plotted
               xlim: (optional) x axis limits 
               classdimname: (optional) pcm classes dimension name (default = 'pcm_class')
               quantdimname: (optional) pcm quantiles dimension name (default = 'quantiles')
               maxcols: (optional) max number of column (default = 3)
               cmap: (optional) colormap name for quantiles (default = 'brg')
               ylabel: (optional) y axis label (default = 'depth (m)')
               xlabel: (optional) x axis label (default = 'feature')
               ylim: (optional) y axis limits (default = 'auto'
               save_fig: path to save figure (default: [])

           Returns
           ------
               fig : :class:`matplotlib.pyplot.figure.Figure`

               ax : :class:`matplotlib.axes.Axes` object or array of Axes objects.
                    *ax* can be either a single :class:`matplotlib.axes.Axes` object or an
                    array of Axes objects if more than one subplot was created.  The
                    dimensions of the resulting array can be controlled with the squeeze
                    keyword.

               '''

        # select quantile variable
        da = self.ds[q_variable]

        # da must be 3D with a dimension for: CLASS, QUANTILES and a vertical axis
        # The QUANTILES dimension is called "quantile"
        # The CLASS dimension is identified as the one matching m.K length.
        if classdimname in da.dims:
            CLASS_DIM = classdimname
        elif (np.argwhere(np.array(da.shape) == self.m.K).shape[0] > 1):
            raise ValueError(
                "Can't distinguish the class dimension from the others")
        else:
            CLASS_DIM = da.dims[np.argwhere(
                np.array(da.shape) == self.m.K)[0][0]]
        QUANT_DIM = quantdimname
        VERTICAL_DIM = list(
            set(da.dims) - set([CLASS_DIM]) - set([QUANT_DIM]))[0]

        nQ = len(da[QUANT_DIM])  # Nb of quantiles

        if isinstance(plot_q, str):  # plot all quantiles, default
            q_range = np.arange(0, nQ)
        else:
            q_range = np.where(da[QUANT_DIM].isin(plot_q))[0]

        nQ_p = len(q_range)  # Nb of plots

        # cmap_discretize(plt.cm.get_cmap(name='Paired'), m.K)
        cmapK = self.m.plot.cmap(name=self.cmap_name)
        #cmapK = self.cmap_discretize(plt.cm.get_cmap(name='Accent'), self.m.K)
        if not cmap:
            cmap = self.cmap_discretize(plt.cm.get_cmap(name='brg'), nQ)

        if not xlim:
            var_name = q_variable[0:-2]
            xlim = np.array([self.ds[var_name].min(), self.ds[var_name].max()])

        maxcols = 4
        fig_max_size = 2.5*nQ_p if nQ_p < maxcols else 10
        defaults = {'figsize': (fig_max_size, 8), 'dpi': 80,
                    'facecolor': 'w', 'edgecolor': 'k'}
        fig, ax = self.m.plot.subplots(
            maxcols=maxcols, K=nQ_p, **defaults, sharey=True,  squeeze=False)

        cnt = 0
        for q in q_range:
            Qq = da.loc[{QUANT_DIM: da[QUANT_DIM].values[q]}]
            for k in self.m:
                Qqk = Qq.loc[{CLASS_DIM: k}]
                ax[cnt].plot(Qqk.values.T, da[VERTICAL_DIM], label=(
                    "K=%i") % (Qqk[CLASS_DIM]), color=cmapK(k), linewidth=1.5)
            ax[cnt].set_title(("quantile: %.2f") % (
                da[QUANT_DIM].values[q]), color=cmap(q), fontsize=12)
            #ax[cnt].legend(loc='lower right', fontsize=11)
            ax[cnt].set_xlim(xlim)

            if isinstance(ylim, str):
                ax[cnt].set_ylim(
                    np.array([da[VERTICAL_DIM].min(), da[VERTICAL_DIM].max()]))
            else:
                ax[cnt].set_ylim(ylim)
            # ax[k].set_xlabel(Q.units)
            if cnt == 0:
                ax[cnt].set_ylabel(ylabel)
            ax[cnt].grid(True)
            cnt = cnt+1

        lgd = ax[cnt-1].legend(bbox_to_anchor=(1.6, 1), loc='upper right', fontsize=10)
        plt.subplots_adjust(top=0.90)
        plt.rc('xtick', labelsize=10)
        plt.rc('ytick', labelsize=10)
        supt = fig.suptitle('$\\bf{Vertical\\ structure\\ of\\ classes}$', fontsize=12)
        fig_size = fig.get_size_inches()
        #plt.draw()
        fig.text((fig_size[0]/2)/fig_size[0], 0.05, xlabel, va='center', fontsize=10)
        # fig.text(0.04, 0.5, 'depth (m)', va='center',
        #         rotation='vertical', fontsize=12)
        #plt.tight_layout()
        #plt.show()
        
        if save_fig!=[]:
            plt.savefig(save_fig, bbox_extra_artists=(lgd, supt,), bbox_inches='tight', dpi=300)

    def spatial_distribution(self, proj=ccrs.PlateCarree(), extent='auto', time_slice=0, lonlat_grid =[4,4], float_traj=False, save_fig=[]):
        '''Plot spatial distribution of classes

           Parameters
           ----------
               proj: projection
               extent: map extent
               time_slice: time snapshot to be plot (default 0). If time_slice = 'most_freq_label', most frequent label in dataseries is plotted.
                        most_freq_label option can only be used with gridded data
               lonlat_grid: space between lon lat ticks (default: [4,4])
               float_traj: plot float trajectory (default: False)
               save_fig: path to save figure (default: [])

           Returns
           -------
               fig : :class:`matplotlib.pyplot.figure.Figure`

               ax : :class:`matplotlib.axes.Axes` object or array of Axes objects.
                    *ax* can be either a single :class:`matplotlib.axes.Axes` object or an
                    array of Axes objects if more than one subplot was created.  The
                    dimensions of the resulting array can be controlled with the squeeze
                    keyword.

               '''

        def get_most_freq_labels(this_ds):
            this_ds = this_ds.stack(
                {'N_OBS': [d for d in this_ds['PCM_LABELS'].dims if d != 'time']})

            def fct(this):
                def most_prob_label(vals):
                    return np.argmax(np.bincount(vals))
                mpblab = []
                for i in this['N_OBS']:
                    val = this.sel(N_OBS=i)['PCM_LABELS'].values
                    res = np.nan
                    if np.count_nonzero(~np.isnan(val)) != 0:
                        res = most_prob_label(val.astype('int'))
                    mpblab.append(res)
                mpblab = np.array(mpblab)
                return xr.DataArray(mpblab, dims='N_OBS', coords={'N_OBS': this['N_OBS']}, name='PCM_MOST_FREQ_LABELS').to_dataset()
            this_ds['PCM_MOST_FREQ_LABELS'] = this_ds.map_blocks(
                fct)['PCM_MOST_FREQ_LABELS'].load()
            return this_ds.unstack('N_OBS')
        
        # lontitud from 0-360 to -180+180
        long_data = np.mod((self.ds[self.coords_dict.get('longitude')]+180),360)-180

        # spatial extent
        if isinstance(extent, str):
            extent = np.array([min(long_data), max(long_data), min(
                self.ds[self.coords_dict.get('latitude')]), max(self.ds[self.coords_dict.get('latitude')])]) 
            #+ np.array([-0.1, +0.1, -0.1, +0.1])

        if time_slice == 'most_freq_label':
            dsp = get_most_freq_labels(self.ds)
            var_name = 'PCM_MOST_FREQ_LABELS'
            title_str = '$\\bf{Spatial\\ ditribution\\ of\\ classes}$' + \
                ' \n (most frequent label in time series)'
        else:
            if 'time' in self.coords_dict and self.data_type == 'gridded':
                dsp = self.ds.sel(time=time_slice, method='nearest').squeeze()
                title_str = '$\\bf{Spatial\\ ditribution\\ of\\ classes}$' + \
                    ' \n (time: ' + \
                    '%s' % dsp["time"].dt.strftime(
                        "%Y/%m/%d %H:%M").values + ')'
            else:
                dsp = self.ds
                title_str = '$\\bf{Spatial\\ ditribution\\ of\\ classes}$'
            var_name = 'PCM_LABELS'

        subplot_kw = {'projection': proj, 'extent': extent}
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(
            10, 10), dpi=120, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)
        # TODO: function already in pyxpcm
        kmap = self.m.plot.cmap(name=self.cmap_name)

        # check if gridded or profiles data
        if self.data_type == 'profiles':
            sc = ax.scatter(long_data, dsp[self.coords_dict.get('latitude')], s=4,
                            c=self.ds[var_name], cmap=kmap, transform=proj, vmin=0, vmax=self.m.K, zorder=1)
            selected_float_index = [i for i, isource in enumerate(dsp['source'].values) if 'selected_float' in isource]
        if self.data_type == 'gridded':
            sc = ax.pcolormesh(long_data, dsp[self.coords_dict.get(
                'latitude')], dsp[var_name], cmap=kmap, transform=proj, vmin=0, vmax=self.m.K)
        
        # cycle number in float trajectory
        if float_traj:
            float_source = dsp['source'].isel(n_profiles = selected_float_index)
            float_cycles = [int(float_source.values[i].lstrip('selected_float_')) for i in range(len(float_source))]
            cycles_labels = np.arange(10,float_cycles[-1]+1,10)
            
            p2 = ax.plot(dsp[self.coords_dict.get('longitude')].isel(n_profiles = selected_float_index), 
                         dsp[self.coords_dict.get('latitude')].isel(n_profiles = selected_float_index), 
                         'ko', markerfacecolor="None", transform=proj, markersize = 4, zorder=2)
        
            p2 = ax.plot(np.mod((dsp[self.coords_dict.get('longitude')].isel(n_profiles = selected_float_index)+180),360)-180, 
                         dsp[self.coords_dict.get('latitude')].isel(n_profiles = selected_float_index), 
                         'k-', markerfacecolor="None", transform=proj, markersize = 4, zorder=-1)
            
            transform = ccrs.PlateCarree()._as_mpl_transform(ax)
            for icycle in cycles_labels:
                prof_data = dsp['source'].where(dsp['source'] == 'selected_float_' + str(icycle), drop=True)
                if np.size(prof_data['n_profiles'].values) == 0:
                    continue
                cycle_index = prof_data['n_profiles'].values[0]
                lat_value = prof_data['lat'].values[0]
                long_value = prof_data['long'].values[0]
                ax.annotate(str(icycle), xy=(long_value+0.07, lat_value+0.07), xycoords=transform, fontsize=7, weight='bold')

        cbar = plt.colorbar(sc, shrink=0.3)
        cbar.set_ticks(np.arange(0.5, self.m.K+0.5))
        cbar.set_ticklabels(range(self.m.K))

        ax.set_xticks(np.arange(int(extent[0]), int(
            extent[1]+1), lonlat_grid[0]), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(int(extent[2]), int(
            extent[3]+1), lonlat_grid[1]), crs=ccrs.PlateCarree())
        lon_formatter = LongitudeFormatter()
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)
        plt.grid(True,  linestyle='--')
        cbar.set_label('Class', fontsize=12)
        ax.tick_params(axis="x", labelsize=8)
        ax.tick_params(axis="y", labelsize=8)

        land_feature = cfeature.NaturalEarthFeature(
            category='physical', name='land', scale='50m', facecolor=[0.9375, 0.9375, 0.859375])
        ax.add_feature(land_feature, edgecolor='black')
        title_str = '$\\bf{Spatial\\ ditribution\\ of\\ classes\\ (CTD\\ database)}$'
        ax.set_title(title_str)
        fig.canvas.draw()
        fig.tight_layout()
        #plt.margins(0.1)
        
        if save_fig!=[]:
            plt.savefig(save_fig, bbox_inches='tight', dpi=300)


    def temporal_distribution(self, time_bins, start_month=0, save_fig=[]):
        '''Plot temporal distribution of classes by moth or by season

           Parameters
           ----------
                time_bins: 'month' or 'season'
                start_month: (optional) start plot in this month (index from 1:Jan to 12:Dec)
                save_fig: path to save figure (default: [])

            Returns
            -------
               fig : :class:`matplotlib.pyplot.figure.Figure`

               ax : :class:`matplotlib.axes.Axes` object or array of Axes objects.
                    *ax* can be either a single :class:`matplotlib.axes.Axes` object or an
                    array of Axes objects if more than one subplot was created.  The
                    dimensions of the resulting array can be controlled with the squeeze
                    keyword.

        '''

        # check if more than one temporal step
        assert (len(self.ds[self.coords_dict.get('time')]) >
                1), "Length of time variable should be > 1"

        # data to be plot
        pcm_labels = self.ds['PCM_LABELS']
        kmap = self.m.plot.cmap(name=self.cmap_name)

        if time_bins == 'month':
            xaxis_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May',
                            'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            if start_month != 0:
                new_order = np.concatenate(
                    (np.arange(start_month, 13), np.arange(1, start_month)))
                xaxis_labels = [xaxis_labels[i-1] for i in new_order]
        if time_bins == 'season':
            seasons_dict = {1: 'DJF', 2: 'MAM', 3: 'JJA', 4: 'SON'}
            xaxis_labels = ['DJF', 'MAM', 'JJA', 'SON']

        fig, ax = plt.subplots(figsize=(10, 6))

        # loop in k for counting
        for cl in range(self.m.K):
            # get time array with k=cl
            pcm_labels_k = pcm_labels.where(pcm_labels == cl)

            if cl == 0:
                counts_k = pcm_labels_k.groupby(
                    self.coords_dict.get('time') + '.' + time_bins).count(...)
            else:
                counts_k = xr.concat([counts_k, pcm_labels_k.groupby(
                    self.coords_dict.get('time') + '.' + time_bins).count(...)], "k")

        counts_k = counts_k/sum(counts_k)*100
        # change order
        if start_month != 0:
            counts_k = counts_k.reindex({'month': new_order})

        # start point in stacked bars
        counts_cum = counts_k.cumsum(axis=0)

        # loop for plotting
        for cl in range(self.m.K):

            if time_bins == 'month':
                x_ticks_k = [xaxis_labels[i] for i in counts_k.month.values -1]
                starts = counts_cum.isel(k=cl) - counts_k.isel(k=cl)
                #ax.barh(counts_k.month, counts_k.isel(k=cl), left=starts, color=kmap(cl), label='K=' + str(cl))
                #ax.barh(x_ticks_k, counts_k.isel(k=cl), left=starts,
                #        color=kmap(cl), label='K=' + str(cl))
                ax.barh(counts_k.month.values-1, counts_k.isel(k=cl), left=starts,
                        color=kmap(cl), label='K=' + str(cl))

            if time_bins == 'season':
                x_ticks_k = []
                for i in range(len(counts_k.season)):
                    x_ticks_k.append(
                        list(seasons_dict.values()).index(counts_k.season[i])+1)
                    # print(x_ticks_k)
                # plot
                starts = counts_cum.isel(k=cl) - counts_k.isel(k=cl)
                ax.barh(x_ticks_k, counts_k.isel(k=cl), left=starts, label='K=' + str(cl),
                        color=kmap(cl))

        # format
        title_string = r'Percentage of profiles in each class by $\bf{' + time_bins + '}$'
        ylabel_string = '% of profiles'
        plt.gca().invert_yaxis()
        if time_bins == 'season':
            ax.set_yticks(np.arange(1, len(xaxis_labels)+1))
        else:
            ax.set_yticks(np.arange(len(xaxis_labels)))
        ax.set_yticklabels(xaxis_labels, fontsize=12)
        plt.yticks(fontsize=12)
        ax.legend(fontsize=12, bbox_to_anchor=(1.01, 1), loc='upper left')
        ax.set_xlabel(ylabel_string, fontsize=12)
        ax.set_title(title_string, fontsize=14)
        fig.tight_layout()
        
        if save_fig!=[]:
            plt.savefig(save_fig, dpi=300)
        
    def float_traj_classes(self, save_fig=[]):
        '''Plot float trajectory with float profile classes 

           Parameters
           ----------
                save_fig: path to save figure (default: [])

            Returns
            -------


        '''
        
        selected_float_index = [i for i, isource in enumerate(self.ds['source'].values) if 'selected_float' in isource]

        kmap = self.m.plot.cmap(name=self.cmap_name)

        proj=ccrs.PlateCarree()
        subplot_kw = {'projection': proj}
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(
                    8, 8), dpi=120, facecolor='w', edgecolor='k', subplot_kw=subplot_kw)

        p1 = ax.plot(self.ds['long'].isel(n_profiles = selected_float_index), self.ds['lat'].isel(n_profiles = selected_float_index), 
                         'k-', transform=proj, zorder=-1)

        p2 = ax.scatter(self.ds['long'].isel(n_profiles = selected_float_index), self.ds['lat'].isel(n_profiles = selected_float_index),
                        s=20, c=self.ds['PCM_LABELS'].isel(n_profiles = selected_float_index), cmap=kmap, transform=proj, vmin=0, vmax=self.m.K, zorder=1)
        
        # plot cycle number
        float_source = self.ds['source'].isel(n_profiles = selected_float_index)
        float_cycles = [int(float_source.values[i].lstrip('selected_float_')) for i in range(len(float_source))]
        cycles_labels = np.arange(10,float_cycles[-1]+1,10)
        transform = ccrs.PlateCarree()._as_mpl_transform(ax)
        for icycle in cycles_labels:
            prof_data = self.ds['source'].where(self.ds['source'] == 'selected_float_' + str(icycle), drop=True)
            if np.size(prof_data['n_profiles'].values) == 0:
                continue
            cycle_index = prof_data['n_profiles'].values[0]
            lat_value = prof_data['lat'].values[0]
            long_value = prof_data['long'].values[0]
            ax.annotate(str(icycle), xy=(long_value+0.07, lat_value+0.07), xycoords=transform, fontsize=7, weight='bold')

        land_feature = cfeature.NaturalEarthFeature(
                    category='physical', name='land', scale='50m', facecolor=[0.9375, 0.9375, 0.859375])
        ax.add_feature(land_feature, edgecolor='black')

        defaults = {'linewidth': .5, 'color': 'gray', 'alpha': 0.5, 'linestyle': '--'}
        gl = ax.gridlines(crs=ax.projection,draw_labels=True, **defaults)
        gl.xlocator = mticker.FixedLocator(np.arange(-180, 180+1, 4))
        gl.ylocator = mticker.FixedLocator(np.arange(-90, 90+1, 4))
        gl.xformatter = LONGITUDE_FORMATTER
        gl.yformatter = LATITUDE_FORMATTER
        gl.xlabel_style = {'fontsize': 8}
        gl.ylabel_style = {'fontsize': 8}
        gl.xlabels_top = False
        gl.ylabels_right = False
        lon_180 = np.mod((self.ds['long'].isel(n_profiles = selected_float_index)+180),360)-180
        ax.set_xlim([lon_180.min()-1, lon_180.max()+1])
        ax.set_ylim([self.ds['lat'].isel(n_profiles = selected_float_index).min()-1, self.ds['lat'].isel(n_profiles = selected_float_index).max()+1])

        cbar = plt.colorbar(p2, shrink=0.3, pad=0.02)
        cbar.set_ticks(np.arange(0.5, self.m.K+0.5))
        cbar.set_ticklabels(range(self.m.K))
        cbar.ax.tick_params(labelsize=10)
        cbar.set_label('Class', fontsize=10)
        
        title_string = 'Float profile classes'
        ax.set_title(title_string, fontsize=12)
        
        if save_fig!=[]:
            plt.savefig(save_fig, bbox_inches='tight', dpi=300)
        
        
    def float_cycles_prob(self, var_name='PCM_POST', save_fig=[]):
        '''Plot posterior or robustness of float profiles 

           Parameters
           ----------
                var_name: probabilité variable name (default: 'PCM_POST')
                save_fig: path to save figure (default: [])

            Returns
            -------


        '''
        
        selected_float_index = [i for i, isource in enumerate(self.ds['source'].values) if 'selected_float' in isource]
        float_labels = self.ds['PCM_LABELS'].isel(n_profiles = selected_float_index)
        
        if 'ROBUSTNESS' in var_name: 
            float_prob = self.ds[var_name].isel(n_profiles = selected_float_index)
        else:
            float_prob = self.ds[var_name].isel(n_profiles = selected_float_index, pcm_class = float_labels.astype(np.int16))
            
        float_source = self.ds['source'].isel(n_profiles = selected_float_index)
        float_cycles = [int(float_source.values[i].lstrip('selected_float_')) for i in range(len(float_source))]
        cycles_labels = np.arange(10,float_cycles[-1],10)

        kmap = self.m.plot.cmap(name=self.cmap_name)

        fig, ax = plt.subplots(figsize=(18, 5))

        rects1 = ax.plot(float_cycles, float_prob, '-', zorder=-1)
        rects2 = ax.scatter(float_cycles, float_prob, s=40, c=float_labels, cmap=kmap, zorder=1, vmin=0, vmax=self.m.K)

        if 'ROBUSTNESS' in var_name: 
            ax.set_ylabel('Robustness', fontsize=16)
            ax.set_yticks(np.arange(1,5+1))
            ax.set_yticklabels(self.ds['PCM_ROBUSTNESS_CAT'].attrs['legend'], fontsize=14)
            ax.set_title('Robustness of each float profile', fontsize=16)
        else:
            ax.set_ylabel('Posteriors', fontsize=16)
            ax.set_title('Probability of a profile to be in its class', fontsize=16)

        ax.set_xlabel('Float profile number', fontsize=16)
        ax.set_xticks(cycles_labels)
        ax.set_xlim([0.5, float_cycles[-1]+0.5])

        ax.tick_params(axis='both', which='major', labelsize=14)        

        cbar = plt.colorbar(rects2, shrink=0.6, pad=0.02)
        cbar.set_ticks(np.arange(0.5, self.m.K+0.5))
        cbar.set_ticklabels(range(self.m.K))
        cbar.ax.tick_params(labelsize=16)
        cbar.set_label('Class', fontsize=16)
        
        if save_fig!=[]:
            plt.savefig(save_fig, bbox_inches='tight', dpi=300)