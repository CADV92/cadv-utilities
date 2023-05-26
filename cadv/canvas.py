import os
import numpy as np
import cartopy, cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from PIL import Image

class Canvas:
    def __init__(self, extent=None, projection=ccrs.PlateCarree(), central_longitude=0, darkStyle=False, grid=True, cgrid=False):
        self.extent = extent
        self.projection = projection
        self.central_longitude = central_longitude
        self.darkStyle = darkStyle
        self.maincolor, self.labelcolor = ('white', 'black') if not self.darkStyle else ('black', 'white')
        self.grid = grid
        self.cgrid = cgrid
        self.frame(self.extent, central_longitude=self.central_longitude)

    def frame(self, extent, figsize=(10,8), central_longitude=None):
        if extent is not None:
            xfig, yfig = self.__proportion(*extent)
            yfig *= 1.12
            figsize = (xfig, yfig)

        self.__fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=200)
        
        glDelta = 0.04 if self.grid else 0
        # self.__proportion_extent = (self.extent[3]-self.extent[2])/(self.extent[1]-self.extent[0])
        # if self.__proportion_extent<0.5:
        #     _dy_axes = [0.03, 0.92, 0.05]
        # else:
        #     _dy_axes = [0.03, 0.92, 0.05]
        _dy_axes = [0.03, 0.92, 0.05]

        if central_longitude is not None:
            self.__ax_main = self.__fig.add_axes([0+glDelta, _dy_axes[0]+glDelta, 1-glDelta, _dy_axes[1]-glDelta], projection=ccrs.PlateCarree(central_longitude=central_longitude))
        else:
            self.__ax_main = self.__fig.add_axes([0, _dy_axes[0], 1, _dy_axes[1]], crs=self.projection)
        
        if extent is not None:
            self.__ax_main.set_extent(extent, crs=self.projection)
        
        _ydelta = self.__deltatick(self.extent[2],self.extent[3], n=6)
        _xdelta = self.__deltatick(self.extent[0],self.extent[1], n=6)
        
        if _xdelta <= 0:
            _xdelta = 1
        self.__xlocs = np.arange(-180,180,_xdelta)
        if _ydelta <= 0:
            _ydelta = 1
        self.__ylocs = np.arange(-90,90,_ydelta)

        if self.grid or self.cgrid:
            gl = self.__ax_main.gridlines(
                crs=ccrs.PlateCarree(), color='gray', linestyle='--', linewidth=0.25, alpha=0.4,
                xlocs=self.__xlocs, ylocs=self.__ylocs)
            # gl.top_labels = False
            # gl.right_labels = False
            gl.left_labels = True
            gl.bottom_labels = True
            gl.ylabel_style = {'color': 'k', 'weight': 'bold', 'size': 7.5, 'verticalalignment': 'center', 'bbox': dict(boxstyle="round", pad=0.3, fc="red", ec="gray", lw=0.5)}
            gl.xlabel_style = {'color': 'k', 'weight': 'bold', 'size': 7.5, 'bbox': {'facecolor':'green',
               'alpha':1, 'pad':0}}
            if self.cgrid:
                gl.ypadding = -6
                gl.xpadding = -6

        # Header
        self.__ax_header = self.__fig.add_axes([0.01, 1-_dy_axes[2], 0.98, _dy_axes[2]], facecolor=self.maincolor, frameon=False)
        self.__ax_header.set_axis_off()

        # Footer
        self.__ax_footer = self.__fig.add_axes([0+glDelta, 0, 1-glDelta, _dy_axes[0]])
        self.__ax_footer.set_visible(True)
        # self.__ax_bottom_frame = self.__fig.add_axes([0, 0.0, 1, 0.03])
        # self.__ax_bottom_frame.set_visible(True)
        self.__ax_footer.set_axis_off()
        self.manual_tick_axes()
    
    def __deltatick(self, min, max, n=5):
        delta = (max-min)/n
        if delta < 10:
            delta = int(delta)
        else:
            delta = (delta //10) * 10
        return delta
    
    def __proportion(self, loni, lonf, lati, latf):
        dlon = lonf-loni
        dlat = latf-lati
        max = dlon if dlon>dlat else dlat
        dy = dlat*10/max
        dx = dlon*10/max
        return dx, dy
    
    def __check_extent(self):
        extent = self.__subextent if hasattr(self, 'extent') else self.extent
        return extent

    def set_extent(self, extent, **kwargs):
        # self.frame(extent, central_longitude=self.central_longitude)
        return self.__ax_main.set_extent(extent, **kwargs)

    def imshow(self, data, extent, **kwargs):
        return self.__ax_main.imshow(data, extent=extent, **kwargs)

    def pcolormesh(self, x, y, data, **kwargs):
        return self.__ax_main.pcolormesh(x, y, data, **kwargs)
    
    def contour(self, lons, lats, data, fill=False, **kwargs):
        from cartopy.util import add_cyclic_point
        dlon = (lons[1]-lons[0])
        data, lons = add_cyclic_point(data, coord=np.arange(lons[0],lons[-1]+dlon*0.5, dlon))
        if 'transform' in kwargs:
            self.__transform = kwargs['transform']
        if fill:
            return self.__ax_main.contourf(lons, lats, data, **kwargs)
        else:
            return self.__ax_main.contour(lons, lats, data, **kwargs)
        
    def clevels(self, obj, levels, **kwargs):
        if hasattr(self, '_Canvas__transform'):
            return self.__ax_main.contour(obj, levels=levels, transform=self.__transform, **kwargs)
        else:
            return self.__ax_main.contour(obj, levels=levels, **kwargs)
    
    def clabels(self, obj, **kwargs):
        kwargs['fontsize'] = kwargs['fontsize'] if 'fontsize' in kwargs else self.scalling_value(0.5)
        return self.__ax_main.clabel(obj, **kwargs)
    
    def colorbar(self, image, **kwargs):
        self.__ax_footer.set_axis_on()
        cbar = plt.colorbar(image, cax=self.__fig.add_axes(self.__ax_footer), orientation='horizontal', pad=0.05, fraction=0.5, aspect=100, **kwargs)
        cbar.draw_all()
        cbar.ax.tick_params(labelcolor=self.labelcolor, direction='in', pad=1.5)
        if 'ticks' in kwargs:
            cbar.ax.set_xticklabels(kwargs['ticks'], fontsize=8, weight='bold')
        return cbar

    def legend_palette(self, colors=None, names=None, **kwargs):
        import matplotlib.patches as mpatches
        self.__ax_footer.axis('off')
        squares = [mpatches.Patch(facecolor=color, label=name, edgecolor="#fff8", linewidth=0.4) for color, name in zip(colors, names)]
        self.__ax_footer.legend(handles=squares, ncol=4, loc='center', framealpha=0, bbox_to_anchor=(0, 0, 1, 1), **kwargs)
    
    def title(self, title, **kwargs):
        img_size = self.__fig.get_size_inches()*self.__fig.dpi

        bbox = self.__ax_header.get_window_extent()
        proportion = self.scalling_value(1)
        
        size = kwargs['size']*proportion if 'size' in kwargs else proportion
        weight = kwargs['weight'] if 'weight' in kwargs else 'normal'
        fontname = kwargs['fontname'] if 'fontname' in kwargs else 'DejaVu Sans'
        align = kwargs['align'] if 'align' in kwargs else 'center'
        loc = kwargs['loc'] if 'loc' in kwargs else 'c'

        # Define locs
        if loc == 'c':
            x, y = (0.5,0.5)
        else:
            _loc = {'r': np.array([0.5,0]),
                    't': np.array([0,0.25]),
                    'c': np.array([0.5,0.5])}
            _loc['l'], _loc['b'] = -_loc['r'], -_loc['t']
            x, y = np.sum([_loc[i] for i in loc], axis=0)+_loc['c']

        self.__ax_header.text(x, y, title, horizontalalignment=align,
            verticalalignment='center', rotation=0, fontsize=size, style='normal',
            weight=weight, color=self.labelcolor, fontname=fontname)
    
    def scalling_value(self, value):
        ancho, alto = self.__fig.get_size_inches()*self.__fig.dpi
        relacion = ancho / alto
        if relacion < 1:
            proporcion = 0.008 * alto
        elif relacion > 1 and relacion <2:
            proporcion = 0.012 * alto
        elif relacion >= 2:
            proporcion = 0.016 * alto
        else:
            proporcion = 0.008 * (ancho + alto) / 2
        return proporcion*value*3/4

    def add_logo(self, logo):
        logo = Image.open(logo)
        # Obtener el tamaño de la imagen
        img_size = self.__fig.get_size_inches()*self.__fig.dpi

        Percent = (img_size[1]*self.__ax_header.get_position().height/logo.size[1])
        
        LogoWidth = int(float(logo.size[0])*float(Percent))
        LogoHeight = int(float(logo.size[1])*float(Percent))
        logo = logo.resize((LogoWidth, LogoHeight), Image.BILINEAR)
        LogoHeight = float(logo.size[1])

        offset = 1 if self.grid else 0.3
        ylocation = img_size[1]*(self.__ax_footer.get_position().height + self.__ax_main.get_position().height) + LogoHeight*offset

        xlocation = img_size[0]*self.__ax_main.get_position().xmin*0.8

        self.__fig.figimage(logo, xlocation, ylocation+31, origin='upper', zorder=3)
    
    def add_shp(self, shpfile, points=False, **kwargs):
        width = kwargs['width'] if 'width' in kwargs else 0.5
        alpha = kwargs['alpha'] if 'alpha' in kwargs else 1
        filter = kwargs['filter'] if 'filter' in kwargs else None 

        lcolor = kwargs['lcolor']  if 'lcolor' in kwargs else 'none'
        fcolor = kwargs['fcolor']  if 'fcolor' in kwargs else 'none'

        if not points:
            shape_feature = shpreader.Reader(shpfile).geometries()
            self.__ax_main.add_geometries(shape_feature, self.projection, edgecolor=lcolor, facecolor=fcolor, linewidth=width, alpha=alpha)
        else:
            shape_feature = shpreader.Reader(shpfile)
            for record in shape_feature.records():
                lon, lat = record.geometry.coords[0]
                extent = self.__check_extent()
                if (lon<extent[1] and lon>extent[0]) and (lat<extent[3] and lat>extent[2]):
                    self.__ax_main.plot(lon, lat, marker='.', color='m', markersize=3, markeredgecolor='m', transform=self.projection)
                    self.__ax_main.plot(lon, lat, marker='+', color='m', markersize=3, markeredgecolor='m', transform=self.projection)
    
    def border(self, color='black', width=0.2, fill=True):
        # Add coastlines, borders and gridlines
        # self.__ax_main.outline_patch.set_edgecolor(color)
        # self.__ax_main.gridlines(color=color, alpha=0.5, linestyle='--', linewidth=1.5)
        if fill:
            self.__ax_main.add_feature(cartopy.feature.LAND, edgecolor='black', facecolor='#ababaa', zorder=4, linewidth=width*0.75)
        self.__ax_main.add_feature(cartopy.feature.BORDERS, edgecolor=color, linewidth=width, zorder=4)
        self.__ax_main.coastlines(resolution='10m', color=color, linewidth=width, zorder=4)
        # import cartopy.io.img_tiles as cimgt
        # background = cimgt.Stamen('terrain-background')
        # self.__ax_main.add_image(background, 8)

    def savefig(self, name, path='.', compress=False, **kwargs):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        plt.rcParams['savefig.facecolor'] = self.maincolor
        plt.rcParams['figure.facecolor'] = self.maincolor
        plt.rcParams['axes.facecolor'] = self.maincolor

        self.__ax_header.set_alpha(None)
        self.__ax_main.set_alpha(None)
        self.__ax_footer.set_alpha(None)

        plt.savefig(os.path.join(path,name), bbox_inches='tight',pad_inches=0, **kwargs)
        plt.clf()
        plt.close('all')

        if compress:
            os.system(f'pngquant --quality=75-90 --strip {os.path.join(path,name)} -o {os.path.join(path,name)} --force')
        return os.path.join(path,name)

    def save_img(self, filename, **kwargs):
        # if os.path.dirname(filename) or not os.path.isdir(os.path.dirname(filename)):
        #     os.makedirs(os.path.dirname(filename), exist_ok=True)

        plt.rcParams['savefig.facecolor'] = self.maincolor
        plt.rcParams['figure.facecolor'] = self.maincolor
        plt.rcParams['axes.facecolor'] = self.maincolor

        self.__ax_header.set_alpha(None)
        self.__ax_main.set_alpha(None)
        self.__ax_footer.set_alpha(None)
        savefig = plt.savefig(filename, bbox_inches='tight',pad_inches=0, **kwargs)
        plt.clf()
        plt.close('all')
        return savefig
        
    def legend_lines(self,labels,colors,style='square', **kwargs):
        import matplotlib.lines as mlines
        import matplotlib.patches as mpatches
        patches = []
        for patch in zip(labels,colors):
            if style=='square':
                patches.append(mpatches.Rectangle((0, 0), 10, 10, linewidth=0, facecolor=patch[1], label=patch[0]))
            else:
                patches.append(mlines.Line2D([], [], color=patch[1], label=patch[0]))
        legend = self.__ax_header.legend(handles=patches, fontsize=self.scalling_value(0.6),
                                        facecolor='w', labelcolor='k',
                                        framealpha=0.8, loc='best', bbox_to_anchor=(0.81, 0, 0.2, -0.2), borderpad=0.6, **kwargs)
        legend.get_title().set_color('k')
        legend.get_title().set_fontsize(self.scalling_value(0.7))
        legend.get_title().set_fontweight('bold')
        legend.get_title().set_fontstyle('italic')

    def text(self, lon, lat, string, **kwargs):
        loni = lon + 360 - self.central_longitude
        if loni>180:
            loni-=360
        # print(lon)
        return self.__ax_main.annotate(string, xy=(loni,lat), **kwargs)
    
    def rectangle(self, extent, **kwargs):
        # check area
        if self.extent[0]<=extent[0] and self.extent[1]>=extent[2] and self.extent[2]<=extent[1] and self.extent[3]>=extent[3]:
            import matplotlib.patches as mpatches
            lstyle = kwargs['lstyle'] if 'lstyle' in kwargs else "-"
            lwidth = kwargs['lwidth'] if 'lwidth' in kwargs else 0.5
            lcolor = kwargs['lcolor'] if 'lcolor' in kwargs else 'k'
            self.__ax_main.add_patch(mpatches.Rectangle(xy=[extent[0],extent[1]],
                                                    width=extent[2]-extent[0],
                                                    height=extent[3]-extent[1],
                                                    # alpha=0.2,
                                                    edgecolor=lcolor,
                                                    facecolor='none',
                                                    linestyle=lstyle,
                                                    linewidth=lwidth,
                                                    zorder=5,
                                                    transform=ccrs.PlateCarree()))
            
    def annot(self,lonlat:tuple, text:str, **kwargs):
        if self.extent[0]<=lonlat[0] and self.extent[1]>=lonlat[0] and self.extent[2]<=lonlat[1] and self.extent[3]>=lonlat[1]:
            self.text(lonlat[0],lonlat[1], text, size=self.scalling_value(0.45), **kwargs)
    
    def manual_tick_axes(self):
        gl = self.__ax_main.gridlines(
            crs=ccrs.PlateCarree(), color='gray', linestyle='--', linewidth=0.25, alpha=0.4,
            xlocs=self.__xlocs, ylocs=self.__ylocs)
        
        ylim = self.__ax_main.get_ylim()
        xlim = self.__ax_main.get_xlim()

        yrange=ylim[1]-ylim[0]
        xrange=xlim[1]-xlim[0]
        for xtick in self.__xlocs:
            if xtick>0:
                string = f'{xtick:3.0f}° E'
            else:
                string = f'{abs(xtick):3.0f}° W'
            self.text(xtick, ylim[0]+yrange*0.01, string, size=self.scalling_value(0.6), color='w', ha='center', va='bottom', bbox=dict(boxstyle='round', facecolor='k', edgecolor='none', alpha=0.28, pad=0.28),weight="bold", zorder=5)

        for ytick in self.__ylocs[::2]:
            if ytick>0:
                string = f'{ytick:3.0f}° N'
            else:
                string = f'{abs(ytick):3.0f}° S'
            self.text(self.extent[0]+xrange*0.006, ytick, string, size=self.scalling_value(0.6), color='w', ha='left', va='center', bbox=dict(boxstyle='round', facecolor='k', edgecolor='none', alpha=0.28, pad=0.28),weight="bold", zorder=5)

    def show(self):
        plt.show()


def ccmap(colors, intervals, increase):
    from matplotlib.colors import LinearSegmentedColormap
    colors = colors
    intervals = np.array(intervals)

    if not isinstance(increase, list):
        increase = np.ones([len(colors)])*increase
    elif len(colors)==len(increase):
        increase = increase

    NColors_List = ((intervals[1:]-intervals[:-1])/increase).astype('int')
    IColors_Split = np.full([len(colors)], 0)
    EColors_Split = NColors_List

    labels = []
    for idx, color in enumerate(colors):
        # Full colorbar labels
        ini_range = intervals[idx]
        end_range = intervals[idx+1]+increase[idx] if idx == len(colors)-1 else intervals[idx+1]

        FCLabels = np.arange(ini_range, end_range, increase[idx])
        labels = np.concatenate((labels, FCLabels))

        if isinstance(color, LinearSegmentedColormap):
            Pallete = color(np.linspace(0, 11, NColors_List[idx]))
            Pallete = Pallete[IColors_Split[idx]:EColors_Split[idx]]
        elif isinstance(color,list):
            color = [np.array(clr)/255 if isinstance(clr,list) else clr for clr in color]

            Pallete = LinearSegmentedColormap.from_list(name='ColorGradient',
                                                        colors=color, N=NColors_List[idx])
            Pallete = Pallete(np.linspace(0, 1, NColors_List[idx]))
            Pallete = Pallete[IColors_Split[idx]:EColors_Split[idx]]
            # print(Pallete)
        cmap = Pallete if idx==0 else np.vstack((cmap, Pallete))
    cmap = LinearSegmentedColormap.from_list('ColorGradient', cmap, cmap.shape[0])
    return cmap


if __name__ == '__main__':
    
    # Crear un mapa de datos de ejemplo
    np.random.seed(0)
    x = np.linspace(-90, -60, 361)
    y = np.linspace(-20, 5, 181)

    xx, yy = np.meshgrid(x, y)
    data = (np.sin(2*xx) + np.cos(2*yy)) * np.exp(-(xx**2 + yy**2)/2000)

    # Añadir ruido aleatorio
    data += np.random.normal(scale=0.01, size=data.shape)

    extent = [x.min(), x.max(), y.min(), y.max()]

    projection = ccrs.PlateCarree()
    canvas = Canvas(extent=extent, projection=projection)

    # img = canvas.pcolormesh(x,y,data)
    img = canvas.imshow(data, extent=extent)

    lticks = np.linspace(data.min(), data.max(), 8).astype(np.float16)[:-1]
    print(lticks, data.min(), data.max())
    cbar = canvas.colorbar(img, extend='neither', lticks=lticks)

    canvas.title('ESTE ES UN TITULO', weight='bold', fontname='Liberation Serif', loc='t')
    canvas.title('Subtítulo para información adicional', fontname='Liberation Serif', loc='b', size=0.85)
    from datetime import datetime
    canvas.title(f'{datetime.today():%Y-%m-%d} - UTC ', align='right', loc='rt', size=0.7)
    canvas.title(f'{datetime.today():%Y-%m-%d} - Perú', align='right', loc='rb', size=0.7)
    # canvas.add_logo('/home/spm/SENAMHI/goes_models/assets/logos/SENAMHI_LOGO3.png')
    # canvas.add_shp('/home/spm/SENAMHI/goes_models/assets/shapes/suramerica_geo.shp', lcolor='#ffffff', width=1.4,)
    canvas.border()
    canvas.savefig('pru.png')