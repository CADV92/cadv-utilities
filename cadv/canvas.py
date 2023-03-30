import os
import numpy as np
import cartopy, cartopy.crs as ccrs
import matplotlib.pyplot as plt
import cartopy.io.shapereader as shpreader
from PIL import Image

class Canvas:
    def __init__(self, figsize=(10,8), projection=None, extent=None, axis='off'):
        self.extent = extent
        self.projection = projection

        if extent is not None:
            xfig, yfig = self.__proportion(*extent)
            yfig *= 1.12
            figsize = (xfig, yfig)


        self.__fig = plt.figure(figsize=figsize, constrained_layout=True, dpi=150)
        
        if projection is not None:
            self.__ax_main = self.__fig.add_axes([0, 0.05, 1, 0.90], projection=projection)
            if extent is not None:
                self.__ax_main.set_extent(extent, crs=projection)
        else:
            self.__ax_main = self.__fig.add_axes([0, 0.05, 1, 0.90])
        # self.__ax_main.set_axis(axis)

        # Header
        self.__ax_header = self.__fig.add_axes([0.01, 0.95, 0.98, 0.05], facecolor='black', frameon=False)
        self.__ax_header.set_axis_off()

        # Footer
        self.__ax_footer = self.__fig.add_axes([0, 0.025, 1, 0.025])
        self.__ax_footer.set_visible(True)
        self.__ax_bottom_frame = self.__fig.add_axes([0, 0.0, 1, 0.025])
        self.__ax_bottom_frame.set_visible(True)
        self.__ax_bottom_frame.set_axis_off()
    
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

    def imshow(self, data, extent, **kwargs):
        return self.__ax_main.imshow(data, extent=extent, **kwargs)

    def pcolormesh(self, x, y, data, **kwargs):
        return self.__ax_main.pcolormesh(x, y, data, **kwargs)
    
    def colorbar(self, image, lticks=None, **kwargs):
        cbar = plt.colorbar(image, cax=self.__fig.add_axes(self.__ax_footer), orientation='horizontal', pad=0.05, fraction=0.5, aspect=100, **kwargs)
        cbar.draw_all()
        cbar.ax.tick_params(labelsize=10.0, labelcolor='w', direction='in', pad=1.5)
        print(self.__ax_footer.get_position().height)
        if lticks is not None:
            cbar.ax.set_xticklabels(lticks)
        return cbar
    
    def title(self, title, **kwargs):
        img_size = self.__fig.get_size_inches()*self.__fig.dpi

        bbox = self.__ax_header.get_window_extent()
        width, height = bbox.width, bbox.height
        proportion = self.__scalling_values(width/height)

        print(f'Proportion: {proportion}')
        print(width, height, width/height, img_size[1], img_size[1]*0.035)

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
            weight=weight, color='w', fontname=fontname)
    
    def __scalling_values(self, value):
        min_valor = 2
        max_valor = 35
        a = 10
        b = 15
        valor_esc = a + (value - min_valor) * (b - a) / (max_valor - min_valor)
        return valor_esc

    def add_logo(self, logo):
        logo = Image.open(logo)
        # Obtener el tamaño de la imagen
        img_size = self.__fig.get_size_inches()*self.__fig.dpi

        Percent = (img_size[1]*self.__ax_header.get_position().height/logo.size[1])*.8
        LogoWidth = int(float(logo.size[0])*float(Percent))
        LogoHeight = int(float(logo.size[1])*float(Percent))
        logo = logo.resize((LogoWidth, LogoHeight), Image.BILINEAR)
        LogoHeight = float(logo.size[1])

        self.__fig.figimage(logo, 0, img_size[1]-LogoHeight*1.1, origin='upper', zorder=3)
    
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
    
    def border(self):
        # Add coastlines, borders and gridlines
        self.__ax_main.outline_patch.set_edgecolor('white')
        self.__ax_main.gridlines(color='white', alpha=0.5, linestyle='--', linewidth=1.5)
        self.__ax_main.coastlines(resolution='10m', color='white', linewidth=1.5)
        self.__ax_main.add_feature(cartopy.feature.BORDERS, edgecolor='white', linewidth=1.5)

    def savefig(self, name, path='.'):
        if not os.path.isdir(path):
            os.makedirs(path, exist_ok=True)

        plt.rcParams['savefig.facecolor'] = '#000'
        plt.rcParams['figure.facecolor'] = '#000'
        plt.rcParams['axes.facecolor'] = '#000'

        plt.savefig(os.path.join(path,name), bbox_inches='tight',pad_inches=0)
        plt.clf()
        plt.close('all')
        return os.path.join(path,name)
    
    def show(self):
        plt.show()

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