from .canvas import ccmap
from matplotlib.colors import Colormap

class Palette(Colormap):
    def __init__(self, colors, intervals, increase):
        self._colors = colors
        self._intervals = intervals
        self._increase = increase
        self._cmap = ccmap(colors, intervals, increase)
    
    def __call__(self, *args, **kwargs):
        """Permite usar la paleta directamente como un colormap"""
        return self.cmap(*args, **kwargs)
    
    def __getattr__(self, attr):
        """Redirige cualquier atributo no encontrado al colormap subyacente"""
        return getattr(self._cmap, attr)
    
    def __str__(self):
        """Devuelve el nombre del colormap al imprimirlo"""
        return str(self._cmap)
        
    @property
    def colors(self):
        """Lista de colores que componen la paleta"""
        return self._colors
        
    @property
    def intervals(self):
        """Intervalos de la paleta"""
        return self._intervals
        
    @property
    def increase(self):
        """Incremento usado para generar el colormap"""
        return self._increase
        
    @property
    def min(self):
        """Valor mínimo del rango de la paleta"""
        return self._intervals[0]
        
    @property
    def max(self):
        """Valor máximo del rango de la paleta"""
        return self._intervals[-1]
        
    @property
    def cmap(self):
        """Colormap subyacente"""
        return self._cmap

# Dictionary with all palette definitions
_PALETTE_DEFINITIONS = {
    'VIS': {
        'colors': [
            [(5, 5, 5), (100, 100, 100)],
            [(100, 100, 100), (190, 190, 190), (230, 230, 230), '#FFFFFF']
        ],
        'intervals': [0, 13, 100],
        'increase': 1
    },
    'NIR4': {
        'colors': [
            [(10, 10, 10), (240, 240, 240)],
            ['darkblue', 'deepskyblue', 'skyblue']
        ],
        'intervals': [0.0, 40.0, 100.0],
        'increase': 0.1
    },
    'NIR5': {
        'colors': [
            [(30, 30, 30), (255, 255, 255)]
        ],
        'intervals': [0.0, 100.0],
        'increase': 1
    },
    'NIR6': {
        'colors': [
            ['black', 'dimgray'],
            [(169, 208, 207), (0, 110, 111)],
            ['darkblue', (169, 170, 218)],
            ['yellow', (166, 0, 0)],
            ['fuchsia', (255, 169, 255)],
            [(169, 169, 169), (255, 255, 255)]
        ],
        'intervals': [0.0, 30.0, 35.0, 40.0, 50.0, 55.0, 100.0],
        'increase': 0.5
    },
    'WV': {
        'colors': [
            [(85, 0, 84), (174, 46, 172), (239, 139, 238)],
            [(0, 54, 0), 'lawngreen'],
            ['darkblue', 'white'],
            [(240, 240, 240), (60, 60, 60)],
            [(65, 36, 2), 'orange', 'red', 'darkred', (63, 0, 0), 'black']
        ],
        'intervals': [-90.0, -75.0, -60.0, -45.0, -25.0, 15.0],
        'increase': 0.5
    },
    'IR': {
        'colors': [
            [(96, 32, 184), (226, 70, 249)],
            [(202, 0, 0), (255, 125, 125)],
            ['mediumblue', 'deepskyblue'],
            [(8, 130, 8), (4, 222, 4), 'greenyellow'],
            [(250, 255, 107), (227, 227, 0), (164, 165, 0)],
            [(255, 199, 90), 'orange', (203, 101, 25), (139, 69, 19), (65, 36, 2)],
            [(140, 140, 140), (0, 0, 0)]
        ],
        'intervals': [-90.0, -80.0, -70.0, -55.0, -45.0, -35.0, -10.0, 50.0],
        'increase': 0.5
    },
    'RAIN': {
        'colors': [
            ['#FFFFFF', '#ffeb99'],  
            ['#ffeb99', '#caff66', '#42ff9e'],  
            ['#42ff9e', '#00eaff', '#3b5bff', '#c400ff', '#ff0099']
        ],
        'intervals': [0,4,20,120], 
        'increase': 1
    }
}

# Create palette instances
VIS = Palette(**_PALETTE_DEFINITIONS['VIS'])
NIR4 = Palette(**_PALETTE_DEFINITIONS['NIR4'])
NIR5 = Palette(**_PALETTE_DEFINITIONS['NIR5'])
NIR6 = Palette(**_PALETTE_DEFINITIONS['NIR6'])
WV = Palette(**_PALETTE_DEFINITIONS['WV'])
IR = Palette(**_PALETTE_DEFINITIONS['IR'])
RAIN = Palette(**_PALETTE_DEFINITIONS['RAIN'])

# Definir qué paletas están disponibles para importación
__all__ = ['VIS', 'NIR4', 'NIR5', 'NIR6', 'WV', 'IR', 'RAIN']
