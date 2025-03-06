from .canvas import ccmap

class Palette:
    def __init__(self, colors, intervals, increase):
        self.colors = colors
        self.intervals = intervals
        self.increase = increase
        self.min_value = min(intervals)
        self.max_value = max(intervals)
        self.colormap = ccmap(colors, intervals, increase)

    def __call__(self, *args, **kwargs):
        return self.colormap(*args, **kwargs)

# Definiciones de paletas
_PALETTE_DEFINITIONS = {
    'VIS': {
        'colors': [
            [(5/255, 5/255, 5/255), (100/255, 100/255, 100/255)],
            [(100/255, 100/255, 100/255), (190/255, 190/255, 190/255), (230/255, 230/255, 230/255), '#FFFFFF']
        ],
        'intervals': [0, 13, 100],
        'increase': 1
    },
    'NIR4': {
        'colors': [
            [(10/255, 10/255, 10/255), (240/255, 240/255, 240/255)],
            ['darkblue', 'deepskyblue', 'skyblue']
        ],
        'intervals': [0.0, 40.0, 100.0],
        'increase': 0.1
    },
    'NIR5': {
        'colors': [
            [(30/255, 30/255, 30/255), (255/255, 255/255, 255/255)]
        ],
        'intervals': [0.0, 100.0],
        'increase': 1
    },
    'NIR6': {
        'colors': [
            ['black', 'dimgray'],
            [(169/255, 208/255, 207/255), (0/255, 110/255, 111/255)],
            ['darkblue', (169/255, 170/255, 218/255)],
            ['yellow', (166/255, 0/255, 0/255)],
            ['fuchsia', (255/255, 169/255, 255/255)],
            [(169/255, 169/255, 169/255), (255/255, 255/255, 255/255)]
        ],
        'intervals': [0.0, 30.0, 35.0, 40.0, 50.0, 55.0, 100.0],
        'increase': 0.5
    },
    'WV': {
        'colors': [
            [(85/255, 0/255, 84/255), (174/255, 46/255, 172/255), (239/255, 139/255, 238/255)],
            [(0/255, 54/255, 0/255), 'lawngreen'],
            ['darkblue', 'white'],
            [(240/255, 240/255, 240/255), (60/255, 60/255, 60/255)],
            [(65/255, 36/255, 2/255), 'orange', 'red', 'darkred', (63/255, 0/255, 0/255), 'black']
        ],
        'intervals': [-90.0, -75.0, -60.0, -45.0, -25.0, 15.0],
        'increase': 0.5
    },
    'IR': {
        'colors': [
            [(96/255, 32/255, 184/255), (226/255, 70/255, 249/255)],
            [(202/255, 0/255, 0/255), (255/255, 125/255, 125/255)],
            ['mediumblue', 'deepskyblue'],
            [(8/255, 130/255, 8/255), (4/255, 222/255, 4/255), 'greenyellow'],
            [(250/255, 255/255, 107/255), (227/255, 227/255, 0/255), (164/255, 165/255, 0/255)],
            [(255/255, 199/255, 90/255), 'orange', (203/255, 101/255, 25/255), (139/255, 69/255, 19/255), (65/255, 36/255, 2/255)],
            [(140/255, 140/255, 140/255), (0/255, 0/255, 0/255)]
        ],
        'intervals': [-90.0, -80.0, -70.0, -55.0, -45.0, -35.0, -10.0, 50.0],
        'increase': 0.5
    }
}

# Creación de instancias de paletas
VIS = Palette(**_PALETTE_DEFINITIONS['VIS'])
NIR4 = Palette(**_PALETTE_DEFINITIONS['NIR4'])
NIR5 = Palette(**_PALETTE_DEFINITIONS['NIR5'])
NIR6 = Palette(**_PALETTE_DEFINITIONS['NIR6'])
WV = Palette(**_PALETTE_DEFINITIONS['WV'])
IR = Palette(**_PALETTE_DEFINITIONS['IR'])

__all__ = ['VIS', 'NIR4', 'NIR5', 'NIR6', 'WV', 'IR']

