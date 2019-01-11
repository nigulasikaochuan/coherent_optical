from scipy.spatial import Voronoi
from scipy.signal import lfilter
from scipy.interpolate import griddata
import plotly.graph_objs as go
import visdom
import numpy as np
from scipy.signal import convolve2d

from shapely.geometry import Polygon
def polyarea(x, y):
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def density(x, y):
    points = np.column_stack((x, y))
    voro_obj = Voronoi(points, qhull_options='Qbb')

    vertices = voro_obj.vertices
    regions = voro_obj.regions
    region_index = voro_obj.point_region
    dd = np.zeros((len(x), 1))
    cnt = 0
    for index in region_index:
        if len(regions[index]) == 0:
            continue
        elif -1 in regions[index]:
            cnt += 1
            continue
        else:
            k = regions[index]

            coords = np.column_stack((vertices[k,0],vertices[k,1])).tolist()
            polgyon = Polygon(coords)
            area = polgyon.area
            dd[cnt, 0] = 1 / area
            cnt += 1
    return dd


def downsample(x, sps):
    x2 = x[0:len(x):sps]
    return x2


def main():
    from scipy.io import loadmat
    x = loadmat('/Volumes/D/zaxiang/mysimulation/gitcode/dsp_processing/matlab.mat')['rxSamples']
    x1 = x[0, :]
    x1 = downsample(x1, 2)
    dd = density(np.real(x1), np.imag(x1))
    x = np.real(x1)
    y = np.imag(x1)
    xi, yi = np.meshgrid(np.linspace(np.min(x), np.max(x), 100), np.linspace(np.min(y), np.max(y), 100))
    zi = griddata(np.column_stack((x, y)), dd.reshape(-1), (xi, yi), fill_value=0)
    coef = np.ones(5) / 5
    # zif = lfilter(coef, 1, zi, axis=0)
    # zif = lfilter(coef, 1, zif, axis=1)
    coef.shape = -1, 1
    zif = convolve2d(zi, coef @ coef.T, mode='same')
    ddf = griddata((xi.ravel(), yi.ravel()), zif.ravel(), (x, y))
    gsp(x, y, ddf, 4)
    print("sss")


def gsp(x, y, dd, ms):
    from scipy.io import loadmat
    import os
    path = os.path.dirname(os.path.abspath(__file__))
    colormap = loadmat(os.path.join(path, 'color.mat'))['map']
    color_index = np.floor(((dd - np.min(dd)) / (np.max(dd) - np.min(dd)) * (colormap.shape[0] - 1)))
    viz = visdom.Visdom(env='plot')
    traces = []
    layout = go.Layout(title='scatter plot', showlegend=False)
    for k in range(colormap.shape[0]):
        if np.any(color_index == k):
            xdata = x[color_index == k]
            ydata = y[color_index == k]
            color = (colormap[k, :]).tolist()
            color_string = ''
            for co in color:
                color_string += str(co) + ','
            color_string = color_string[:-1]
            color_string = f'rgb({color_string})'
            size = [4] * len(xdata)
            traces.append(go.Scattergl(x=xdata, y=ydata, mode='markers',
                                       marker=dict(size=size,color=color_string, colorscale=[[0.0, '#3e26a8'],
                                                                     [0.1111111111111111, '#4741e5'],
                                                                     [0.2222222222222222, '#4269fe'],
                                                                     [0.3333333333333333, '#2e87f7'],
                                                                     [0.4444444444444444, '#1ca9df'],
                                                                     [0.5555555555555556, '#19bfb6'],
                                                                     [0.6666666666666666, '#4acb84'],
                                                                     [0.7777777777777778, '#9dc943'],
                                                                     [0.8888888888888888, '#f0ba36'],
                                                                     [0.9,"#fad62d"],
                                                                     [1.0, '#f9fb15']]))
                          )

            if k==0:
                traces[0]['marker'].update(colorbar=dict(title='density'))

            fig = go.Figure(data=traces, layout=layout)
            if k == 0:
                win = viz.plotlyplot(fig)
            else:
                viz.plotlyplot(fig, win=win)


main()
# import numpy as np
# import matlab
# import matlab.engine
# def dowmsample(x, sps):
#     return x[0:len(x):sps]
#
# eng = matlab.engine.connect_matlab()
# if __name__ == "__main__":
#     from scipy.io import loadmat
#     x = loadmat('/Volumes/D/zaxiang/mysimulation/gitcode/dsp_processing/matlab.mat')['rxSamples']
#     y = x[0]
#     y = dowmsample(y, 2)
#     y2 = x[1]
#     y2 = dowmsample(y2, 2)
#     x_r = np.real(y)
#     x_i  = np.imag(y)
#
#     eng.plot_const(matlab.double(y.tolist(),is_complex=True),nargout=0)
