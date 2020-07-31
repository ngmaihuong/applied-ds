#-------------------------------------------------------------------------------
#NAME:  Huong Mai Nguyen
#DATE:  July 15, 2020
#TITLE: Visualizing probabilistic uncertainty using matplotlib.
#       The colors of the bars in the bar chart changes based on the 
#       probability that a chosen y-value (represented by the red line) falls
#       within the represented distribution.
#-------------------------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from scipy import stats
import matplotlib.colors as colors

# Generate random samples
np.random.seed(12345)

df = pd.DataFrame([np.random.normal(32000,200000,3650),
                   np.random.normal(43000,100000,3650),
                   np.random.normal(43500,140000,3650),
                   np.random.normal(48000,70000,3650)],
                  index=[1992,1993,1994,1995])

# Calculate descriptive statistics: Mean, standard deviation, 95% confidence intervals
value = df.T.mean()
std = df.T.std()
n = df.shape[1] #width
yerr = std / np.sqrt(n) * stats.norm.ppf(1-(1-0.95)/2)

# Define colormap
cmap = cm.get_cmap('seismic')
cpick = cm.ScalarMappable(cmap=cmap, norm=colors.Normalize(vmin=0, vmax=1.0))
cpick.set_array([])

# Define axes
X_MIN = -1
X_MAX = df.shape[0]
Y_MIN = 0
Y_MAX = 55000
#X_VALS = range(X_MIN, X_MAX+1) # possible x values for the line
Y_VALS = np.arange(Y_MIN, Y_MAX, 1000)

# Draw bar chart whose colors change based on the probability that a chosen y-value falls within a distribution
def update_line(num, line):
    y_loc = Y_VALS[num]
    line.set_data( [X_MIN, X_MAX], [y_loc, y_loc] )

    p = []
    for i in range(len(value)):
        if y_loc > value.iloc[i] + yerr.iloc[i]:
            p.append(0)
        elif y_loc < value.iloc[i] - yerr.iloc[i]:
            p.append(1)
        else:
            if y_loc > value.iloc[i]:
                p.append((value.iloc[i] + yerr.iloc[i] - y_loc) / (yerr.iloc[i]*2))
            else:
                p.append(1-((y_loc - (value.iloc[i] - yerr.iloc[i])) / (yerr.iloc[i]*2)))

    plt.bar(range(df.shape[0]), value,
            yerr = yerr,
            error_kw=dict(lw=1, capsize=5, capthick=1),
            color=cpick.to_rgba(p),
            tick_label=value.index)

    return line,

fig = plt.figure()

#plt.figure()
#plt.axhline(threshold, color = 'grey', alpha = 0.5)

# Draw color bar
plt.colorbar(cpick,
             boundaries=np.arange(0, 1, 0.1),
             ticks=np.arange(0, 1, 0.1),
             spacing='uniform',
             drawedges=True,
             orientation='vertical',
             label='Probability that Distribution Contains Certain Values')

plt.show()

# Draw the red line representing the chosen y-value
l , v = plt.plot(X_MIN, Y_MIN, X_MAX, Y_MAX, linewidth=2, color= 'red', zorder=10)

# Label axes and chart
plt.xlim(X_MIN, X_MAX)
plt.ylim(Y_MIN, Y_MAX)
plt.xlabel('Year')
plt.ylabel('Value')
plt.title('Uncertainty Visualization', fontweight='bold')

# Build animation
line_anim = animation.FuncAnimation(fig, update_line, len(Y_VALS), fargs=(l, ), interval=5)

#from matplotlib import rcParams

# make sure the full paths for ImageMagick and ffmpeg are configured
#rcParams['animation.convert_path'] = r'C:\Program Files\ImageMagick\convert'
#rcParams['animation.ffmpeg_path'] = r'C:\Program Files\ffmpeg\bin\ffmpeg.exe'

# save animation at 5 frames per second
#line_anim.save('Week3-assignment.gif', writer='imagemagick', fps=3)
