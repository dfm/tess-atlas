# +
import numpy as np
from matplotlib import patches, pyplot as plt
import exoplanet as xo
import matplotlib as mpl
plt.style.use(
    "https://gist.githubusercontent.com/avivajpeyi/4d9839b1ceb7d3651cbb469bc6b0d69b/raw/4ee4a870126653d542572372ff3eee4e89abcab0/publication.mplstyle")
mpl.rcParams.update({
"axes.titlesize" : 24,
"axes.labelsize" : 20,
"lines.linewidth" : 3,
"lines.markersize" : 10,
"xtick.labelsize" : 16,
"ytick.labelsize" : 16,
"axes.grid": False,
"xtick.minor.visible":  False, 
"ytick.minor.visible":  False, 
})

# compute lc 
p = 5
orbit = xo.orbits.KeplerianOrbit(period=p)
N = 1000
lim = 0.2
r1, r2 = (-lim, lim), (p-lim, p+lim)
x = np.concatenate((np.linspace(*r1, N), np.linspace(*r2, N)))
u = [0.4, 0.5]
y = (
    xo.LimbDarkLightCurve(u).get_light_curve(orbit=orbit, r=0.3, t=x, texp=0.1).eval()
)


# +
## 
f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w', figsize=(6, 4))

# lc
lc_c = "tab:blue"
ax.plot(x, y, zorder=-10, color=lc_c)
ax2.plot(x, y, zorder=-10, color=lc_c)



min_y = min(y)[0]

d_x, d_c = -0.15, "tab:red"
depth_arrow = mpl.patches.FancyArrowPatch(
    (d_x,0),(d_x, min_y), 
    arrowstyle='->', 
    mutation_scale=20,
    color=d_c
)
ax.add_patch(depth_arrow)
ax.text(-.13, -0.05, r'$\delta$', fontsize=20, ha='center', va='center', color=d_c)

# duration arrow
d_x, d_c = 0.125, "tab:green"
dur_arrow = mpl.patches.FancyArrowPatch(
    (-d_x,0),(d_x, 0), 
    arrowstyle='<->', 
    mutation_scale=20,
    color=d_c
)
ax.add_patch(dur_arrow)
ax.text(0, 0, r'$\tau$', fontsize=20, ha='center', va='top', color=d_c)

# period arrow
p_c = "tab:orange"
p_arrow = patches.ConnectionPatch(
    [0, min_y],
    [p, min_y],
    coordsA=ax.transData,
    coordsB=ax2.transData,
    # Default shrink parameter is 0 so can be omitted
    color=p_c,
    arrowstyle="<->",  # "normal" arrow
    mutation_scale=20,  # controls arrow head size
    linewidth=1,
)
f.patches.append(p_arrow)
f.text(.55, 0.2, r'$P$', fontsize=20, ha='center', va='center', color=p_c)

# tmin tmax
f.text(.35, 0.98, r'$t_{\rm min}$', fontsize=20, ha='center', va='center', color='black')
f.text(.75, 0.98, r'$t_{\rm max}$', fontsize=20, ha='center', va='center', color='black')

# f0
f.text(1, 0.9, r'$f_0$', fontsize=20, ha='right', va='top', color='black')

ax.set_xlim(*r1)
ax2.set_xlim(*r2)

ax.tick_params(axis="x",direction="in", top=True, labeltop=False, bottom=True, labelbottom=True )
ax2.tick_params(axis="x",direction="in", top=True, labeltop=False, bottom=True, labelbottom=True )
ax.tick_params(axis="y",direction="in", left=True, labelleft=True, right=False, labelright=False )
ax2.tick_params(axis="y",direction="in", left=False, labelleft=False, right=True, labelright=False )
ax.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)

ax2.locator_params(axis='y', nbins=2)
ax.locator_params(axis='y', nbins=2)
# ax2.locator_params(axis='x', nbins=2)
# ax.locator_params(axis='x', nbins=2)

# ax.xaxis.get_majorticklabels()[-1].set_horizontalalignment("right")
# ax2.xaxis.get_majorticklabels()[0].set_horizontalalignment("left")
ax.xaxis.set_ticklabels(["","0",""])
ax2.xaxis.set_ticklabels(["","5",""])


d = .015 
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, lw=1)
ax.plot((1-d,1+d), (-d,+d), **kwargs)
ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the right axes
ax2.plot((-d,+d), (1-d,1+d), **kwargs)
ax2.plot((-d,+d), (-d,+d), **kwargs)

# dash on lc 
kwargs.update(transform=ax.transAxes, color=lc_c)  
ax.plot((1-d,1+d), (0.92-d, 0.92+d),**kwargs)
kwargs.update(transform=ax2.transAxes)  # switch to the right axes
ax2.plot((-d,+d), (0.92-d, 0.92+d),**kwargs)

# axes lables 
f.text(0.05, 0.55, r'Relative Flux', fontsize=20, ha='center', va='center', color='black', rotation=90)
f.text(0.55, .05, r'Time [days]', fontsize=20, ha='center', va='center', color='black')

f.tight_layout(w_pad=-0.1)
plt.savefig('transit_model.png', bbox_inches='tight')
