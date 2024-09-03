import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt

def common_plt():
    fig = plt.figure(figsize=(9, 6))
    # ax1 = fig.add_subplot(221)
    # ax2 = fig.add_subplot(222)
    # ax3 = fig.add_subplot(223)
    # ax4 = fig.add_subplot(224)
    # fig.tight_layout()

    # [left, bottom, width, height]
    ax1 = fig.add_axes([.05, .58, .4, .37])
    ax2 = fig.add_axes([.55, .58, .4, .37])
    ax3 = fig.add_axes([.05, .05, .4, .37])
    ax4 = fig.add_axes([.55, .05, .4, .37])

    # line color and styles.
    lines = []
    marker = []
    ls = ['o-', 'v:', 'x--', '+-.']
    for i, p in enumerate([.5, 1.0, 1.5, 2.]):
        xs = np.arange(1, 8)
        ys = .01*(xs/2)**p + p*.1 - .07 + .04*xs + .02*np.cos(xs*p)
        style = ls[i]
        lines.append(ax1.plot(xs*.05, ys, style,
                     label=f"ax.plot(x, y, '{style}')")[0])

    ax1.set_xlim(0, .4)
    ax1.set_ylim(0, .8)
    ax1.legend(handles=lines, loc='upper left', ncol=1)
    ax1.text(.01, .38, "ax.legend(loc='upper left')")
    ax1.set_title('ax = fig.add_subplot(2, 2, 1)')

    marker= ['o', 'x', 's', '^']
    color = ['b', 'k', 'g', 'r']
    dots = []
    for i, p in enumerate([.5, 1.0, 1.5, 2.]):
        xs = np.arange(1, 8)
        ys = .01*(xs/3)**p + p*.2 + .01*np.sin(xs*p)
        dots.append(ax2.scatter(xs*.05, ys, s=xs*5, c=color[i],
                                marker=marker[i], alpha=0.5))
    ax2.text(.35, .2, """\
ax.scatter(x, y,
           color=..,
           size=..,
           marker=..,
           alpha=0.5)""")
    ax2.set_xlim(0, .6)
    ax2.set_ylim(0, .6)
    ax2.text(.01, .55, 'ax.set_ylim(0, .6)')

    ax2.hlines(.1, 0, 1, ls=':', color='k')
    ax2.hlines(.5, 0, 1, ls=':', color='k')
    ax2.text(.39, .11, 'ax.hlines(.1, 0, .6)')
    #ax2.text(.39, .51, 'ax.hlines(.5, 0, .6)')
    ax2.set_xticks(np.arange(6)*.04 + .1)
    ax2.set_xticklabels(list('xticks'))
    ax2.text(.03, .01, 'ax.set_xticks(); ax.set_xticklabels()')
    ax2.set_xlabel("ax.set_xlabel(...)")
    ax2.set_ylabel('ax.set_ylabel(...)')
    #ax2.set_title('ax = fig.add_subplot(2, 2, 2)')
    ax2.set_title('ax.set_title(...)')

    methods = ['A', 'B', 'C']
    xs = np.arange(len(methods))
    np.random.seed(3)
    width = 0.2
    multiplier = -.25
    rects = []
    for i in range(4):
        ys = (np.random.rand(3) / (i+1)).round(2)
        offset = width * multiplier
        rects.append(ax3.bar(xs+offset, ys, width))
        multiplier += 1.05
    ax3.bar_label(rects[0], padding=2)
    ax3.bar_label(rects[1], padding=2)
    ax3.set_xticks(xs + width, methods)
    ax3.set_xlim(-.5, 3)
    ax3.set_ylim(0, 1.)
    ax3.set_title('ax = fig.add_subplot(2, 2, 3)')
    ax3.text(1.5, .85, 'B = ax.bar(x, y, 0.2)')
    ax3.annotate('ax.bar_label(B)',
                 xy=(1.1, .75), xytext=(1.5, .75),
                 arrowprops=dict(arrowstyle='->'))

    delta = 0.025
    x = np.arange(-3.0, 3.0, delta)
    y = np.arange(-2.0, 2.0, delta)
    O0, O1 = np.array([[0., .5],
                       [-1, 0]]) + np.random.rand(2,2) * delta
    X, Y = np.meshgrid(x, y)
    r = np.array([X, Y]).transpose(1,2,0)
    rO0 = np.linalg.norm(r - O0, axis=2)
    rO1 = np.linalg.norm(r - O1, axis=2)
    v = .32*erf(3.*rO0)/rO0 - .65*erf(1.5*rO1)/rO1
    CS = ax4.contour(X, Y, v)
    ax4.clabel(CS, inline=True, fontsize=10)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.text(0., -.8, 'C = ax.contour(X, Y, Z)')
    ax4.annotate('ax.clabel(C, inline=True)',
                 xy=(-1.3, -1.1), xytext=(0., -1.2),
                 arrowprops=dict(arrowstyle='->'))
    ax4.set_title('ax = fig.add_subplot(2, 2, 4)')

    plt.show()

if __name__ == '__main__':
    common_plt()
