from pylab import rcParams
import matplotlib.pyplot as plt

def pretty_bar(vals, xlabels, title, xtitle, ytitle, width=0.6, figwidth=5, rot=0, plot_margin=0):
    rcParams['figure.figsize'] = figwidth, 5
    rcParams['xtick.labelsize'] = 16
    rcParams['ytick.labelsize'] = 16
    plt.bar(range(len(xlabels)), vals, tick_label=xlabels, width=width, align='center')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel(xtitle)
    plt.ylabel(ytitle, fontsize=16)
    plt.grid(axis='x')
    plt.title(title, 
              horizontalalignment='center',
              fontsize=20, 
              style='oblique', 
              weight='bold', 
              variant='small-caps')
    x0, x1, y0, y1 = plt.axis()
    plt.axis((x0 - plot_margin, x1 + plot_margin, y0, y1))
    plt.show()
	