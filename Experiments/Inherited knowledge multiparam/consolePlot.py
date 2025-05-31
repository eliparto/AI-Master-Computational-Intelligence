""" Plot child fitnesses before and after learning in the console. """

import plotille
import numpy as np

def consolePlot(fit_old, fit_new):
    """
    Plot pre- and post-learning fitnesses (sorted).
    """
    x = np.arange(0, len(fit_old), 1)
    fig = plotille.Figure()
    fig.set_x_limits(0, len(fit_old))
    fig.set_y_limits(0, np.ceil(max(np.concatenate((fit_old, fit_new)))))
    fig.width = 50
    fig.height = 20
    fig.plot(x, sorted(fit_old), interp = "linear", lc = "magenta",
             label = "Prev. gen")
    fig.plot(x, sorted(fit_new), interp = "linear", lc = "cyan", 
             label = "Curr. gen")
    fig.plot(x, np.ones(len(fit_old))*np.average(fit_new), interp = "linear",
             lc = "yellow", label  ="Curr. gen (AVG)")
    
    print(fig.show(legend=True))
    print("New generation:")
    print(f"Max fitness: {max(fit_new)}")
    print(f"Avg fitness: {np.average(fit_new)}\n")