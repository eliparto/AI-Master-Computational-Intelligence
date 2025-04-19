""" Plot child fitnesses before and after learning in the console. """

import plotille

def consolePlot(fit_old, fit_new):
    """
    Plot all fitnesses after learning
    """
    x = np.arange(0, len(fit_old), 1)
    fig = plotille.Figure()
    fig.set_x_limits(0, len(fit_old))
    fig.width = 60
    fig.height = 30
    fig.plot(x, sorted(fit_old), interp = "linear", lc = "cyan")
    fig.plot(x, sorted(fit_new), interp = "linear", lc = "magenta")
    
    print(fig.show())