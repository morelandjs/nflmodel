"""Graphically validate model predictions."""

import logging

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from . import model


def assess_predictions(mode):
    """
    Plot several statistical tests of the model prediction accuracy.

    """
    # figure style and layout
    width, height = plt.rcParams["figure.figsize"]
    fig, (axl, axr) = plt.subplots(
        ncols=2, figsize=(2*width, height)
    )

    # load nfl spread predictions
    nfl_model = model.EloraTeam.from_cache(mode, calibrate=False)

    # standard normal distribution
    x = np.linspace(-4, 4, 1000)
    y = norm.pdf(x)
    axl.plot(x, y, color="black")

    # raw residuals
    residuals = nfl_model.residuals(standardize=False)
    logging.info("{} residual mean: {:.2f}"
                 .format(mode, residuals.mean()))
    logging.info("{} residual mean absolute error: {:.2f}"
                 .format(mode, nfl_model.mean_abs_error))

    # standardized residuals
    std_residuals = nfl_model.residuals(standardize=True)
    axl.hist(std_residuals, bins=40, histtype="step", density=True)

    # residual figure attributes
    axl.set_xlim(-4, 4)
    axl.set_ylim(0, .45)
    axl.set_xlabel(r"$(y_\mathrm{obs}-y_\mathrm{pred})/\sigma_\mathrm{pred}$")
    axl.set_title("Standardized residuals")

    # quantiles
    quantiles = nfl_model.sf(
        nfl_model.examples.value,
        nfl_model.examples.time,
        nfl_model.examples.label1,
        nfl_model.examples.label2,
        nfl_model.examples.bias
    )[nfl_model.burnin:]

    axr.axhline(1, color="black")
    axr.hist(quantiles, bins=20, histtype="step", density=True)

    # quantile figure attributes
    axr.set_xlim(0, 1)
    axr.set_ylim(0, 1.5)
    axr.set_xlabel(' '.join([r"$\int_{-\infty}^{y_\mathrm{obs}}$",
                             r"$P(y_\mathrm{pred})$" r"$dy_\mathrm{pred}$"]))
    axr.set_title("Quantiles")

    plt.tight_layout()
    plt.savefig("validate_{}.png".format(mode), dpi=200)
