#!/usr/bin/env python2

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from melo_nfl import nfl_spreads


# figure style and layout
#plt.style.use('clean')
width, height = plt.rcParams['figure.figsize']
fig, (axl, axr) = plt.subplots(
    ncols=2, figsize=(2*width, height)
)

# standard normal distribution
x = np.linspace(-4, 4, 1000)
y = norm.pdf(x)
axl.plot(x, y, color='black')

# raw residuals
residuals = nfl_spreads.residuals()[256:]
print('mean: {:.2f}'.format(residuals.mean()))
print('mean absolute error: {:.2f}'.format(np.abs(residuals).mean()))

# standardized residuals
residuals = nfl_spreads.residuals(standardize=True)[256:]
axl.hist(residuals, bins=40, histtype='step', density=True)

# residual figure attributes
axl.set_xlim(-4, 4)
axl.set_ylim(0, .45)
axl.set_xlabel(r'$(y_\mathrm{obs}-y_\mathrm{pred})/\sigma_\mathrm{pred}$')
axl.set_title('Standardized residuals')

# quantiles
quantiles = nfl_spreads.quantiles()[256:]
axr.axhline(1, color='black')
axr.hist(quantiles, bins=20, histtype='step', density=True)

# quantile figure attributes
axr.set_xlim(0, 1)
axr.set_ylim(0, 1.5)
axr.set_xlabel(r'$\int_{-\infty}^{y_\mathrm{obs}} P(y_\mathrm{pred}) dy_\mathrm{pred}$')
axr.set_title('Quantiles')

plt.tight_layout()
plt.savefig('validate.pdf')
