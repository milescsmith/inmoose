# -----------------------------------------------------------------------------
# Copyright (C) 2019-2020 Yuqing Zhang
# Copyright (C) 2022-2023 Maximilien Colange

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

# This file is based on the file 'R/helper_seq.R' of the Bioconductor sva package (version 3.44.0).


from multiprocessing import cpu_count

import numpy as np
import numpy.typing as npt
import scipy as sp
from joblib import Parallel, delayed
from scipy.stats import nbinom
from tqdm.auto import tqdm, trange

from ..utils import LOGGER


class ParallelTqdm(Parallel):
    def __call__(self, iterable, n_tasks):
        self.tqdm = tqdm(total=n_tasks)
        return super().__call__(iterable)

    def print_progress(self):
        self.tqdm.update(self.n_completed_tasks - self.tqdm.n)
        if self._is_completed():
            self.tqdm.close()


def vec2mat(vec, n_times):
    """
    Expand a vector into matrix (columns as the original vector)
    """
    vec = np.asarray(vec)
    vec = vec.reshape(vec.shape[0], 1)
    return np.full((vec.shape[0], n_times), vec)


def match_quantiles(counts_sub, old_mu, old_phi, new_mu, new_phi):
    """
    Match quantiles from a source negative binomial distribution to a target
    negative binomial distribution.

    Arguments
    ---------
    counts_sub : array_like
        the original data following the source distribution
    old_mu : array_like
        the mean of the source distribution
    old_phi : array_like
        the dispersion of the source distribution
    new_mu : array_like
        the mean of the target distribution
    new_phi : array_like
        the dispersion of the target distribution

    Returns
    -------
    ndarray
        adjusted data, corresponding in the target distribution to the same
        quantiles as the input data in the source distribution
    """
    new_counts_sub = np.full(counts_sub.shape, np.nan)

    i = counts_sub <= 1
    new_counts_sub[i] = counts_sub[i]
    i = np.logical_not(i)
    old_size = np.full(old_mu.shape, 1 / old_phi.reshape(old_mu.shape[0], 1))
    new_size = np.full(new_mu.shape, 1 / new_phi.reshape(new_mu.shape[0], 1))
    old_prob = old_size / (old_size + old_mu)
    new_prob = new_size / (new_size + new_mu)
    tmp_p = nbinom.cdf(counts_sub[i] - 1, old_size[i], old_prob[i])
    new_counts_sub[i] = np.where(
        np.abs(tmp_p - 1) < 1e-4,
        counts_sub[i],
        1 + nbinom.ppf(tmp_p, new_size[i], new_prob[i]),
    )

    # Original (pythonized) R code for reference
    #
    # for a in range(counts_sub.shape[0]):
    #    for b in range(counts_sub.shape[1]):
    #        if counts_sub[a,b] <= 1:
    #            new_counts_sub[a,b] = counts_sub[a,b]
    #        else:
    #            tmp_p = pnbinom_opt(counts_sub[a,b]-1, mu=old_mu[a,b], size=1/old_phi[a])
    #            if abs(tmp_p-1) < 1e-4:
    #                # for outlier count, if tmp_p==1, qnbinom(tmp_p) will return Inf values -> use original count instead
    #                new_counts_sub[a,b] = counts_sub[a,b]
    #            else:
    #                new_counts_sub[a,b] = 1+qnbinom_opt(tmp_p, mu=new_mu[a,b], size=1/new_phi[a])

    return new_counts_sub


# The following two functions are used for Monte-Carlo integration to obtain posterior estimates of the batch effect parameters.
# Transliterated from the R code in helper_seq.R of the Bioconductor sva package (version 3.44.0).
# Split in two functions for better readability and to allow parallelization of the Monte-Carlo integration across genes.
def sub_monte(i, dat, mu, gamma, phi, gene_subset_n):
    not_i = np.delete(np.arange(dat.shape[0]), i)
    m = mu[not_i, :][:, ~np.isnan(dat[i, :])]
    x = dat[i, ~np.isnan(dat[i,])]
    gamma_sub = gamma[not_i]
    phi_sub = phi[not_i]

    if gene_subset_n and (gene_subset_n > dat.shape[0] - 1):
        LOGGER.info(
            "gene_subset_n is larger than the number of genes; using all genes for Monte Carlo integration"
        )
        G_sub = dat.shape[0] - 1
    elif gene_subset_n:
        if i == 0:
            LOGGER.info(
                f"Using {gene_subset_n} random genes for Monte Carlo integration"
            )
        # mcint_ind = rng.choice(
        mcint_ind = np.random.choice(
            a=np.arange(dat.shape[0] - 1), size=gene_subset_n, replace=False
        )
        m = m[mcint_ind, :]
        gamma_sub = gamma_sub[mcint_ind]
        phi_sub = phi_sub[mcint_ind]
        G_sub = gene_subset_n
    else:
        LOGGER.info(
            "Using all genes for Monte Carlo integration; the function runs very slow for large number of genes"
        )
        G_sub = dat.shape[0] - 1

    # scipy.stats.nbinom.pmf is the equivalent of {stats}::dnbinom
    # BUT scipy.stats.nbinom.pmf does not have the mu parameter
    # the documentation for dnbiom states that mu is equivalent to
    # prob = size/(size+mu)
    lh = np.nan_to_num(
        x=[
            np.prod(
                sp.stats.nbinom.pmf(
                    k=x,
                    p=(1 / phi_sub[j]) / ((1 / phi_sub[j]) + m[j, :]),
                    n=1 / phi_sub[j],
                )
            )
            for j in range(G_sub)
        ],
        nan=0.0,
    )

    if (z := np.sum(lh)) == 0 or np.isnan(z):
        pos_res = {"gamma_star": np.ravel(gamma)[i], "phi_star": np.ravel(phi)[i]}
    else:
        pos_res = {
            "gamma_star": np.divide(np.sum(np.multiply(gamma_sub, lh)), np.sum(lh)),
            "phi_star": np.divide(np.sum(np.multiply(phi_sub, lh)), np.sum(lh)),
        }

    weights = np.divide(lh, np.sum(lh))
    return (pos_res, weights)


def monte_carlo_int_nb(
    dat: npt.ArrayLike,
    mu: npt.ArrayLike,
    gamma: npt.ArrayLike,
    phi: npt.ArrayLike,
    gene_subset_n: int | None = None,
    n_jobs: int = -1,
) -> dict[str, np.ndarray]:
    """
    Parameters
    ----------
    dat : :class:`numpy.typing.ArrayLike`
        counts
    mu : :class:`numpy.typing.ArrayLike`
        same shape as dat
    gamma : :class:`numpy.typing.ArrayLike`
        a gene-by-batch array
    phi : :class:`numpy.typing.ArrayLike`
        a gene-by-batch array
    gene_subset_n : int | None = None
        unknown
    """

    if n_jobs < 1:
        n_jobs = cpu_count()
    if n_jobs > 1:
        parallel = ParallelTqdm(n_jobs=n_jobs, backend="loky")
        res = parallel(
            (
                delayed(sub_monte)(i, dat, mu, gamma, phi, gene_subset_n)
                for i in range(dat.shape[0])
            ),
            n_tasks=dat.shape[0],
        )
    else:
        res = [
            sub_monte(i, dat, mu, gamma, phi, gene_subset_n)
            for i in trange(dat.shape[0])
        ]
    return {
        "gamma_star": np.array([x[0]["gamma_star"] for x in res]),
        "phi_star": np.array([x[0]["phi_star"] for x in res]),
        "weights": np.transpose(np.vstack([x[1] for x in res])),
    }
