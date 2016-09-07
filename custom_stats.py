import numpy as np


def pair_perm_max_stat_t_test(cond1, cond2):
    """ Paired permutation t-test with maximum statistic correction for MC"""
    from mne.stats import permutation_t_test
    try:
        assert np.all(cond1.shape == cond2.shape)
    except AssertionError:
        raise ValueError('Different sizes for cond1 with shape {}and cond2 with shape {}'.format(cond1.shape, cond2.shape))
    except AttributeError:
        raise AttributeError('Both conditions should be of type numpy.ndarray')
    diff = cond1 - cond2
    T_obs, p_vals_corr, H0 = permutation_t_test(diff, n_permutations=100000)
    return p_vals_corr


def unpair_clust_f_test(cond1, cond2):
    """Unpaired f-test with cluster-level correction for MC"""
    from mne.stats import permutation_cluster_test as pct
    T_obs, clusters, cluster_pv, H0 = pct([cond1, cond2])
    p_vals = np.ones(cond1.shape[1])
    for i, sl in enumerate(clusters):
        p_vals[sl] = cluster_pv[i] * np.ones_like(sl)
    return p_vals


def fdr(test_func):
    def corr_p_vals(cond1, cond2):
        from mne.stats import fdr_correction
        p_vals = test_func(cond1, cond2)
        _, p_vals_corr = fdr_correction(p_vals)
        return p_vals_corr
    return corr_p_vals


def unpair_no_corr_t_test(cond1, cond2):
    """Unpaired t-test without correction for MC"""
    from scipy.stats import ttest_ind
    stat, p_vals = ttest_ind(cond1, cond2)
    return p_vals


def unpair_fdr_t_test(cond1, cond2):
    """Unpaired t-test with FDR correction for MC"""
    from scipy.stats import ttest_ind
    from mne.stats import fdr_correction

    stat, p_vals = ttest_ind(cond1, cond2)
    _, p_vals_corr = fdr_correction(p_vals)
    return p_vals_corr


def no_corr_mannwhitneyu(cond1, cond2):
    from scipy.stats import mannwhitneyu
    p_vals = np.ones(cond1.shape[1])
    for i in xrange(cond1.shape[1]):
        stat, p_vals[i] = mannwhitneyu(
            cond1[:, i], cond2[:, i], alternative='two-sided')
    return p_vals


def unpair_no_corr_perm_t_test(cond1, cond2):
    p_vals = np.ones(cond1.shape[1])
    from permute.core import two_sample
    for i in xrange(cond1.shape[1]):
        p_vals[i], t, dist = two_sample(cond1[:, i], cond2[:, i], reps=1000,
                                        stat='t', alternative='two-sided', keep_dist=True)
    return p_vals


def GetStatMask(cond1, cond2, stat_test_func, p_thresh=0.01):
    """Produce sensor-level binary statistical mask"""
    # ----------------------------------------------------------------- #
    p_vals = stat_test_func(cond1, cond2)
    mask = np.array(p_vals <= p_thresh)
    return mask, p_vals
