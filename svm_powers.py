''' Docstring'''
import os.path as op
import os

from load_data import LoadAutGamma


if __name__ == '__main__':
    EP_PATH = '/media/dmalt/SSD500/aut_gamma/Moscow_baseline_results_new/'
    SUBJ_NAMES = dict()
    SUBJ_NAMES['K'] = [s for s in os.listdir(EP_PATH) if len(s) == 5 and s.startswith('K')]
    SUBJ_NAMES['R'] = [s for s in os.listdir(EP_PATH) if len(s) == 5 and s.startswith('R')]
    # --------------------------------------------------------------- #
    CONDS = ('ec', 'eo')
    BANDS = ('delta', 'theta', 'alpha', 'beta', 'lowgamma', 'highgamma')
    GROUPS = ('K', 'R')

    # --------------------------------------------------------------- #
    BASEDIR = '/media/dmalt/SSD500/aut_gamma/powers'

    DATA = dict()
    FNAMES = dict()

    # ----------- Load all the data ------------ #
    for cond in CONDS:
        DATA[cond] = dict()
        FNAMES[cond] = dict()
        for band in BANDS:
            DATA[cond][band] = dict()
            FNAMES[cond][band] = dict()
            for group in GROUPS:
                subdir = cond + '_' + band
                cond_dir = op.join(BASEDIR, subdir, group)
                cond_data, cond_fnames = LoadAutGamma(cond_dir, SUBJ_NAMES[group])

                DATA[cond][band][group] = cond_data
                FNAMES[cond][band][group] = cond_fnames

    # ----------------------------------------- #
    # print fnames['Closed']['alpha']['Controls']
    # cond1_data = data['-------Closed']['alpha'][]<Paste>

