# -- Define function that can read info from both fif and ds files -- #
import contextlib
import sys
import numpy as np
import os

# -- Works both in python2 and python3 -- #
try:
    from cStringIO import StringIO
except ImportError:
    from io import StringIO
# --------------------------------------- #

def LoadAutGamma(dirname, subj_names):
    data = list()
    fnames = list()
    for subj_name in subj_names:
        fname = [f for f in os.listdir(dirname) if f.startswith(subj_name)]
        if fname:
            fname = fname[0]
            abs_fname = os.path.join(dirname, fname)
            cur_matr = np.load(abs_fname).mean(axis=0)
        else:
            cur_matr = np.full((306,), np.nan)
        if cur_matr.shape[0] != 306:
            cur_matr = cur_matr[:306]
        data.append(cur_matr)
        fnames.append(fname)
    


    data = np.ma.masked_invalid(data)
    fnames = np.ma.masked_array(fnames, mask=np.all(data.mask, axis=1))
    return data, fnames

# --- This function is dataset-specific ---------- #
def LoadSczPowerData(dirname, subj_names, missing_channel_id=245):
    """Load and gather .npy data for subjects in [dir]. Dataset-specific"""
    data = list()
    fnames = list()

    for subj_name in subj_names:
        fname = [f for f in os.listdir(dirname) if f.startswith(subj_name)]
        if fname:
            # Found file for subj_name
            fname = fname[0]
            abs_fname = os.path.join(dirname, fname)
            # load and average power across trials
            cur_matr = np.load(abs_fname).mean(axis=0)
        else:
            # No file for subj_name
            cur_matr = np.full((275,), np.nan)
            fname = ''

        # some of CTF files for in our dataset have 275 MEG channels
        # while others have all
        if cur_matr.shape[0] == 276:
            cur_matr = np.delete(cur_matr, missing_channel_id)
        cur_matr_no_eeg = cur_matr[:-4]  # remove eeg channels
#         print cur_matr_no_eeg.shape
        data.append(cur_matr_no_eeg)
        fnames.append(fname)

    data = np.ma.masked_invalid(data)
    fnames = np.ma.masked_array(fnames, mask=np.all(data.mask, axis=1))
#     print data.shape
    return data, fnames


@contextlib.contextmanager
def nostdout():
    save_stdout = sys.stdout
    sys.stdout = StringIO()
    yield
    sys.stdout = save_stdout


def read_info_custom(fname):
    """Read info from .fif or from .ds"""
    from os.path import splitext
    _, ext = splitext(fname)
    if ext == '.fif':
        from mne.io import read_info
        info = read_info(fname)
    elif ext == '.ds':
        from mne.io import read_raw_ctf
        with nostdout():
            raw = read_raw_ctf(fname)
        info = raw.info
    else:
        raise RuntimeError('Unknown format for {}'.format(fname))
    return info
