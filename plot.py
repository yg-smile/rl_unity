from utils.utils_plots import *
from glob import glob
import os.path

algo = 'sac'
dir = 'res'
envs = [
    '3DBall',
    '3DBallN',
]
ext = ''

files = []
for env in envs:
    l = glob('{}/{}_{}_seed*_ext{}.pkl'.format(dir, env, algo, ext))
    lm = [os.path.basename(ii) for ii in l]  # only get file name
    ln = [os.path.splitext(ii)[0] for ii in lm]  # remove the file extension
    files.append(ln)

plot_training_time(files=files,
                   labels=envs,
                   dir=dir,
                   file_name='3DBall')
