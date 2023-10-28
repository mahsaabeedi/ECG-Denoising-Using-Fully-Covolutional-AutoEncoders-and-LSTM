#  QTdatabase: https://physionet.org/static/published-projects/qtdb/qt-database-1.0.0.zip
#  MIT-BIH Noise Stress Test Database: https://physionet.org/static/published-projects/nstdb/mit-bih-noise-stress-test-database-1.0.0.zip
#  Installing Physionet WFDB package run from your terminal:
#    $ pip install wfdb 
#

import numpy as np
import wfdb
import _pickle as pickle

def prepare(NSTDBPath='data/mit-bih-noise-stress-test-database-1.0.0/bw'):
    signals, fields = wfdb.rdsamp(NSTDBPath)

    for key in fields:
        print(key, fields[key])

    np.save('data/NoiseBWL', signals)
    # Save Data
    with open('data/NoiseBWL.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump(signals, output)
    print('=========================================================')
    print('MIT BIH data noise stress test database (NSTDB) saved as pickle')

