from topi.cce import te_set_version

import subprocess

te_set_version("1.32.0.B080")


process = subprocess.call(['python','./operator/JDEcoder.py'])
print(process)
