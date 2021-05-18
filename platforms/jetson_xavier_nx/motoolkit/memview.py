import numpy as np
import matplotlib.pyplot as plt

with open('top.txt', 'r') as fd:
    text = fd.readlines()
    text = [t.strip() for t in text]
    text = list(filter(lambda t: len(t) > 0, text))
    mem = [t.split()[9] for t in text]
    mem = [float(m) for m in mem]
    fig = plt.figure()
    axe = fig.add_subplot(111)
    axe.plot(mem, '-s')
    axe.set_title('%MEM')
    axe.set_xlabel('Time')
    axe.set_ylabel('%MEM')
    fig.savefig('top.png', dpi=800)
    plt.show()