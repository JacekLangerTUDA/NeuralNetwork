# This is a sample Python script.
import json
import random as ran
from datetime import datetime as dt
from json import JSONEncoder
from math import e

import numpy as np


# Press Umschalt+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

def sig(x):
    return 1. / (1. + e ** -x)


LABELS = open("data/train-labels.idx1-ubyte", "rb").read()
IMAGES = np.frombuffer(open("data/train-images.idx3-ubyte", "rb").read(), dtype="ubyte") / 254
WEIGHTS_PATH = "weigths.json"
TOTAL_IMAGE_COUNT = 60000


def image(index):
    return np.array(IMAGES[index + 16:index + 16 + 784], ndmin=2).T


def label(index):
    return LABELS[index + 8]


def printBar(lbl, current, total, start):
    cur = (current / total) * 100.
    bars = "=" * int(cur / 10)
    ws = " " * int(10 - (cur / 10))
    bar = f"|{bars}>{ws}|"
    time = dt.now() - start
    # display progress
    print("%5.2f \t%s\t %s testing %s" % (cur, bar, time, lbl), flush=True, end='\r')


class NeuralNetwork:

    def __init__(self, who, whi):
        self.final_out = None
        self.who = who
        self.whi = whi

    def evolve(self, gen, learningrate=.1):
        total = gen * TOTAL_IMAGE_COUNT
        start = dt.now()
        for i in range(gen):
            for j in range(TOTAL_IMAGE_COUNT):
                r = ran.randint(0, TOTAL_IMAGE_COUNT - 1)
                lbl = label(r)
                inp = image(r)
                printBar(lbl, (i + 1) * j, total, start)
                # generate target array
                tar = np.array([0] * 10, ndmin=2).T
                tar[lbl] = 1

                self.train(tar, inp, learningrate)

    def eval(self, inp, lbl):
        print("\n")
        # calculate the hidden layer
        hid = np.dot(self.whi, inp)
        hid_out = sig(hid)
        out = np.dot(self.who, hid_out)
        self.final_out = sig(out)
        print(f"expected {lbl}")
        for i in range(10):
            print("%d: %5.2f" % (i, out[i][0]))

    def train(self, target, inp, learningrate):
        # calculate the hidden layer
        lr = learningrate
        hid = np.dot(self.whi, inp)
        hid_out = sig(hid)
        out = np.dot(self.who, hid_out)
        self.final_out = sig(out)
        # calculate the individual error
        err = target - self.final_out
        hid_err = np.dot(self.who.T, err)
        # correct hidden output weigths
        # todo(jack): assess this function, the values produced are totally of the charts.
        output_error_m = err * self.final_out * (1 - self.final_out) * out
        self.who += lr * np.dot(output_error_m, np.transpose(hid_out))
        # correct hidden input weights
        hidden_error_m = hid_err * hid_out * (1 - hid_out) * hid
        self.whi += lr * np.dot(hidden_error_m, np.transpose(inp))


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


# convert to array and save to file
def save(whi, who):
    data = {"whi": whi,
            "who": who}
    encodedNumpyData = json.dumps(data, cls=NumpyArrayEncoder)
    f = open(WEIGHTS_PATH, "w").write(encodedNumpyData)


# load from file
def load(path=WEIGHTS_PATH, weights="whi"):
    data = open(path, "r").read()
    decoded = json.loads(data)
    return np.asarray(decoded[weights])


if __name__ == '__main__':

    try:
        whi = load()
        who = load(weights="who")
    except FileNotFoundError:
        whi = np.random.random((89, 784)) * .1
        who = np.random.random((10, 89)) * .1
    net = NeuralNetwork(
        whi=whi,
        who=who)
    start = dt.now()
    net.evolve(gen=1)
    end = dt.now()
    r = ran.randint(0, TOTAL_IMAGE_COUNT - 1)
    net.eval(image(r), label(r))
    print(f"duration {end - start}")
    save(net.whi, net.who)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
