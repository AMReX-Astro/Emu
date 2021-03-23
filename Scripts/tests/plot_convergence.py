import numpy as np
import matplotlib.pyplot as plt

# Read forward Euler data
class ConvergenceData(object):
    def __init__(self, filename=None):
        self.data = {
            "cfl": [],
            "f_ee error": [],
            "f_xx error": [],
            "f_eebar error": [],
            "f_xxbar error": [],
        }

        if filename:
            self.readfrom(filename)

    def readfrom(self, filename):
        f = open(filename, "r")

        while True:
            entry = [f.readline().strip() for i in range(9)]
            if not entry[0]:
                break
            for line in entry:
                ls = line.split(":")
                name = ls[0].strip()
                value = ls[-1].strip()
                for k in self.data.keys():
                    if name == k:
                        self.data[k].append(float(value))

        f.close()

        for k in self.data.keys():
            self.data[k] = np.array(self.data[k])

    def get(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()

    def error_keys(self):
        return [k for k in self.data.keys() if k != "cfl"]

    def average_convergence(self, key):
        # get the average convergence order for the keyed quantity
        err = self.get(key)
        cfl = self.get("cfl")

        orders = []
        for i in range(len(err) - 1):
            order = np.log10(err[i + 1] / err[i]) / np.log10(cfl[i + 1] / cfl[i])
            orders.append(order)
        orders = np.array(orders)

        order_average = np.average(orders)
        return order_average

    def plot_on_axis(self, axis, key, label, color):
        log_cfl = np.log10(self.get("cfl"))
        log_err = np.log10(self.get(key))
        axis.plot(
            log_cfl, log_err, label=label, marker="o", linestyle="None", color=color
        )

        order = self.average_convergence(key)
        iMaxErr = np.argmax(log_err)
        intercept = log_err[iMaxErr] - order * log_cfl[iMaxErr]
        log_order_err = intercept + order * log_cfl
        axis.plot(
            log_cfl,
            log_order_err,
            label="$O({}) = {:0.2f}$".format(label, order),
            marker="None",
            linestyle="--",
            color=color,
        )


cdata = {}
cdata["fe"] = ConvergenceData("msw_test_fe.txt")
cdata["trapz"] = ConvergenceData("msw_test_trapz.txt")
cdata["ssprk3"] = ConvergenceData("msw_test_ssprk3.txt")
cdata["rk4"] = ConvergenceData("msw_test_rk4.txt")

variables = cdata["fe"].error_keys()

for v in variables:
    fig, ax = plt.subplots()

    ax.set_xlabel("log10 flavor CFL")
    ax.set_ylabel("log10 {}".format(v))

    colors = ["red", "blue", "green", "magenta"]

    for k, c in zip(cdata.keys(), colors):
        cd = cdata[k]
        cd.plot_on_axis(ax, v, k, c)

    ax.invert_xaxis()

    ax.legend(loc=(1.05, 0.0))
    fig.tight_layout()

    plt.savefig("convergence_{}.eps".format(v.replace(" ", "_")))
    plt.savefig("convergence_{}.png".format(v.replace(" ", "_")), dpi=300)
    plt.clf()
