from matplotlib import pyplot as plt


def plot_hist(values, x_label, filename, bins=None):
    plt.figure(figsize=(4, 3))

    plt.hist(values, bins=bins)

    plt.xlabel(x_label)
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(filename + ".png", dpi=300, format="png")

    plt.close()

def plot_bar(x_values, y_values, x_label, y_label, filename, y_errors=None):
    plt.subplots(1, 1, figsize=(4, 3))

    if y_errors:
        plt.errorbar(x_values, y_values, yerr=y_errors)
    else:
        plt.bar(x_values, y_values)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()

    plt.savefig(filename + ".png", dpi=300, format="png")

    plt.close()