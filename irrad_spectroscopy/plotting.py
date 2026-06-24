"""Plotting functions for gamma spectroscopy."""
import numpy as np


def plot_signal_vs_background(
    energies,
    sig_counts,
    bg_scaled,
    net_counts,
    sig_live,
    bg_live,
    scale,
    output_path,
    show=False,
):
    """Plot signal, background, and net spectrum in 3 panels.

    Parameters
    ----------
    energies : ndarray
        Energy values (keV)
    sig_counts : ndarray
        Signal counts
    bg_scaled : ndarray
        Background counts (scaled)
    net_counts : ndarray
        Net counts (signal - background)
    sig_live : float
        Signal live time (seconds)
    bg_live : float
        Background live time (seconds)
    scale : float
        Background scale factor
    output_path : str or Path
        Output file path
    show : bool
        Show plot interactively
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    pos = net_counts > 0
    net_min = max(net_counts[pos].min() * 0.5, 1) if pos.any() else 1

    axes[0].errorbar(
        energies, sig_counts, yerr=np.sqrt(sig_counts),
        marker=".", lw=0.5, ls="None", color="C0",
    )
    axes[0].set_ylabel("Counts")
    axes[0].set_title(f"Signal measurement ({sig_live / 3600:.1f} h live time)")
    axes[0].set_yscale("log")
    sig_pos = sig_counts[sig_counts > 0]
    axes[0].set_ylim(bottom=max(sig_pos.min() * 0.5, 1) if len(sig_pos) > 0 else 1)
    axes[0].grid(True, alpha=0.3)

    axes[1].errorbar(
        energies, bg_scaled, yerr=np.sqrt(bg_scaled),
        marker=".", lw=0.5, ls="None", color="C1",
    )
    axes[1].set_ylabel("Counts")
    axes[1].set_title(f"Background ({bg_live / 3600:.1f} h live time, scaled by {scale:.4f})")
    axes[1].set_yscale("log")
    bg_pos = bg_scaled[bg_scaled > 0]
    axes[1].set_ylim(bottom=max(bg_pos.min() * 0.5, 1) if len(bg_pos) > 0 else 1)
    axes[1].grid(True, alpha=0.3)

    axes[2].errorbar(
        energies, net_counts, yerr=np.sqrt(net_counts),
        marker=".", lw=0.5, ls="None", color="C2",
    )
    axes[2].axhline(0, color="k", lw=0.5, ls="--")
    axes[2].set_ylabel("Net counts")
    axes[2].set_xlabel("Energy / keV")
    axes[2].set_title("Signal - Background")
    axes[2].set_yscale("log")
    axes[2].set_ylim(bottom=net_min)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    print(f"Signal vs Background plot saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_fitted(energies, net_counts, found_peaks, bkg, output_path, show=False, title=None):
    """Plot fitted spectrum with peak fits.

    Parameters
    ----------
    energies : ndarray
        Energy values (keV)
    net_counts : ndarray
        Net counts
    found_peaks : dict
        Fitted peak data
    bkg : callable
        Background interpolation function
    output_path : str or Path
        Output file path
    show : bool
        Show plot interactively
    title : str, optional
        Plot title
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from irrad_spectroscopy.spectroscopy import gauss, lin

    fig, ax = plt.subplots(figsize=(14, 7))

    mask = net_counts > 0
    ax.errorbar(
        energies[mask], net_counts[mask], yerr=np.sqrt(net_counts[mask]),
        marker=".", markersize=1, lw=0.4, ls="None",
        color="steelblue", alpha=0.7, label="Net spectrum", zorder=2,
    )

    bkg_vals = bkg(energies)
    ax.plot(
        energies, bkg_vals,
        color="goldenrod", lw=1.0, ls="--",
        label="Interpolated background", zorder=3,
    )

    colors = plt.cm.tab10(np.linspace(0, 1, min(len(found_peaks), 10)))
    color_cycle = [colors[i % len(colors)] for i in range(len(found_peaks))]

    for (name, pk), color in zip(
        sorted(found_peaks.items(), key=lambda x: x[1]["peak_fit"]["popt"][0]),
        color_cycle,
    ):
        popt = pk["peak_fit"]["popt"]
        mu, sigma, height = popt
        low_e, high_e = pk["peak_fit"]["int_lims"]
        bkg_type = pk["background"]["type"]
        bkg_popt = pk["background"]["popt"]

        x_fit = np.linspace(low_e, high_e, 200)
        gauss_vals = gauss(x_fit, mu, sigma, height)
        bkg_local = (
            lin(x_fit, *bkg_popt) if bkg_type == "local"
            else np.full_like(x_fit, bkg_popt[1])
        )
        total = gauss_vals + bkg_local

        ax.fill_between(x_fit, total, bkg_local, color=color, alpha=0.25, zorder=4)
        ax.plot(x_fit, total, color="red", lw=1.2, ls="--", zorder=5)
        ax.plot(x_fit, bkg_local, color="k", lw=0.8, ls=":", zorder=5)

        label_text = f"{name}\n{mu:.1f} keV"
        y_text = height + bkg_local[np.argmin(np.abs(x_fit - mu))]
        ax.annotate(
            label_text, xy=(mu, y_text), xytext=(0, 12),
            textcoords="offset points", fontsize=6, ha="center", va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="gray", alpha=0.8),
            arrowprops=dict(arrowstyle="->", color="gray", lw=0.5), zorder=6,
        )

    ax.set_xlabel("Energy / keV", fontsize=12)
    ax.set_ylabel("Counts", fontsize=12)
    ax.set_title(title or "Background-Subtracted Spectrum with Peak Fits", fontsize=13)
    ax.set_yscale("log")
    data_max = net_counts[mask].max() if mask.any() else 1
    ax.set_ylim(bottom=max(net_counts[mask].min() * 0.3, 1), top=data_max * 3)
    ax.set_xlim(left=max(energies[0], 0))
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, which="major", alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    print(f"Fitted plot saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_raw(energies, counts, output_path, show=False, title=None):
    """Plot raw spectrum.

    Parameters
    ----------
    energies : ndarray
        Energy values (keV)
    counts : ndarray
        Count values
    output_path : str or Path
        Output file path
    show : bool
        Show plot interactively
    title : str, optional
        Plot title
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.errorbar(
        energies, counts, yerr=np.sqrt(counts),
        marker=".", lw=0.5, ls="None", color="steelblue",
    )
    ax.set_xlabel("Energy / keV")
    ax.set_ylabel("Counts")
    ax.set_title(title or "Raw Spectrum")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(counts[counts > 0].min() * 0.5, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    print(f"Raw plot saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close()


def plot_net(energies, net_counts, output_path, show=False, title=None):
    """Plot background-subtracted (net) spectrum.

    Parameters
    ----------
    energies : ndarray
        Energy values (keV)
    net_counts : ndarray
        Net counts
    output_path : str or Path
        Output file path
    show : bool
        Show plot interactively
    title : str, optional
        Plot title
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12, 6))
    pos = net_counts > 0
    ax.errorbar(
        energies[pos], net_counts[pos], yerr=np.sqrt(net_counts[pos]),
        marker=".", lw=0.5, ls="None", color="seagreen",
    )
    ax.set_xlabel("Energy / keV")
    ax.set_ylabel("Net counts")
    ax.set_title(title or "Background-Subtracted")
    ax.set_yscale("log")
    ax.set_ylim(bottom=max(net_counts[pos].min() * 0.5, 1) if pos.any() else 1)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(str(output_path), bbox_inches="tight")
    print(f"Net plot saved to {output_path}")
    if show:
        plt.show()
    else:
        plt.close()
