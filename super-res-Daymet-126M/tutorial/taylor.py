import numpy as np
import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
import mpl_toolkits.axisartist.floating_axes as FA
import mpl_toolkits.axisartist.grid_finder as GF

# Cookbook: https://projectpythia.org/advanced-viz-cookbook/notebooks/4-taylor-diagrams.html#overview
class TaylorDiagram:
    def __init__(self, refstd, fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """ Reference: https://gist.github.com/ycopin/3342888"""
        self.refstd = refstd
        self.smin, self.smax = srange[0] * refstd, srange[1] * refstd
        self.tmax = np.pi if extend else np.pi / 2

        # Set up grid locator and formatter for correlation (angular axis)
        rlocs = np.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            rlocs = np.concatenate((-rlocs[:0:-1], rlocs))
        tlocs = np.arccos(rlocs)
        gl1 = GF.FixedLocator(tlocs)
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        ghelper = FA.GridHelperCurveLinear(
            PolarAxes.PolarTransform(),
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        # Create figure and axis
        fig = fig or plt.figure()
        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Customize axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")

        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")
        ax.axis["bottom"].toggle(ticklabels=False, label=False)

        self.ax = ax.get_aux_axes(PolarAxes.PolarTransform())

        # Reference point
        self.ax.plot([0], self.refstd, 'ko', fillstyle='none',  ms=10, label=label)
        print(np.linspace(0.015, self.tmax).size)
        self.ax.plot(np.linspace(0.015, self.tmax), [self.refstd] * np.linspace(0.015, self.tmax).size, 'k--')
        self.samplePoints = [self.ax]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """Add sample to the diagram."""
        sample_point, = self.ax.plot(np.arccos(corrcoef), stddev, *args, **kwargs)
        self.samplePoints.append(sample_point)
        return sample_point

    def add_grid(self, *args, **kwargs):
        """Add grid."""
        self.ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """Add RMSD contours."""
        rs, ts = np.meshgrid(np.linspace(self.smin, self.smax), np.linspace(0, self.tmax))
        rms = np.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*np.cos(ts))
        return self.ax.contour(ts, rs, rms, levels, **kwargs)

def test():
    # Example usage
    ref_std = 1.0
    samples = [(1.1, 0.9, 'Model A'), (0.9, 0.85, 'Model B'), (1.2, 0.8, 'Model C')]

    fig = plt.figure(figsize=(8, 8))
    dia = TaylorDiagram(ref_std, fig=fig, rect=111, label="Reference")

    # Add samples
    markers = {
        0: 'o', 1: "v", 2: "^", 3: "<", 4: ">", 5: "s", 6:"p", 7: "q",  8: "8"
    }
    colors = plt.matplotlib.cm.jet(np.linspace(0, 1, len(samples)))
    for i, (std, corr, label) in enumerate(samples):
        dia.add_sample(std, corr, 
                    marker= markers[i % (i+1)], ms=10, ls='', mfc=colors[i], mec=colors[i],
                    label=label)

    # Add contours
    dia.add_contours(levels=5, colors='0.5')

    # Show plot
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1))
    plt.title("Taylor Diagram: Std Dev (Radial) and Correlation (Angular)")
    plt.show()
