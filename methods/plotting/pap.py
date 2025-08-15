import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

def plot_parallel_coords_with_kde(ax, results_agg, dict_results_disagg_MC, columns, soln_labels,
                                   ideal_direction='top', fontsize=10):
    # Define objective bounds
    if len(soln_labels) == 1:
        objmins = [11, 35, -100, 100]
        objmaxs = [26, 115, 31, 1000]
    else:
        objmins = [11, 20, -100, 100]
        objmaxs = [26, 115, 31, 1000]

    ressat_wcu = results_agg.loc[:, columns].copy()

    # Normalize objectives
    tops, bottoms = np.array(objmaxs), np.array(objmins)
    tops[-1], bottoms[-1] = bottoms[-1], tops[-1]

    if ideal_direction == 'bottom':
        ressat_wcu.iloc[:, :-1] = (bottoms[:-1] - ressat_wcu.iloc[:, :-1]) / (bottoms[:-1] - tops[:-1])
        ressat_wcu.iloc[:, -1] = (ressat_wcu.iloc[:, -1] - bottoms[-1]) / (tops[-1] - bottoms[-1])
    elif ideal_direction == 'top':
        ressat_wcu.iloc[:, -1] = (bottoms[-1] - ressat_wcu.iloc[:, -1]) / (bottoms[-1] - tops[-1])
        ressat_wcu.iloc[:, :-1] = (ressat_wcu.iloc[:, :-1] - bottoms[:-1]) / (tops[:-1] - bottoms[:-1])
    else:
        raise ValueError('ideal_direction must be "top" or "bottom"')

    # Plot background lines
    for i in range(ressat_wcu.shape[0] - 1):
        for j in range(len(columns) - 1):
            y = [ressat_wcu.iloc[i, j], ressat_wcu.iloc[i, j + 1]]
            x = [j, j + 1]
            ax.plot(x, y, c='0.8', alpha=0.4, zorder=1, lw=1)

    # Axis lines and ticks
    for j in range(len(columns)):
        ax.annotate(str(round(tops[j])), [j, 1.02], ha='center', va='bottom', fontsize=fontsize)
        bottom_label = f"{round(bottoms[j])}+" if j == len(columns) - 1 else str(round(bottoms[j]))
        ax.annotate(bottom_label, [j, -0.02], ha='center', va='top', fontsize=fontsize)
        ax.plot([j, j], [0, 1], c='k', zorder=2)
        for y in np.arange(0, 1.001, 0.2):
            ax.plot([j - 0.03, j + 0.03], [y, y], c='k', zorder=2)
    
    # Clean aesthetics
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_ylim(-0.4, 1.1)
    ax.patch.set_alpha(0)

    if ideal_direction == 'top':
        ax.arrow(-0.15, 0.1, 0, 0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    elif ideal_direction == 'bottom':
        ax.arrow(-0.15, 0.9, 0, -0.7, head_width=0.08, head_length=0.05, color='k', lw=1.5)
    ax.annotate('Direction of preference', xy=(-0.3, 0.5), ha='center', va='center', rotation=90, fontsize=fontsize)

    # Axis labels
    labels = ['Number\nof\npartners', 'Captured\nwater\ngain\n(GL/yr)',
              'Captured\nwater\ngain for\nnon-partners\n(GL/yr)', 'Cost of\ngains for\nworst-off\npartner\n($/ML)']
    for i, l in enumerate(labels):
        ax.annotate(l, xy=(i, -0.12), ha='center', va='top', fontsize=fontsize)

    # Highlight selected solutions
    colors_brewer = {'s3/soln212': '#1b9e77', 'statusquo/soln0': '#d95f02', 'other': '#7570b3'}
    for soln_label in soln_labels:
        c = colors_brewer.get(soln_label, 'blue')
        soln_data = ressat_wcu.loc[results_agg['label'] == soln_label]
        xx, yy = [], []
        for j in range(len(columns) - 1):
            x = np.linspace(j, j + 1, 11)
            y = soln_data.iloc[0, j] + (x - j) * (soln_data.iloc[0, j + 1] - soln_data.iloc[0, j])
            xx += list(x); yy += list(y)
        ax.plot(xx, yy, c='k', lw=3, zorder=5)
        ax.plot(xx, yy, c=c, lw=2, zorder=6)
        
        # KDE shading
        results_MC = dict_results_disagg_MC[soln_label].copy()
        columns_MC = ['cog_wp' if col == 'cog_wp_p90' else col for col in columns]
        for o, obj in enumerate(columns_MC):
            scaled_col = f"{obj}_scaled"
            if obj == 'cog_wp':
                results_MC[scaled_col] = (results_MC[obj] - bottoms[o]) / (tops[o] - bottoms[o]) \
                    if ideal_direction == 'bottom' else (bottoms[o] - results_MC[obj]) / (bottoms[o] - tops[o])
            else:
                results_MC[scaled_col] = (bottoms[o] - results_MC[obj]) / (bottoms[o] - tops[o]) \
                    if ideal_direction == 'bottom' else (results_MC[obj] - bottoms[o]) / (tops[o] - bottoms[o])

        for o, obj in enumerate(columns_MC[1:]):
            y = np.arange(0, 1, 0.01)
            data = results_MC[f'{obj}_scaled']
            kde = sm.nonparametric.KDEUnivariate(data)
            kde.fit(bw=0.025)
            x = np.array([kde.evaluate(v)[0] * 0.12 if not np.isnan(kde.evaluate(v)[0]) else 0 for v in y])
            ax.fill_betweenx(y, x + o + 1, o + 1, where=(x > 0.00005), lw=1, alpha=0.6, zorder=4, fc=c, ec='k')

        # Debug print
        print(f"{soln_label} min, mean, max:")
        for o, obj in enumerate(columns_MC[1:]):
            print(f"{obj}: {results_MC[obj].min():.2f}, {results_MC[obj].mean():.2f}, {results_MC[obj].max():.2f}")
        print()

    ax.set_xlim([-0.3, 4.3])
    ax.annotate('b)', (0.4, 0.93), xycoords='subfigure fraction', ha='left', va='top', fontsize=fontsize + 2, weight='bold')



#%%
# Define required solution labels
soln_labels = ['s3/soln212', 'statusquo/soln0', 'soln1', 'soln2', 'soln3']

# Create results_agg with necessary columns
results_agg = pd.DataFrame({
    'label': soln_labels,
    'n_partner': np.random.randint(11, 26, size=len(soln_labels)),
    'captured_gain': np.random.uniform(20, 115, size=len(soln_labels)),
    'captured_nonpartner': np.random.uniform(-100, 31, size=len(soln_labels)),
    'cog_wp_p90': np.random.uniform(100, 1000, size=len(soln_labels))
})
results_agg['cog_wp'] = results_agg['cog_wp_p90']

# Create dict_results_disagg_MC with matching keys
dict_results_disagg_MC = {
    label: pd.DataFrame({
        'n_partner': np.random.randint(11, 26, size=100),
        'captured_gain': np.random.uniform(20, 115, size=100),
        'captured_nonpartner': np.random.uniform(-100, 31, size=100),
        'cog_wp_p90': np.random.uniform(100, 1000, size=100),
        'cog_wp': np.random.uniform(100, 1000, size=100)
    })
    for label in soln_labels
}

# fig = plt.figure(figsize=(10, 6))
# gs = fig.add_gridspec(1, 2)
# ax = fig.add_subplot(gs[0, 1])

fig, ax = plt.subplots()
plot_parallel_coords_with_kde(
    ax=ax,
    results_agg=results_agg,  # Your preloaded DataFrame
    dict_results_disagg_MC=dict_results_disagg_MC,  # Your dictionary of MC samples
    columns=['n_partner', 'captured_gain', 'captured_nonpartner', 'cog_wp_p90'],
    soln_labels=['s3/soln212', 'statusquo/soln0'],
    ideal_direction='top',
    fontsize=10
)

plt.tight_layout()
plt.show()