"""Shared serif typography and forge palette for the three blog experiments."""
from pathlib import Path
import os
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/disordered-minds-mplconfig')
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, font_manager

ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / 'assets' / 'disordered-minds'
FIGURES.mkdir(parents=True, exist_ok=True)
font_manager.fontManager.addfont(str(ROOT / 'assets' / 'fonts' / 'Alegreya.ttf'))

def configure(theme='light'):
    dark = theme == 'dark'
    colors = dict(
        ink='#e0c7a1' if dark else '#62472f',
        accent='#dc976c' if dark else '#a65332',
        muted='#b9a285' if dark else '#8a765c',
        paper='#191918' if dark else '#ffffff',
        rule='#45443f' if dark else '#dedbd5',
        secondary='#b8b5ac' if dark else '#62605c',
    )
    plt.rcParams.update({
        'font.family': 'Alegreya', 'font.size': 12,
        'mathtext.fontset': 'stix', 'axes.titlesize': 12,
        'axes.labelsize': 12, 'xtick.labelsize': 10.5, 'ytick.labelsize': 10.5,
        'legend.fontsize': 10.5, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.edgecolor': colors['rule'], 'axes.labelcolor': colors['ink'],
        'text.color': colors['ink'], 'xtick.color': colors['secondary'],
        'ytick.color': colors['secondary'], 'axes.facecolor': colors['paper'],
        'figure.facecolor': colors['paper'], 'savefig.facecolor': colors['paper'],
        'legend.frameon': False, 'lines.linewidth': 1.6,
    })
    return colors

def save_figure(fig, stem, theme):
    fig.savefig(FIGURES / f'{stem}-{theme}.png', dpi=220, bbox_inches='tight')
    if theme == 'light':
        fig.savefig(FIGURES / f'{stem}.pdf', bbox_inches='tight')
    plt.close(fig)
