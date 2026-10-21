"""Create analysis.ipynb (executed afterwards with nbconvert)."""
import nbformat as nbf
nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# BEED study — analysis walkthrough\n\nThis notebook walks through the main pipeline: data audit, provenance check, "
       "leakage diagnostics, a random-versus-blocked comparison, and the stored results produced by `run_all.py`. "
       "All numbers are computed live from the code in `src/beed` or loaded from `results/`."),
    code("import sys, json\nsys.path.insert(0, 'src')\nimport numpy as np, pandas as pd\n"
         "from beed import config as C, data, features, evaluation as E, experiments as X\n"
         "pd.set_option('display.width', 160); pd.set_option('display.max_columns', 30)"),
    md("## 1. Load data and basic checks"),
    code("df, Xr, y, pos = data.load_beed()\nprint(df.shape); print(df.y.value_counts().sort_index())\n"
         "print('duplicate rows:', df.duplicated().sum())\nprint('label changes at rows:', np.where(np.diff(y)!=0)[0])"),
    md("## 2. What do X1–X16 represent? Within-row versus across-row autocorrelation"),
    code("for c in range(4):\n    Xc = Xr[y==c]; Cc = np.corrcoef(Xc.T)\n"
         "    within = np.mean(np.diag(Cc,1)); across = np.mean([np.corrcoef(Xc[:-1,j], Xc[1:,j])[0,1] for j in range(16)])\n"
         "    print(f'class {c}: within-row lag-1 r = {within:.3f}; across-row lag-1 r = {across:.3f}; SD = {Xc.std():.1f}')"),
    md("Class 0 behaves like 16 consecutive samples of one signal (strong within-row, no across-row correlation); "
       "classes 1–3 behave like 16 channels sampled at consecutive time points (strong across-row correlation)."),
    md("## 3. Provenance: exact matching of rows against the Bonn-derived UCI file"),
    code("data.ensure_bonn()\nbm, _ = data.match_to_bonn(Xr, y)\nprint(bm.groupby('beed_class').agg(n=('index','size'), "
         "labels=('bonn_label', lambda s: s.value_counts().to_dict()), offsets=('offset', lambda s: s.value_counts().to_dict()), "
         "segments=('bonn_segment','nunique')))"),
    md("## 4. Leakage diagnostic: nearest training neighbour under a random split"),
    code("pd.read_csv('results/leakage_nearest_neighbour.csv')"),
    md("## 5. Random versus blocked hold-out with a fixed LightGBM (quick re-run)"),
    code("from lightgbm import LGBMClassifier\ngroups = data.make_groups(y, pos, bm)\nF = features.build_feature_sets(Xr)['raw+eng'].values\n"
         "for prot in ['random', 'blocked']:\n    tr, te = data.holdout(prot, y, pos, groups)\n"
         "    m = LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.1, subsample=0.8, subsample_freq=1,\n"
         "                       colsample_bytree=0.8, random_state=C.SEED, verbose=-1, n_jobs=4).fit(F[tr], y[tr])\n"
         "    print(prot, {k: round(v, 4) for k, v in E.metrics(y[te], m.predict(F[te])).items()})"),
    md("## 6. Stored results from `run_all.py`"),
    code("h = pd.read_csv('results/holdout_metrics.csv')\n"
         "h[h.task=='4class'].pivot_table(index=['model','features'], columns='protocol', values='macro_f1').round(3)"),
    code("pd.read_csv('results/cv_summary.csv').round(3)"),
    code("pd.read_csv('results/stats_friedman.csv').round(4).T"),
    code("pd.read_csv('results/edge_deployment.csv').round(3)"),
    md("## 7. Key figures"),
    code("from IPython.display import Image, display\nfor f in ['fig_structure', 'fig_provenance', 'fig_block_curve', 'fig_holdout_bars', 'fig_shap']:\n"
         "    display(Image(filename=f'figures/{f}.png', width=900))"),
]
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "analysis.ipynb")
