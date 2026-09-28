from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PALETTE = {"principal": "#1F3A66", "accent": "#8A6D1F", "bon": "#2F7D4F", "alerte": "#B23A48", "neutre": "#6B7280"}
FEU_COLORS = {"vert": "#2F7D4F", "jaune": "#D9A21B", "rouge": "#B23A48"}


def set_style() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams.update({"figure.dpi": 110, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})


def default_rate_over_time(monthly: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(monthly["mois"], monthly["taux_defaut_12m"], color=PALETTE["principal"])
    ax.set_title("Taux de défaut à 12 mois par date d'arrêté")
    ax.set_ylabel("taux")
    fig.autofmt_xdate()
    return fig


def reliability(y: np.ndarray, probs: dict[str, np.ndarray], bins: int = 10) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color=PALETTE["neutre"], label="calibrage parfait")
    for name, prob in probs.items():
        edges = np.unique(np.quantile(prob, np.linspace(0, 1, bins + 1)))
        idx = np.clip(np.searchsorted(edges, prob, side="right") - 1, 0, len(edges) - 2)
        pred = [prob[idx == b].mean() for b in range(len(edges) - 1)]
        obs = [y[idx == b].mean() for b in range(len(edges) - 1)]
        ax.plot(pred, obs, marker="o", label=name)
    ax.set_xlabel("PD prédite (moyenne du décile)")
    ax.set_ylabel("taux de défaut observé")
    ax.set_title("Courbe de fiabilité")
    ax.legend()
    return fig


def importance(table: pd.DataFrame, top: int = 15) -> plt.Figure:
    data = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * top + 1))
    ax.barh(data["terme"], data["part"], color=PALETTE["principal"])
    ax.set_title("Importance globale des termes de l'EBM")
    ax.set_xlabel("part de l'importance totale")
    return fig


def shape(shape_df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.2))
    if "borne_inf" in shape_df:
        ax.step(shape_df["borne_inf"].astype(float), shape_df["contribution_logit"], where="post", color=PALETTE["principal"])
        ax.set_xlabel("valeur de la variable")
    else:
        ax.bar(shape_df["modalite"].astype(str), shape_df["contribution_logit"], color=PALETTE["principal"])
        ax.tick_params(axis="x", rotation=30)
    ax.axhline(0, color=PALETTE["neutre"], lw=0.8)
    ax.set_ylabel("contribution (logit)")
    ax.set_title(title)
    return fig


def gini_stability(discrimination: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.6))
    gini = 2 * discrimination["auc"] - 1
    ax.errorbar(discrimination["annee"], gini, yerr=[gini - (2 * discrimination["auc_ic95_bas"] - 1), (2 * discrimination["auc_ic95_haut"] - 1) - gini], fmt="o-", color=PALETTE["principal"], capsize=3)
    for _, row in discrimination[discrimination["split"].isin(["test", "out_of_time"])].iterrows():
        ax.axvspan(row["annee"] - 0.4, row["annee"] + 0.4, color=PALETTE["accent"], alpha=0.12)
    ax.set_title("Gini par cohorte annuelle (zones dorées : hors échantillon)")
    ax.set_ylabel("Gini")
    return fig


def traffic_light_grid(by_grade: pd.DataFrame) -> plt.Figure:
    grades, years = sorted(by_grade["grade"].unique()), sorted(by_grade["annee"].unique())
    fig, ax = plt.subplots(figsize=(1.0 * len(years) + 2, 0.5 * len(grades) + 1.5))
    for i, grade in enumerate(grades):
        for j, year in enumerate(years):
            row = by_grade[(by_grade["grade"] == grade) & (by_grade["annee"] == year)]
            if row.empty:
                continue
            row = row.iloc[0]
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=FEU_COLORS[row["feu"]], alpha=0.85))
            ax.text(j + 0.5, i + 0.5, f"{row['observed_dr']:.1%}\n({row['mean_pd']:.1%})", ha="center", va="center", fontsize=7, color="white")
    ax.set_xticks(np.arange(len(years)) + 0.5, years)
    ax.set_yticks(np.arange(len(grades)) + 0.5, [f"grade {g}" for g in grades])
    ax.set_xlim(0, len(years))
    ax.set_ylim(0, len(grades))
    ax.set_title("Feux tricolores par grade et par année — défaut observé (PD prédite)")
    return fig


def psi_bars(table: pd.DataFrame, top: int = 15) -> plt.Figure:
    data = table.head(top).iloc[::-1]
    colors = data["statut"].map({"stable": PALETTE["bon"], "à surveiller": FEU_COLORS["jaune"], "alerte": PALETTE["alerte"]})
    fig, ax = plt.subplots(figsize=(7, 0.35 * top + 1))
    ax.barh(data["variable"], data["psi"], color=colors)
    ax.axvline(0.10, ls="--", color=PALETTE["neutre"])
    ax.axvline(0.25, ls="--", color=PALETTE["alerte"])
    ax.set_title("PSI par variable (référence : validation)")
    return fig


def psi_time(scores: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(scores["mois"], scores["psi_score"], marker="o", color=PALETTE["principal"])
    ax.axhline(0.10, ls="--", color=FEU_COLORS["jaune"])
    ax.axhline(0.25, ls="--", color=PALETTE["alerte"])
    ax.set_title("PSI mensuel du score de PD")
    ax.tick_params(axis="x", rotation=60)
    return fig


def ecl_by_stage(table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.bar(table["stage"].astype(str), table["couverture"], color=PALETTE["principal"])
    ax.set_title("Taux de couverture (ECL / EAD) par stage")
    ax.set_xlabel("stage")
    return fig
