"""Construit et exécute le notebook IFRS 9 : narration + appels courts au paquet, tout le calcul est dans `credit_risk_graph.ifrs9`."""
from __future__ import annotations

from pathlib import Path

import mlflow
import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

from credit_risk_graph.ifrs9.tracking import tracked_run

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "ifrs9_ecl_ebm.ipynb"

CELLS: list[tuple[str, str]] = [
    ("md", """# Pipeline IFRS 9 — PD par Explainable Boosting Machine, ECL et gouvernance

**Périmètre.** Défaut, PD, calibration, backtesting, LGD, EAD, staging, scénarios macro, ECL, explicabilité, monitoring. Le calcul vit dans le paquet `credit_risk_graph.ifrs9` ; ce notebook lit les sorties versionnées (DVC) et journalisées (MLflow) et les commente.

**Avertissement.** Les données sont **synthétiques**. Le modèle est *aligné sur* les méthodes de Bâle III et d'IFRS 9, il n'est ni validé ni approuvé au sens réglementaire (voir `docs/regulatory_mapping.md`). Les jeux de données sont régénérables à l'identique (graine 42) ; le calendrier de simulation va de 2016 à 2024.

**Reproductibilité.** `dvc repro` exécute toutes les étapes ; chaque étape crée un run MLflow (expérience `ifrs9-ebm-pipeline`) avec l'intégralité de `params.yaml`, la version du code et l'empreinte des données."""),
    ("code", """import json
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

from credit_risk_graph.ifrs9 import plots
from credit_risk_graph.ifrs9.params import METRICS_DIR, TABLES_DIR, load_params
from credit_risk_graph.ifrs9.stages.common import load_model
from credit_risk_graph.ifrs9.tracking import data_version

warnings.filterwarnings("ignore")
plots.set_style()
pd.options.display.float_format = "{:,.4f}".format
params = load_params()
table = lambda name: pd.read_csv(TABLES_DIR / f"{name}.csv")
metrics = lambda name: json.loads((METRICS_DIR / f"{name}.json").read_text(encoding="utf-8"))
print("Version des données (dvc.lock) :", data_version())"""),
    ("md", "## 1. Analyse exploratoire\n\nContrôles de réalisme du calendrier, structure du portefeuille, taux de défaut et stress macro."),
    ("code", """from credit_risk_graph.ifrs9.eda import calendar_report, portfolio_overview, default_rate_by

display(calendar_report())
display(portfolio_overview())"""),
    ("code", """fig = plots.default_rate_over_time(default_rate_by("mois"))
plt.show()
display(default_rate_by("product_type"))
display(default_rate_by("region"))"""),
    ("md", "## 2. Définition du défaut\n\nDéfaut = retard de paiement d'au moins 90 jours (Bâle CRE36.68, Instruction BCEAO 026-11-2016). Deux variantes sont mesurées en sensibilité : ajout d'un indicateur d'improbabilité de paiement (restructuration en retard) et période de guérison. Le libellé d'entraînement reste le défaut à 90 jours."),
    ("code", """from credit_risk_graph.ifrs9.eda import definition_sensitivity_report

display(definition_sensitivity_report())"""),
    ("md", "## 3. Modèle de PD : Explainable Boosting Machine\n\nL'EBM est un modèle additif de type GAM boosté : la PD est la somme de fonctions de forme par variable (et de quelques interactions), donc **lisible terme à terme**. Trois profils de variables sont comparés (du plus simple au plus riche) ; le profil champion est le plus simple qui reste à moins de 0,005 de Gini du meilleur et qui contient une variable macro, condition d'une PD prospective (IFRS 9 §5.5.17). Des contraintes de monotonie (retard, taux d'effort, couverture de la garantie) sont imposées après vérification empirique de leur signe."),
    ("code", """comparison = table("pd_model_comparison")
splits = ["train", "validation", "test", "out_of_time"]
for measure in ("auc", "gini", "pr_auc"):
    print(measure.upper())
    display(comparison.pivot_table(index=["profil", "modele"], columns="split", values=measure).reindex(columns=splits))
print("Profil champion :", metrics("pd_model")["champion_profile"])"""),
    ("md", """### Recherche des hyperparamètres

Les hyperparamètres de l'EBM sont choisis par **recherche aléatoire** (et non par grille exhaustive) avec une validation croisée **temporelle** à fenêtre croissante : ajustement sur le passé, mesure sur une année postérieure. Un découpage aléatoire serait faussé, car les instantanés mensuels d'un même prêt sont corrélés. Le critère est l'AUC moyenne des plis ; chaque configuration est journalisée dans MLflow."""),
    ("code", """tuning = table("ebm_tuning")
display(tuning.head(8))
print("Meilleure configuration :", metrics("ebm_tuning")["best_params"])"""),
    ("md", """### Diagnostic : pourquoi les variables de graphe sont écartées

Le profil `graph_enriched` n'est jamais éligible (variables de tiers, calculées sur toute la période d'apprentissage). Son comportement l'illustre : très bon en apprentissage, il s'effondre hors période, signature d'une information du futur."""),
    ("code", """ebm = comparison[comparison["modele"] == "ebm"].pivot_table(index="profil", columns="split", values="auc").reindex(columns=splits)
ebm["ecart_apprentissage_hors_periode"] = ebm["train"] - ebm["out_of_time"]
display(ebm)"""),
    ("code", """importance = table("pd_ebm_importance")
plots.importance(importance)
plt.show()"""),
    ("md", "## 4. Calibration\n\nLa PD brute est recalée sur les taux de défaut observés (échantillon de validation). Les méthodes Platt (monotone, lisse) et isotonique sont comparées ; la pente et l'ordonnée à l'origine de calibrage valent 1 et 0 pour un calibrage parfait."),
    ("code", """calib = table("pd_calibration_metrics")
display(calib.pivot_table(index=["split", "pd"], values=["ece", "brier", "slope", "intercept", "calibration_in_the_large"]))"""),
    ("code", """from credit_risk_graph.ifrs9.eda import reliability_inputs

y, probs = reliability_inputs("out_of_time")
plots.reliability(y, probs)
plt.show()"""),
    ("md", "## 5. Backtesting\n\nCohortes annuelles indépendantes : PD au 1er janvier, défaut observé dans les 12 mois. Les tests sont conduits par grade et par année (binomial, ajusté de la corrélation de défaut, Jeffreys, Hosmer-Lemeshow) avec des feux tricolores à 95 % et 99,99 % de probabilité cumulée. Seules les années `test` et `out_of_time` sont hors échantillon ; les autres années sont en échantillon et servent de contexte."),
    ("code", """portfolio = table("backtest_portfolio")
display(portfolio[["annee", "split", "exposures", "defaults", "mean_pd", "observed_dr", "p_binomial", "p_vasicek", "p_jeffreys", "hl_pvalue", "feu"]])
print(metrics("backtest"))"""),
    ("code", """plots.traffic_light_grid(table("backtest_by_grade"))
plt.show()
plots.gini_stability(table("backtest_discrimination"))
plt.show()"""),
    ("md", "## 6. LGD\n\nLa LGD est apprise par un EBM de régression sur les expositions en défaut (perte réalisée simulée). La variable `base_lgd`, fonction déterministe de la garantie, est exclue. La LGD de ralentissement est la LGD moyenne dans les périodes de stress macro élevé, moins la LGD moyenne ; les planchers prudentiels ne s'appliquent que dans la vue prudentielle."),
    ("code", """lgd = metrics("lgd_ead")
display(pd.DataFrame(lgd["lgd_evaluation"]).T)
print("Add-on de ralentissement :", round(lgd["downturn_add_on"], 4))"""),
    ("md", """## 7. EAD

Les prêts sont à terme, sans engagement hors bilan : le facteur de conversion vaut 1 et l'exposition suit l'échéancier d'amortissement contractuel. L'erreur de projection à 12 mois est mesurée contre les soldes réellement observés. Un biais **négatif** signifie que l'échéancier contractuel *sous-estime* l'exposition : retards et paiements partiels ralentissent l'amortissement réel, ce qui est le sens non prudent. Un facteur de glissement d'amortissement, estimé sur la validation et le test, majore donc l'exposition projetée ; les colonnes `_ajuste` donnent le biais résiduel."""),
    ("code", """display(table("ead_projection_error"))"""),
    ("md", "## 8. Staging\n\nStage 3 : retard d'au moins 90 jours. Stage 2 : retard supérieur à 30 jours, restructuration, ou hausse significative du risque (PD 12 mois au moins 1,7 fois la PD d'octroi et hausse absolue d'au moins 1 point). Stage 1 sinon."),
    ("code", """ecl_stage = table("ecl_by_stage")
display(ecl_stage)
plots.ecl_by_stage(ecl_stage)
plt.show()"""),
    ("md", "## 9. Scénarios macro\n\nTrois scénarios pondérés (central, défavorable, favorable) : le stress macro est décalé de l'écart entre le quantile 90 % (ou 10 %) historique et la médiane, puis revient linéairement à son niveau courant. Le modèle est ré-évalué sous chaque trajectoire ; les pondérations sont des hypothèses de démonstration."),
    ("code", """display(table("ecl_by_scenario"))"""),
    ("md", "## 10. Perte de crédit attendue (ECL)\n\nECL = somme sur les mois de PD marginale × LGD × EAD × facteur d'actualisation (taux d'intérêt effectif) ; 12 mois en Stage 1, durée de vie en Stage 2, LGD × EAD en Stage 3. La vue prudentielle indicative (PD moyenne de long terme avec plancher, LGD de ralentissement avec planchers) est rapprochée de l'ECL ; il ne s'agit pas d'actifs pondérés."),
    ("code", """summary = metrics("ecl")
print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in summary.items() if k != "stage_mix"})
display(table("ecl_vs_prudential"))"""),
    ("md", "## 11. Explicabilité\n\nLe modèle étant additif, l'explication est **exacte et native** : la contribution de chaque terme à la PD d'un dossier se lit directement, sans approximation de type SHAP. Ci-dessous, les fonctions de forme des variables les plus importantes puis la décomposition de trois dossiers."),
    ("code", """from credit_risk_graph.ifrs9.pd_model import term_shape

model = load_model("pd_ebm_champion")
for term in importance["terme"].head(4):
    plots.shape(term_shape(model, term), term)
    plt.show()"""),
    ("code", """from credit_risk_graph.ifrs9.eda import local_explanations

display(local_explanations(model, n=3))"""),
    ("md", "## 12. Monitoring et dérive\n\nPSI de chaque variable et du score par rapport à la population de validation. Repères empiriques (Lewis, 1994) : moins de 0,10 stable, de 0,10 à 0,25 à surveiller, au-delà de 0,25 alerte — ce ne sont pas des seuils réglementaires."),
    ("code", """print(metrics("monitoring"))
plots.psi_bars(table("psi_features").query("split == 'out_of_time'"))
plt.show()
plots.psi_time(table("psi_score_monthly").query("split == 'out_of_time'"))
plt.show()"""),
    ("md", "## 13. Piste GNN\n\nLes variables de graphe financier (voisinage à risque, contagion) sont intégrées comme famille de variables du profil `graph_enriched`. Un GNN reste une **piste de recherche** : il ne participe à aucune sortie de ce pipeline et n'a pas été soumis à la chaîne de validation ci-dessus."),
    ("md", "## Limites\n\n- Données synthétiques : les niveaux de performance ne se transposent pas à un portefeuille réel.\n- Scénarios et pondérations macro : hypothèses de démonstration.\n- Ré-évaluation des PD futures avec le seul stress macro modifié (variables comportementales gelées).\n- Pas de période de probation entre stages ; pas d'actualisation des recouvrements en Stage 3.\n- Vue prudentielle indicative : pas de calcul d'actifs pondérés."),
]


def build() -> nbformat.NotebookNode:
    nb = new_notebook()
    nb.cells = [new_markdown_cell(src) if kind == "md" else new_code_cell(src) for kind, src in CELLS]
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    return nb


def main() -> None:
    nb = build()
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, NOTEBOOK)
    with tracked_run("notebook"):
        mlflow.log_artifact(str(NOTEBOOK), artifact_path="notebook")
    print(f"Notebook exécuté : {NOTEBOOK}")


if __name__ == "__main__":
    main()
