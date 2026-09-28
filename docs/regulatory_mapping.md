# Cartographie réglementaire — IFRS 9 / Bâle III / UEMOA

Ce document relie chaque exigence identifiée aux composants du pipeline et précise ce qui est réellement implémenté. Les sources sont celles consultées lors de la revue documentaire ; les points marqués **[à recouper]** viennent de résumés secondaires (le texte officiel n'a pas pu être lu directement) et doivent être vérifiés avant toute citation dans un document de validation.

## Positionnement

Le modèle est entraîné sur des **données synthétiques**. Il est **aligné sur** les méthodes et les exigences citées ci-dessous, il n'est pas « validé », « approuvé » ni « conforme IRB ». Aucune PD de Bâle, LGD de ralentissement ou facteur de conversion n'est calibré sur un historique de pertes réel (Bâle exige au moins 5 ans d'historique dont une période de ralentissement — CRE36.79). Les scénarios macro et leurs pondérations sont des hypothèses de démonstration.

Le cadre juridique applicable aux banques de l'UEMOA est le dispositif prudentiel de la BCEAO (Bâle II/III, en vigueur depuis le 1er janvier 2018). Les lignes directrices EBA/GL/2017/16 visent les banques IRB de l'Union européenne : elles servent ici de référence méthodologique, pas d'obligation.

## Correspondance exigence → implémentation

| Sujet | Exigence | Implémentation | Source |
|---|---|---|---|
| Définition du défaut | Retard > 90 jours sur une obligation significative ; indicateurs d'improbabilité de paiement | `ifrs9/default_definition.py` : 90 jours, variante avec restructuration en retard (hypothèse 60 j), période de guérison paramétrable (3 mois). Le libellé d'entraînement reste le défaut à 90 jours ; les variantes sont mesurées en sensibilité | Bâle CRE36.68-69 ; Instruction BCEAO 026-11-2016 (90 j, 180 j pour PME et secteur public) ; IFRS 9 B5.5.37 **[à recouper]** |
| PD IFRS 9 | Estimation non biaisée, pondérée par les probabilités, avec information prospective raisonnable | PD point-in-time : EBM avec le stress macro comme variable, recalibré sur la validation, ré-évalué sous trois scénarios | IFRS 9 §5.5.17 **[à recouper]** |
| PD prudentielle | Moyenne de long terme des taux de défaut à un an, historique d'au moins 5 ans, plancher de PD | `ifrs9/basel_view.py` : PD moyenne de long terme par grade, plancher 0,05 % (retail hors revolving) | Bâle CRE36.63, CRE36.79, CRE32.58 |
| Staging (SICR) | Hausse significative du risque de crédit ; retard > 30 jours en filet de sécurité réfutable | `ifrs9/staging.py` : retard > 30 j, restructuration, ou PD 12 mois ≥ 1,7 × PD d'octroi avec hausse absolue ≥ 1 point (seuils internes) | IFRS 9 §5.5.9, §5.5.11, B5.5.17 **[à recouper]** |
| Calibration | Comparer taux de défaut réalisés et attendus ; revue au moins annuelle par une fonction indépendante | Calibrage Platt/isotonique sur la validation ; ECE, pente et ordonnée de calibrage ; revue automatisée à chaque exécution du pipeline | Bâle CRE36.56, CRE36.59 |
| Backtesting | Tests par grade : binomial, Hosmer-Lemeshow, feux tricolores | `ifrs9/backtesting.py` : cohortes annuelles indépendantes ; tests binomial, Vasicek (corrélation d'actifs 5 %, hypothèse), Jeffreys, Hosmer-Lemeshow ; feux vert/jaune/rouge à 95 % et 99,99 % de probabilité cumulée | Staff paper EBA sur le backtesting systématique de la PD ; Tasche (arXiv physics/0606071) |
| Stabilité du pouvoir discriminant | Aucun seuil réglementaire trouvé pour le Gini/AUC | Suivi annuel du Gini avec intervalle bootstrap : présenté comme choix interne | — |
| LGD | Planchers retail (30 % non garanti, 5 % immobilier) ; LGD de ralentissement | `ifrs9/lgd.py` : EBM de régression sur expositions en défaut ; add-on de ralentissement lu sur les trimestres de stress macro élevé ; planchers appliqués uniquement à la vue prudentielle | Bâle CRE32.58 ; EBA/GL/2017/16 **[à recouper]** |
| EAD | Plancher de 50 % du hors-bilan | `ifrs9/ead.py` : échéancier d'amortissement contractuel ; CCF = 1 (prêts à terme, pas de hors-bilan) ; erreur de projection mesurée contre les soldes observés | Bâle CRE32.36 |
| Scénarios macro | Plusieurs scénarios pondérés, sans nombre ni poids imposés | Trois scénarios (central 50 %, défavorable 30 %, favorable 20 %) ; chocs lus sur les quantiles 90 % / 10 % du stress macro historique ; retour progressif au niveau courant | IFRS 9 §5.5.17 **[à recouper]** ; pratique courante, pas une exigence |
| ECL | Perte attendue actualisée : 12 mois (Stage 1), durée de vie (Stage 2), défaut (Stage 3) | `ifrs9/ecl.py` : Σ PD marginale × LGD × EAD × facteur d'actualisation au taux d'intérêt effectif | IFRS 9 §5.5 |
| Dérive | Surveillance des populations et des scores | `monitoring/drift.py` : PSI/CSI par variable et PSI mensuel du score ; < 0,10 stable, 0,10-0,25 à surveiller, > 0,25 alerte (règle empirique de Lewis 1994, pas un seuil réglementaire) | Lewis (1994), repris par Siddiqi |
| Explicabilité et documentation | Documentation complète, challenge indépendant | EBM (modèle additif glass-box) : contributions par terme exactes, sans SHAP ; comparaison à la régression logistique et à LightGBM ; toutes les exécutions journalisées dans MLflow, données et pipeline versionnés par DVC | Fed SR 11-7 |

## Variables relationnelles (graphe) : exclues des modèles éligibles

Pour un crédit aux particuliers, une variable qui pénalise un emprunteur à cause du défaut de tiers (voisins de transaction, communauté, pairs d'un employeur, garant, part de transactions avec des contreparties à risque) pose trois problèmes : explicabilité individuelle du refus, protection des données personnelles de personnes qui ne sont pas parties au contrat, et discrimination indirecte. Le texte de Bâle n'interdit pas ces variables en tant que telles ; le choix est un arbitrage de gouvernance, retenu ici par prudence.

Mise en œuvre :

- `features/schema.py` : `THIRD_PARTY_FEATURES` liste toutes les variables `graph_*` et `risky_tx_share_3m`.
- `ifrs9/pd_model.py` : `fit_pd_model` refuse tout profil éligible qui en contient (`ValueError`) ; un test unitaire vérifie cette garantie sur les profils déclarés dans `params.yaml`.
- Le profil `graph_enriched` n'est entraîné qu'en **diagnostic** : il n'est jamais champion, n'alimente ni l'ECL ni l'API. Ses variables sont en outre calculées sur l'ensemble de la période d'apprentissage (information du futur pour les dates antérieures), donc non point-in-time.
- Le GNN reste une piste de recherche, hors de toute sortie.

Points restant à examiner par une revue d'équité : la région, la taille du ménage et le type d'emploi peuvent servir de proxys de caractéristiques protégées ; ils sont conservés comme variables du client mais doivent faire l'objet d'un contrôle d'impact.

## Nature du portefeuille

Crédit de détail : prêts personnels, avances sur salaire, prêts d'équipement et micro-crédits de petites entreprises, accordés à des particuliers. Point ouvert : l'Instruction BCEAO 026-11-2016 retient 90 jours pour les créances douteuses mais 180 jours pour les PME ; le pipeline applique 90 jours à tous les produits (seuil de Bâle, plus prudent). Une variante à 180 jours pour le produit micro-PME est à trancher avec la banque.

## Périmètre Bâle III

Le pipeline couvre le **risque de crédit** : définition du défaut, PD, LGD, EAD, provisionnement, tests de résistance par scénarios macro (CRE32, CRE36). Il ne traite pas les autres volets de Bâle III : exigences de fonds propres et actifs pondérés (dont plancher de sortie), ratio de levier, ratios de liquidité (LCR, NSFR).

## Ce que le pipeline ne fait pas

- Pas de calcul d'actifs pondérés (aucune fonction de pondération IRB n'est implémentée) : la vue prudentielle est une perte attendue indicative.
- Pas de LGD calibrée sur des recouvrements réels : la LGD synthétique de défaut est la seule cible disponible.
- Pas de projection comportementale des variables au-delà de l'arrêté : la PD des années futures ré-évalue le modèle avec le seul stress macro modifié.
- Pas de période de probation entre stages (transition Stage 3 → 2 → 1).
- Pas d'actualisation des recouvrements en Stage 3.
- Le GNN reste une piste de recherche ; il n'intervient dans aucune sortie réglementaire.

## Sources

- BIS, Basel Framework CRE32 et CRE36 : https://www.bis.org/basel_framework/chapter/CRE/32.htm et https://www.bis.org/basel_framework/chapter/CRE/36.htm
- BCEAO, Instruction n° 026-11-2016 : https://www.bceao.int/fr/reglementations/instruction-ndeg-026-11-2016-relative-la-comptabilisation-et-levaluation-des
- BCEAO, dispositif prudentiel de l'UMOA : https://www.bceao.int/sites/default/files/2017-11/_annexe_decision_013_24_06_2016-_bceao-dispositif_prudentiel_de_l_umoa-2016-1.pdf
- EBA, lignes directrices sur l'estimation de la PD et de la LGD (EBA/GL/2017/16)
- EBA, staff paper sur le backtesting systématique de la PD
- Tasche, « Validation of internal rating systems and PD estimates » : https://arxiv.org/pdf/physics/0606071
- Federal Reserve, SR 11-7 : https://www.federalreserve.gov/boarddocs/srletters/2011/sr1107.pdf
- IFRS 9 §5.5 : résumés PwC et IFRS Community consultés ; texte officiel à recouper
