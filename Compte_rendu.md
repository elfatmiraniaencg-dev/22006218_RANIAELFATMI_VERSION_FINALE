📘 GRAND GUIDE : ANATOMIE D’UN PROJET DATA SCIENCE — CAMPAGNE MARKETING BANCAIRE (BANK MARKETING UCI)

Ce document présente, de A à Z, la logique complète d’un projet de Machine Learning appliqué au marketing bancaire.
L’objectif : passer du statut « je lance un modèle » au statut « j’analyse, je décide, je justifie comme un ingénieur IA ».

1. Le Contexte Métier et la Mission
🎯 Le Problème (Business Case)

Une banque lance régulièrement des campagnes de télémarketing pour proposer à ses clients un dépôt à terme (Term Deposit).

Mais :

La plupart des clients disent non.

Appeler tout le monde coûte cher.

Les campagnes longues fatiguent les conseillers.

Le taux de conversion est faible (≈ 11% dans le dataset).

Objectif Business :

Créer un modèle de Machine Learning qui prédit si un client va dire “yes” et souscrire au produit.

L’Enjeu stratégique :

Optimiser les ressources humaines :

Réduire le nombre d’appels inutiles

Cibler les clients à haute probabilité

Augmenter le taux de conversion

Diminuer le coût par conversion

⚠️ Coût asymétrique des erreurs

Faux Positif (Prédit “yes” mais client = no)

La banque appelle inutilement → Coût en temps et argent

Faux Négatif (Prédit “no” mais client = yes)

Opportunité commerciale manquée → Perte d’argent direct

➡️ Dans ce domaine, on cherche à maximiser le Recall sur la classe “yes”
(capter le plus possible des futurs clients intéressés).

2. Le Code Python (Laboratoire)

Ton code + mon script complet te donnent un pipeline professionnel :

Chargement depuis UCI

Fusion X/y

EDA complet (histogrammes, heatmap, boxplots…)

Pipeline Scikit-Learn (imputation + encodage + RandomForest)

Évaluation finale

Tu peux rappeler ici la structure générale :

df = pd.concat([X, y], axis=1)
...
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(...))
])
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

3. Analyse Approfondie : Comprendre les Données (Profil Marketing)
📂 Le Dataset

Le jeu “Bank Marketing” contient :

41 188 appels téléphoniques

20 variables caractéristiques

1 variable cible : y (yes / no)

Types de variables
Type	Exemples	Rôle
Numériques	âge, durée de l’appel, taux d’emploi	Facteurs quantitatifs
Catégorielles	job, marital, education, contact	Profil socio-éco
Macroéconomie	euribor3m, cons.conf.idx	Contexte financier
Historique client	previous, poutcome	Impact de campagnes passées
💡 Insight clé

La variable duration est très puissante (fortement liée au résultat),
mais NE DOIT PAS être utilisée en production, car elle n’est connue qu’après l’appel.
Elle crée un Data Leakage naturel.

Ton modèle, lui, encode correctement toutes les variables catégorielles et nettoie les NaN via :

numeric_transformer = Pipeline([('imputer', SimpleImputer(strategy='median'))])
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

4. Analyse Exploratoire (EDA) — Lecture “Pro”
📊 A. Distribution de la cible

Le dataset est fortement déséquilibré :

88% : no

12% : yes

➡️ Un modèle naïf qui prédit “no” tout le temps fait déjà 88% d’accuracy.
Donc l’accuracy est une métrique trompeuse.

📈 B. Profilage des variables
1. Variables socio-économiques

Les profils “blue-collar” souscrivent moins

Les professions “management” et “technician” souscrivent davantage

2. Variables du comportement bancaire

duration (durée de l’appel) est très corrélée à la conversion
(les conversations longues → plus de chance d’un “yes”)

3. Variables macroéconomiques

Lorsque les taux d’intérêt euribor sont faibles → les clients ont tendance à souscrire

4. Multicorrélation

Heatmap :
Certaines colonnes macro présentent de fortes corrélations :

euribor3m

cons.price.idx

nr.employed

➡️ Cela ne gêne pas Random Forest, mais serait problématique pour une Régression Logistique.

5. Méthodologie : Split Train/Test

Ton code utilise :

train_test_split(X, y, test_size=0.2, random_state=42)

Pourquoi ?

Test Size 20%
Équilibre optimal entre :

assez de données pour entraîner un modèle robuste

assez de données pour tester sa généralisation

Random State = 42
Garantit la reproductibilité.

Problème du Data Leakage
Ton pipeline Scikit-Learn corrige cela automatiquement :

L’imputer et le OneHot sont entraînés uniquement sur le Train

Puis appliqués au Test

⭐ C’est la bonne pratique industrielle.

6. FOCUS THÉORIQUE : Pourquoi Random Forest fonctionne si bien ici ?

Le Random Forest est parfait pour :

✔ mélanger variables numériques + catégorielles
✔ gérer les interactions non linéaires
✔ survivre aux outliers
✔ gérer les corrélations fortes
✔ éviter l’overfitting grâce au bagging

🔥 A. Le principe : Une armée d’arbres

Chaque arbre :

voit un sous-échantillon différent de clients (bootstrapping)

apprend des règles différentes

se concentre sur un sous-ensemble aléatoire de colonnes

➡️ Diversité = Robustesse

🔥 B. Exemple concret dans ton dataset

Arbre 1 : classe selon l'âge et le job
Arbre 2 : classe selon duration et poutcome
Arbre 3 : classe selon macro-économie

Chacun a tort parfois…
Mais en votant tous ensemble → l’erreur de chacun s’annule.

🔥 C. Pourquoi la durée est risquée

L’arbre adore la variable duration.

Mais en production :

on ne connaît pas la durée d’un appel avant de téléphoner,

donc le modèle serait biaisé.

➡️ Il faut l’exclure pour un vrai modèle industriel.

7. Analyse Finales : Évaluation de l’IA Marketing
🧮 A. Matrice de Confusion

Pour une banque, les erreurs coûtent :

Faux Positifs (FP) : Appels inutiles → coût

Faux Négatifs (FN) : clients perdus → manque à gagner

Ton modèle RandomForest donne généralement :

✔ accuracy correcte
✔ recall moyen sur la classe “yes” (classe minoritaire et difficile)

📌 B. Les bonnes métriques marketing
1. Precision (classe yes)

Mesure de la qualité du ciblage.

Si faible → tu gaspilles des appels.

2. Recall (classe yes)

Critique ici :
"Parmi les clients réellement intéressés, combien ne ratais-tu pas ?"

Un bon modèle doit viser :
Recall élevé sur “yes”, même si Precision baisse un peu.

3. F1-score

Compromis entre Precision & Recall.

Conclusion Générale — Intelligence Marketing Basée IA

Ce projet montre que :

Le Machine Learning n’est pas une simple prédiction

C’est une stratégie business complète

L’analyse métier conditionne les métriques importantes (ici : Recall sur “yes”)

Le preprocessing structuré (pipeline) réduit les erreurs humaines

Random Forest est un excellent modèle pour un premier prototype

🎯 Ce modèle aide la banque à appeler moins… mais mieux
→ optimisation du budget
→ amélioration du taux de conversion
→ satisfaction des conseillers et des clients
