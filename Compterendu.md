
<p align="left">
  <img src="logoencgsettat.jpg" width="100" alt="Logo">
</p>

# 📘 GUIDE COMPLET : PROJET DATA SCIENCE - MARKETING BANCAIRE

<p align="center">
  <img src="image%20rania.jpg" width="350">
</p>


Réalisé par : RANIA EL FATMI 

Groupe : 2 FIN

---

## 1️⃣ Le Contexte Métier et la Mission

### 🎯 Le Problème (Business Case)

Une banque lance des campagnes de télémarketing pour proposer un **dépôt à terme** (Term Deposit).

**Problèmes actuels :**
- La plupart des clients refusent
- Appeler tout le monde coûte cher
- Les campagnes longues fatiguent les conseillers
- Taux de conversion faible : **≈ 11%**

### 🎯 Objectif Business

Créer un modèle ML qui prédit si un client va souscrire au produit.

**Enjeux stratégiques :**
- ✅ Réduire les appels inutiles
- ✅ Cibler les clients à haute probabilité
- ✅ Augmenter le taux de conversion
- ✅ Diminuer le coût par conversion

### ⚠️ Coût Asymétrique des Erreurs

| Type d'Erreur | Impact Business |
|---------------|-----------------|
| **Faux Positif** (prédit "yes" → client dit "no") | Coût : appel inutile |
| **Faux Négatif** (prédit "no" → client dit "yes") | **CRITIQUE** : opportunité perdue |

**➡️ Métrique prioritaire : RECALL sur la classe "yes"**

---

## 2️⃣ Structure du Code Python

```python
# 1. Chargement des données
df = pd.concat([X, y], axis=1)

# 2. Preprocessing Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 3. Modèle complet
model = Pipeline(steps=[
    ('preprocessing', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# 4. Entraînement et prédiction
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

---

## 3️⃣ Analyse des Données (Profil Marketing)

### 📂 Le Dataset "Bank Marketing"

- **41 188** appels téléphoniques
- **20** variables explicatives
- **1** variable cible : `y` (yes / no)

### Types de Variables

| Type | Exemples | Rôle |
|------|----------|------|
| **Numériques** | age, duration, euribor3m | Facteurs quantitatifs |
| **Catégorielles** | job, marital, education | Profil socio-économique |
| **Macro-économie** | cons.conf.idx, emp.var.rate | Contexte financier |
| **Historique** | previous, poutcome | Campagnes passées |

### 💡 Insight Clé : Variable "duration"

⚠️ **DATA LEAKAGE** : La durée de l'appel (`duration`) est très corrélée au résultat, MAIS elle n'est connue qu'**après** l'appel.

**➡️ NE PAS l'utiliser en production !**

---

## 4️⃣ Analyse Exploratoire (EDA)

### 📊 Distribution de la Cible

Le dataset est **déséquilibré** :
- **88%** : no
- **12%** : yes

**⚠️ Conséquence** : L'accuracy seule est trompeuse. Un modèle naïf qui prédit toujours "no" aurait 88% d'accuracy !

### 📈 Profilage des Variables

**1. Variables socio-économiques**
- Profils "blue-collar" → souscrivent moins
- Profils "management" / "technician" → souscrivent plus

**2. Comportement bancaire**
- `duration` élevée → forte corrélation avec "yes"

**3. Variables macro-économiques**
- Taux `euribor3m` faibles → plus de souscriptions

**4. Multicorrélation**
- Colonnes corrélées : `euribor3m`, `cons.price.idx`, `nr.employed`
- ✅ Pas de problème pour Random Forest
- ⚠️ Problématique pour Régression Logistique

---

## 5️⃣ Méthodologie : Split Train/Test

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Pourquoi ces paramètres ?

| Paramètre | Valeur | Raison |
|-----------|--------|--------|
| `test_size` | 0.2 (20%) | Équilibre train/test optimal |
| `random_state` | 42 | Reproductibilité des résultats |

### ✅ Prévention du Data Leakage

Le **Pipeline Scikit-Learn** garantit que :
- L'imputation et l'encodage sont ajustés uniquement sur le **train**
- Puis appliqués au **test**

**➡️ Bonne pratique industrielle**

---

## 6️⃣ FOCUS : Pourquoi Random Forest excelle ici ?

### Avantages pour ce Cas d'Usage

✅ Gère variables numériques + catégorielles  
✅ Capture les interactions non linéaires  
✅ Résiste aux outliers  
✅ Supporte les corrélations fortes  
✅ Évite l'overfitting (bagging)  

### 🔥 Principe : Une Armée d'Arbres

**Chaque arbre :**
1. Voit un sous-échantillon différent (bootstrapping)
2. Apprend des règles différentes
3. Se concentre sur des colonnes aléatoires

**➡️ Diversité = Robustesse**

### Exemple Concret dans Bank Marketing

```
Arbre 1 : classe selon [age, job]
Arbre 2 : classe selon [duration, poutcome]
Arbre 3 : classe selon [macro-économie]
```

Chacun se trompe parfois…  
**Mais le vote collectif annule les erreurs individuelles !**

---

## 7️⃣ Évaluation : Les Métriques qui Comptent

### 🧮 Matrice de Confusion

Pour une banque, les erreurs ont un coût :

| Erreur | Impact |
|--------|--------|
| **Faux Positifs (FP)** | Appels inutiles → coût opérationnel |
| **Faux Négatifs (FN)** | Clients perdus → **manque à gagner** |

### 📌 Métriques Marketing Critiques

**1. Precision (classe "yes")**
- Mesure la qualité du ciblage
- Si faible → gaspillage d'appels

**2. Recall (classe "yes")** ⭐ **PRIORITÉ**
- *"Parmi les clients réellement intéressés, combien capte-t-on ?"*
- **Objectif** : Maximiser le Recall, même si Precision baisse

**3. F1-Score**
- Compromis harmonique entre Precision & Recall

---

## 🎯 Conclusion : Intelligence Marketing Basée IA

### Ce Projet Démontre Que :

1. **Le ML n'est pas qu'une prédiction** → c'est une stratégie business complète
2. **L'analyse métier dicte les métriques** → ici : Recall sur "yes"
3. **Le preprocessing structuré** → réduit les erreurs humaines
4. **Random Forest** → excellent pour un premier prototype robuste

### Impact Business

| Avant ML | Après ML |
|----------|----------|
| Appels aléatoires | Ciblage intelligent |
| Taux conversion faible | Taux optimisé |
| Coût élevé | Budget optimisé |
| Équipes fatiguées | Productivité améliorée |

**🎯 Ce modèle aide la banque à appeler moins... mais mieux !**

---

## 📚 Pour Aller Plus Loin

### Améliorations Possibles

1. **Gestion du déséquilibre**
   - SMOTE (sur-échantillonnage)
   - Ajustement des poids de classe
   - Seuil de décision personnalisé

2. **Feature Engineering**
   - Retirer `duration` pour la production
   - Créer des interactions (ex: age × job)
   - Agrégations macro-économiques

3. **Autres Modèles**
   - XGBoost (performances souvent supérieures)
   - LightGBM (plus rapide)
   - Régression Logistique (baseline interprétable)

4. **Optimisation Hyperparamètres**
   - GridSearchCV / RandomizedSearchCV
   - Bayesian Optimization

### Déploiement

```python
# Sauvegarder le modèle
import joblib
joblib.dump(model, 'bank_marketing_model.pkl')

# Charger et utiliser
model_loaded = joblib.load('bank_marketing_model.pkl')
predictions = model_loaded.predict(new_customers)
```

---

