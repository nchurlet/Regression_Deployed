Regression_Deployed
![image](https://user-images.githubusercontent.com/7347376/115956415-d4923e80-a4fc-11eb-9bb9-96fc76afab30.png)
Utilisation de plusieurs modèles pour prédire le prix moyen d'un logement d'un quartier de Californie en fonction de plusieurs critères de ce quartier.
Plusieurs modèles ont été exporté sur un notebook : 
- Régression linéaire
- Arbre de décision
- Random Forest (Trop lourd pour être envoyé sur git > 100Mo)
- Gradient Boost
- Réseau de neurones

Le déploiement implique des précautions supplémentaires à prendre par rapport avec un travail local et un déploiement local.  
Les principales précautions à prendre sont liées au fichier requirements.txt.  
En effet, il faut faire attention à la version de l'environnement scikit_learn notamment à cause de l'export des objets python.  
1 - Par exemple, le `scaler` exporté en version `scikit_learn==0.23.1` lèvera une erreur si la version de l'environnement est en `0.24.1` car il ne possèdera pas l'attribut `clip`.  
2 - D'autre part, certains objets qui fonctionnent avec un import strict en local ont besoin d'un import global en deploiement.  
Par exemple pour le réseau de neurone et la fonction `load_model`. 
```py
from keras.models import load_model # <-- ceci ne fonctionnera pas
import tensorflow as tf # <-- Il faut importer tout le package
```
On devra donc charger le model d'une façon différente :
```py
# load the model
model_ann_0 = load_model('./modelann.h5') # old way
model_ann_0 = tf.keras.models.load_model('./modelann.h5') # new way
```
Dans le cas contraire, une erreur d'absence de l'attribut populate_dict sera levée.

3 - Tensorflow doit être importé dans sa dernière version.  
4 - Il ne faut pas laisser d'espaces dans le fichier requirements.txt
