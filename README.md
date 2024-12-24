# AI_Loto


Première étape, dans un terminal, faire:
- pip install -r requirement.txt

Ensuite, on vas crée une images par boules à partir de l'image imageBoules.png.
- Faire dans le terminal: python .\create90ImagesBoule.py

Ce script vas prendre l'image avec les 90 boules et génerer une images pour chaques boules dans un dossier qui vas s'appeler boules

Ensuite, on vas générer pour chaqu'une des ces images nouvellement générer des images en plus ,un peux plus zoomer/décaler afin d'obtenir plus de données pour entrainer l'ia
- Faire dans le terminal: python .\generateMoreImage.py

Maintenant qu'on as toute nos données on peut crée le model d'ia, l'entrainer et le tester. Toutles fichiers pour ce faire sont dans le dossier src.

Faire dans le terminal: python .\src\createModel.py
Cela vas crée le model et le stocker dans le fichier "image_classification_model.h5"

Ensuite on vas pouvoir directement tester le modèle.
Faire dans le terminal: python .\src\testOriginModel.py

Ce script vas prédire la classe d'une images, pour choisir l'images à deviner il faut changer la ligne 9 du fichier pour choisir une autre image puis re executer le fichier.



Commentaires Annexe:
le fichier installPip.py devrait être inutile, c'était pour moi pour installer pip au début mais si tu installe python sur internet et que dans l'installeur tu coche bien la case installer pip alors tu ne devrais pas avoir besoin du script.


Lien utiles:
Pour être honnête j'ai principalement utilisé chatgpt pour faire ça mais j'ai un autre lien utile ou il y a un bon exemple de comment crée et entrainer une ia pour classifier des images:

lien:  https://www.tensorflow.org/tutorials/images/classification?hl=fr