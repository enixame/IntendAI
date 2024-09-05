import pytest
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
from src.intendai.pipeline.intent_prediction_pipeline import IntentPredictionPipeline

# Ajout de données d'entraînement pour l'intention "greetings"
greetings_data = [
    "Salut à tous !", "Bonjour tout le monde !", "Hey !", "Bonsoir à vous tous !", 
    "Bonjour", "Hey", "Salut", "Salutation", "yo", "Bonsoir", "Bon matin",
    "Coucou, comment ça va ?", "Bonjour à tous, bon matin !", "Hello !", "Salut, ravi de te voir !",
    "Salut, comment vas-tu aujourd'hui ?", "Bonjour à tous, j'espère que vous allez bien !", 
    "Hey, ça faisait longtemps !", "Salut à tous, comment ça va ?", "Hello, comment ça se passe ?", 
    "Bonsoir tout le monde, comment ça va ?", "Salut, c'est sympa de te voir ici !", "Bonjour, tout le monde se porte bien ?", 
    "Hello, ravi de te voir ici !", "Salut, heureux de te retrouver !", "Hey, comment ça roule ?", "Bonjour tout le monde, une belle journée, non ?", 
    "Salut, comment ça se passe pour toi ?", "Hello, prêt pour une nouvelle journée ?", "Bonsoir à tous, comment ça va ?", 
    "Salut, content de te voir !", "Bonjour à toi, comment te sens-tu ?", "Hey, quoi de neuf ?", "Salut, comment va la famille ?", 
    "Bonjour, belle journée à tous !", "Hey, ça fait plaisir de te voir !", "Salut, comment va le moral ?", 
    "Bonjour, comment va ta journée ?", "Hello tout le monde, comment allez-vous ?", "Salut, comment ça bouge ?", 
    "Salut les amis, comment ça va ?", "Bonjour à tous, ça fait plaisir de vous voir !", "Hey, c'est une belle journée pour discuter, non ?", 
    "Salut, comment ça se passe aujourd'hui ?", "Bonjour à toi, comment s'est passée ta journée ?", "Salut, content de te revoir ici !", 
    "Hello, comment se déroule ta semaine ?", "Salut à tous, ravi de vous voir !", "Bonjour tout le monde, comment ça roule ?", 
    "Hey, comment se passe votre journée ?", "Salut, tout le monde va bien ?", "Bonjour, une belle matinée pour toi ?", 
    "Salut, comment va ton énergie aujourd'hui ?", "Hello, comment se passe ta matinée ?", "Salut, c'est un plaisir de te revoir !",
    "Bonjour, ravi de te voir ici !", "Salut tout le monde, comment allez-vous ?", "Hey, heureux de vous retrouver ici !", 
    "Bonjour à tous, comment ça va ce matin ?", "Hello, une belle journée n'est-ce pas ?",
    "Salut, comment ça va aujourd'hui ?", "Bonjour tout le monde, j'espère que vous passez une bonne journée !", 
    "Hey, quoi de neuf ?", "Salut à toi, ça faisait longtemps !", "Hello, comment vas-tu aujourd'hui ?", 
    "Bonsoir tout le monde, content de vous voir ici !", "Salut les amis, comment ça se passe ?", 
    "Bonjour à tous, prêt pour une nouvelle journée ?", "Hey, ravi de te voir ici !", "Salut, comment va la famille ?", 
    "Hello tout le monde, comment se passe votre semaine ?", "Salut, heureux de vous revoir !", 
    "Bonjour, c'est un plaisir de te revoir !", "Hey, comment va la vie ?", "Salut, tout se passe bien ?", 
    "Hello, j'espère que tu passes une bonne journée !", "Salut à tous, content de vous retrouver !", 
    "Bonsoir, comment allez-vous ce soir ?", "Salut, une belle journée pour se rencontrer, n'est-ce pas ?", 
    "Hello, prêt pour une autre journée productive ?", "Salut, comment va votre moral aujourd'hui ?", 
    "Bonjour, comment se déroule votre matinée ?", "Salut à tous, heureux de vous voir ici !", 
    "Hey, qu'est-ce que tu deviens ?", "Salut, comment s'est passée ta journée ?", 
    "Hello tout le monde, prêt pour un bon moment ?", "Bonjour à tous, content de vous voir !", 
    "Salut, c'est génial de vous voir tous ici !", "Hey, comment s'est passée votre semaine ?", 
    "Salut, comment ça va de votre côté ?", "Bonjour à tous, quelle belle journée !", 
    "Hello, c'est toujours agréable de vous voir !", "Salut, prêt pour une nouvelle aventure ?", 
    "Bonsoir, comment s'est passée votre journée ?", "Salut, c'est agréable de te revoir ici !", 
    "Hello, comment vas-tu en ce moment ?", "Salut tout le monde, comment va la forme ?", 
    "Bonjour, content de te voir ici ce matin !", "Hey, ça faisait un bail !", "Salut, comment ça roule ?", 
    "Hello, heureux de vous retrouver ici !", "Salut, qu'est-ce que tu racontes de beau ?", 
    "Bonjour tout le monde, comment ça va par ici ?", "Salut à tous, quoi de neuf aujourd'hui ?", 
    "Hey, comment se passe votre journée ?", "Salut, tout va bien pour toi ?", 
    "Bonjour, c'est toujours un plaisir de te voir !", "Salut, comment vas-tu après tout ce temps ?", 
    "Hello, prêt pour une journée excitante ?", "Salut à vous tous, ravi de vous revoir !", 
    "Salut, comment s'est passé ton week-end ?", "Bonjour tout le monde, heureux de vous revoir !", 
    "Hey, comment ça va de ton côté ?", "Salut à tous, c'est génial de vous voir !", 
    "Hello, comment va ta famille ?", "Bonjour, ravi de vous retrouver ici !",  "Salut, comment se passe ta matinée ?", "Bonjour à tous, une belle journée pour être ensemble !", 
    "Hey, c'est génial de te voir !", "Bonsoir à vous, comment ça va ce soir ?", "Hello, comment allez-vous en cette belle journée ?", 
    "Salut tout le monde, ça fait plaisir de vous voir ici !", "Bonjour, comment se passe votre journée ?", 
    "Hey, comment se passe votre matinée ?", "Salut à tous, vous avez passé une bonne journée ?", 
    "Bonjour tout le monde, prêts pour une nouvelle aventure ?", "Hello à tous, comment ça va aujourd'hui ?", 
    "Salut, comment s'est passée ta semaine ?", "Bonsoir, comment allez-vous ce soir ?", 
    "Salut à tous, comment se déroule votre journée ?", "Bonjour, comment s'est passé votre week-end ?", 
    "Hey, heureux de vous voir ici ce matin !", "Hello, comment ça va ce matin ?", 
    "Salut à vous tous, j'espère que vous passez une bonne journée !", "Bonjour tout le monde, prêts pour une nouvelle semaine ?", 
    "Salut, comment ça va de votre côté aujourd'hui ?"
]
greetings_intentions = ['greetings'] * len(greetings_data)  # Label pour chaque phrase d'accueil

# Ajout de données d'entraînement pour l'intention "health_status"
health_status_data = [
    "Comment te sens-tu aujourd'hui ?", "Tu te sens bien ?", "Tout va bien de ton côté ?", 
    "Es-tu en forme ce matin ?", "Comment est ton moral aujourd'hui ?", "Tu es fatigué ?",
    "As-tu bien dormi hier soir ?", "Es-tu satisfait de ta journée ?", "Comment te sens-tu aujourd'hui ?", "Tu es en forme ce matin ?", "As-tu bien dormi la nuit dernière ?", 
    "Tout va bien avec toi ?", "Comment va ton moral aujourd'hui ?", "Tu te sens comment après hier ?", 
    "Es-tu de bonne humeur ce matin ?", "Tu es fatigué aujourd'hui ?", "Comment va ta santé ces derniers temps ?", 
    "Tu sembles un peu épuisé, tout va bien ?", "As-tu récupéré de la semaine dernière ?", "Comment ça va mentalement aujourd'hui ?", 
    "Ton moral est bon aujourd'hui ?", "Tu te sens mieux aujourd'hui ?", "Comment va ton énergie ce matin ?", 
    "Tu as bien dormi ?", "Es-tu en pleine forme ?", "Comment est ton humeur aujourd'hui ?", 
    "Tu te sens comment ces jours-ci ?", "As-tu eu une bonne nuit de sommeil ?", "Tu vas bien aujourd'hui ?", 
    "Comment est ta forme ce matin ?", "Comment te portes-tu aujourd'hui ?", "As-tu des soucis de santé actuellement ?", 
    "Es-tu content aujourd'hui ?", "Comment est ton niveau d'énergie ?", "Tu sembles stressé, ça va ?", 
    "Comment te sens-tu physiquement ?", "Est-ce que tu te sens en forme ?", "Comment va ta santé mentale ?", 
    "As-tu bien récupéré ?", "Comment est ta vitalité aujourd'hui ?", "Es-tu de bonne humeur aujourd'hui ?", 
    "Comment va ton moral après ce week-end ?", "As-tu pris soin de toi récemment ?", "Comment va ta santé émotionnelle ?", 
    "Tu te sens comment ce matin ?", "Comment est ton esprit aujourd'hui ?", "As-tu une bonne énergie aujourd'hui ?", 
    "Comment est ta santé générale ?", "As-tu bien commencé ta journée ?", "Comment te sens-tu après ce long voyage ?",
     "Comment te sens-tu aujourd'hui ?", "Ça va bien, toi ?", "Es-tu en forme ce matin ?", 
    "Comment va ta santé récemment ?", "Tu sembles fatigué, tout va bien ?", "Est-ce que tu te sens bien aujourd'hui ?", 
    "Comment est ton moral aujourd'hui ?", "As-tu bien dormi la nuit dernière ?", 
    "Tu es de bonne humeur ce matin ?", "Comment te sens-tu après ce long voyage ?", 
    "Comment va ta forme en ce moment ?", "Es-tu satisfait de ta journée jusqu'à présent ?", 
    "Comment ça se passe pour toi ces jours-ci ?", "Tu sembles un peu stressé, ça va ?", 
    "Comment est ton niveau d'énergie aujourd'hui ?", "Tu vas bien ?", "Comment est ta santé mentale aujourd'hui ?", 
    "Tu te sens comment physiquement ?", "Es-tu en bonne santé ?", "Comment est ta forme physique aujourd'hui ?", 
    "Tu es heureux aujourd'hui ?", "Comment te sens-tu ce matin ?", "As-tu eu une bonne nuit de sommeil ?", 
    "Comment est ton moral aujourd'hui ?", "Est-ce que tu te sens fatigué ?", "Tu te sens mieux qu'hier ?", 
    "Comment ça va mentalement ?", "Ton humeur est bonne aujourd'hui ?", "Es-tu prêt pour la journée ?", 
    "Tu as bien récupéré ?", "Comment est ton énergie ce matin ?", "Es-tu en pleine forme ?", 
    "Tu te sens comment ces jours-ci ?", "As-tu bien commencé ta journée ?", "Comment est ton esprit aujourd'hui ?", 
    "Est-ce que tu te sens en forme ?", "Comment va ta santé générale ?", "As-tu bien dormi cette nuit ?", 
    "Tu es en forme ce matin ?", "Comment est ta vitalité aujourd'hui ?", "Comment va ton moral après ce week-end ?", 
    "Tu te sens mieux aujourd'hui ?", "Es-tu prêt pour une journée productive ?", "Comment te sens-tu émotionnellement ?", 
    "Tu sembles en pleine forme aujourd'hui ?", "Comment ça va aujourd'hui ?", "Tu as l'air en forme ce matin ?", 
    "Comment va ton énergie récemment ?", "Es-tu de bonne humeur ?", "Comment est ton niveau d'énergie ?", 
    "Tu as l'air plus détendu aujourd'hui, ça va ?", "Comment va ta forme physique ?", "Comment va ton moral ?", 
    "Tu te sens comment aujourd'hui ?", "Tu es de bonne humeur ce matin ?", "Comment va ta santé mentale ?", 
    "As-tu bien récupéré de la semaine dernière ?", "Tu sembles en bonne santé, comment te sens-tu ?", 
    "Est-ce que tout va bien pour toi ?", "Comment va ton bien-être aujourd'hui ?", "Tu te sens bien ce matin ?", 
    "Comment est ton humeur aujourd'hui ?", "Es-tu satisfait de ta forme ?", "Comment est ta santé aujourd'hui ?", 
    "Tu te sens comment en ce moment ?", "As-tu bien dormi ?", "Comment ça va physiquement ?", 
    "Tu as l'air en forme, comment te sens-tu ?", "Comment va ton moral cette semaine ?", "Est-ce que tu vas bien ?", "Comment te sens-tu après cette longue journée ?", "As-tu bien récupéré ce week-end ?", 
    "Tu sembles en forme aujourd'hui, ça va bien ?", "Comment est ton humeur après cette semaine ?", 
    "Es-tu reposé aujourd'hui ?", "As-tu bien dormi cette nuit ?", "Comment est ta forme physique aujourd'hui ?", 
    "Est-ce que tu te sens mieux maintenant ?", "Comment te portes-tu après la réunion ?", 
    "Tu sembles un peu fatigué, tout va bien ?", "As-tu une bonne énergie aujourd'hui ?", 
    "Comment est ta santé mentale aujourd'hui ?", "Es-tu satisfait de ta forme cette semaine ?", 
    "Tu vas mieux depuis hier ?", "Comment ça va moralement ?", "Ton humeur est-elle bonne aujourd'hui ?", 
    "Tu te sens comment mentalement aujourd'hui ?", "Es-tu en pleine forme après ce week-end ?", 
    "Comment est ta vitalité aujourd'hui ?", "Es-tu heureux aujourd'hui ?", "Comment va ta santé après cette épreuve ?", 
    "Es-tu en bonne santé ces jours-ci ?", "Tu sembles stressé, comment ça va ?", "Comment est ton énergie en ce moment ?", 
    "Tu sembles en pleine forme, comment ça va ?", "Comment va ta santé en général ?"
]
health_status_intentions = ['health_status'] * len(health_status_data)  # Label pour chaque phrase de statut de santé

# Ajout de données d'entraînement pour l'intention "backseat"
backseat_data = [
   "Je pense que tu devrais utiliser cette approche à la place.", "Tu devrais essayer de résoudre ce problème d'une autre manière.", 
    "Il serait mieux de commencer par la partie la plus simple.", "Pourquoi ne pas utiliser cet outil pour améliorer le processus ?", 
    "Tu devrais envisager de changer de stratégie ici.", "Essaie de suivre mon conseil pour une fois.", 
    "Je te conseille de revoir ton plan, il pourrait être amélioré.", "Pourquoi ne pas essayer une méthode différente ?", 
    "Tu ferais mieux de ralentir, c'est plus sûr.", "Tu devrais prendre une pause maintenant, c'est important.", 
    "Il vaut mieux que tu te concentres sur une tâche à la fois.", "Je pense que tu devrais faire ça autrement.", 
    "Essaie de garder un œil sur les détails, c'est crucial.", "Tu devrais envisager une autre option.", 
    "Pourquoi ne pas consulter un expert avant de continuer ?", "Il serait plus efficace de revoir tes priorités.", 
    "Tu devrais vraiment envisager d'utiliser cet outil.", "Je te recommande de prendre le temps de bien réfléchir avant d'agir.", 
    "Essaie de réévaluer ta stratégie, elle pourrait être améliorée.", "Tu devrais changer de méthode, celle-ci ne semble pas fonctionner.", 
    "Je te suggère de demander l'avis de quelqu'un d'autre.", "Tu ferais mieux d'éviter cette approche, elle est risquée.", 
    "Pourquoi ne pas essayer une approche plus douce ?", "Tu devrais vraiment prendre du recul et réfléchir.", 
    "Il serait préférable de ne pas précipiter les choses.", "Je te recommande de faire une pause et de réévaluer.", 
    "Tu devrais peut-être revoir tes priorités.", "Pourquoi ne pas commencer par les bases ?", 
    "Essaie d'explorer différentes options.", "Tu devrais prendre le temps d'analyser la situation.", 
    "Je te conseille de réfléchir à deux fois avant d'agir.", "Il vaut mieux que tu prennes une approche plus stratégique.", 
    "Essaie de voir les choses sous un autre angle.", "Tu devrais envisager de demander de l'aide.", 
    "Tu devrais vraiment prendre à gauche ici.", "Pourquoi ne fais-tu pas ça à ma manière ?", 
    "Tu devrais essayer de le faire comme ça.", "Je pense que tu devrais ralentir un peu.", 
    "Pourquoi ne pas utiliser cet outil à la place ?", "Tu devrais utiliser une couleur différente pour ça.", 
    "Je ferais ça différemment si j'étais toi.", "Tu devrais suivre mon conseil et prendre le chemin plus long.", 
    "Essaie de changer ta stratégie, ça marcherait mieux.", "Tu devrais mettre plus de sel dans ta recette.", 
    "Je te conseille de commencer par là.", "Ne fais pas comme ça, fais plutôt comme ça.", 
    "Il vaut mieux que tu le fasses de cette manière.", "Tu devrais vraiment écouter mes conseils pour une fois.", 
    "Fais-le maintenant avant qu'il ne soit trop tard.", "Tu dois vraiment revoir ta façon de faire cela.", 
    "À ta place, je prendrais une autre approche.", "Tu ferais mieux de garder tes mains sur le volant.", 
    "Ce n'est pas la bonne façon, tu devrais le faire autrement.", "Tu devrais regarder ici plutôt que là-bas.", 
    "Pourquoi tu ne t'y prends pas comme ça ?", "Tu devrais appeler le service client pour obtenir de l'aide.", 
    "Je pense que tu devrais utiliser un autre logiciel.", "Tu devrais revoir ton plan, il y a une meilleure façon.", 
    "Tu devrais vraiment prendre une pause maintenant.", "Il serait mieux de changer ton approche.", 
    "Essaie de le faire plus lentement.", "Tu devrais consulter un expert pour ça.", 
    "Tu devrais utiliser une autre méthode.", "Fais attention, tu devrais peut-être revoir ton code.", 
    "Il serait plus efficace d'essayer une autre stratégie.", "Tu devrais laisser quelqu'un d'autre le faire.", 
    "Pourquoi ne pas utiliser une approche différente ?", "Tu devrais recommencer depuis le début.", 
    "Je te conseille de prendre un autre chemin.", "Tu devrais ajuster ton plan pour de meilleurs résultats.", 
    "Je pense que tu ferais mieux de reconsidérer cette option.", "Tu devrais vraiment demander l'avis de quelqu'un d'autre.", 
    "Pourquoi ne pas essayer de faire autrement ?", "Il vaut mieux que tu te concentres sur autre chose.", 
    "Tu devrais changer de méthode, celle-ci ne fonctionne pas.", "Je te recommande de suivre mes conseils.", 
    "Tu devrais essayer un autre angle.", "Il serait mieux de ne pas le faire de cette façon.", 
    "Tu devrais écouter ce que je te dis.", "Essaie de faire comme je te l'ai dit.", 
    "Je pense que tu devrais prendre un peu de recul.", "Tu devrais vérifier deux fois avant de continuer.", 
    "Pourquoi ne pas suivre le guide pour une fois ?", "Tu devrais changer de stratégie.", 
    "Essaie de voir ça sous un autre angle.", "Tu devrais vraiment te concentrer sur cette partie.", 
    "Il serait plus prudent de faire une pause ici.", "Tu devrais garder un œil sur les détails.", 
    "Pourquoi ne pas essayer de penser différemment ?", "Tu devrais toujours commencer par une analyse approfondie.", 
    "Tu devrais consulter un mentor pour des conseils.", "Tu devrais peut-être revoir ton approche.", 
    "Pourquoi ne pas essayer de faire ça autrement ?", "Je te suggère de changer cette partie.", 
    "Essaie plutôt de le faire comme ça.", "Tu devrais utiliser un autre angle ici.", 
    "Je pense que ce serait mieux si tu utilisais une autre méthode.", "Tu devrais prendre à droite ici.", 
    "Pourquoi tu n'accélères pas un peu ?", "Essaie de passer à une autre stratégie maintenant.", 
    "Tu devrais changer de voie.", "Tu ferais mieux de ralentir.", "Pourquoi ne pas essayer de faire ça comme ça ?", 
    "Tu devrais vraiment écouter mes conseils.", "Il serait préférable de suivre cette route.", 
    "Tu devrais envisager une autre approche.", "Pourquoi ne pas utiliser cette méthode à la place ?", 
    "Tu devrais essayer de changer de direction ici.", "Je pense que tu devrais ralentir.", 
    "Pourquoi ne fais-tu pas ça à ma manière?", "Essaie de faire comme ça, ce serait mieux.", 
    "Tu devrais vraiment écouter mes conseils pour cette fois.", "Il serait préférable de prendre cette voie.", 
    "À ta place, je ferais autrement.", "Tu devrais essayer une autre méthode.", 
    "Tu devrais vraiment utiliser une autre stratégie ici.", "Je te recommande de ralentir à ce point.", 
    "Pourquoi tu ne fais pas ce que je suggère?", "Essaie plutôt comme ça, c'est mieux.", 
    "Il serait préférable de suivre cette approche.", "Je ferais autrement à ta place.", 
    "Tu devrais essayer cette méthode.", "Prends une autre direction ici.", 
    "Tu devrais vraiment prendre une autre approche ici.", "Pourquoi ne pas faire ce que je te suggère ?", 
    "Il serait préférable de ralentir maintenant.", "Essaie de suivre mon conseil pour une fois.", 
    "Prends le chemin de droite, c'est mieux.", "Je pense que tu devrais essayer ça à la place.", 
    "Essaie une autre stratégie, celle-ci n'est pas efficace.", "Je te recommande de faire différemment.", "Tu devrais vraiment envisager une autre méthode ici.", "Pourquoi ne pas essayer de faire différemment cette fois ?", 
    "Je pense que tu ferais mieux de prendre cette direction.", "Essaie de faire comme ça pour de meilleurs résultats.", 
    "Tu devrais peut-être changer de tactique maintenant.", "Pourquoi ne pas utiliser cette approche à la place ?", 
    "Je te recommande de ralentir à ce point-ci.", "Tu ferais mieux de suivre mon conseil.", 
    "Il serait préférable de revoir cette partie.", "Tu devrais consulter un expert pour obtenir des conseils.", 
    "Pourquoi ne pas essayer cette méthode alternative ?", "Essaie d'adopter une nouvelle perspective ici.", 
    "Tu devrais revoir ton plan et ajuster en conséquence.", "Je te conseille de réévaluer tes options.", 
    "Il vaut mieux que tu prennes une autre voie.", "Tu devrais vraiment prendre en compte cette suggestion.", 
    "Essaie de prendre un autre angle sur ce problème.", "Pourquoi ne pas envisager un autre chemin ?", 
    "Tu devrais suivre cette approche pour une meilleure efficacité.", "Il serait mieux de ne pas précipiter les choses.",
    "Vas tout droit", "Vas à droite", "Vas là bas", "vais au milieu", "Vas ici", "Vas là", "Vas",
    "Tu dois aller à droite", "Tu dois faire ça", "Tu dois attendre", "Tu dois combattre", "Tu dois taper ici",
    "Tu dois fabriquer çà", "Tu dois aller à gauche", "Tu dois aller tout droit", "gauche", "droite", "en haut", "en bas"
]
backseat_intentions = ['backseat'] * len(backseat_data)  # Label pour chaque phrase de suggestion de contrôle

# Ajout de données d'entraînement pour l'intention "bad"
bad_data = [
    "C'est vraiment nul ce que tu fais", "Qu'est ce que c'est nul", "il est nul ton stream", "C'est variment nul",
    "Nul", "Je n'ai jamais vu quelque chose d'aussi nul", "C'est d'une nullité", "C'est totalement nul ce que tu fais !",
    "ton travail est vraiment nul", "c'est comlètement nul", "c'est incroyablement nul",
    "Ce que tu fais est vraiment pathétique.", "C'est absolument horrible.", "Tu n'as aucun talent pour ça.", 
    "C'est vraiment le pire travail que j'ai jamais vu.", "Franchement, tu devrais abandonner.", 
    "C'est complètement nul, tu ne devrais même pas essayer.", "Ton travail est lamentable.", 
    "C'est une honte ce que tu as fait.", "Je n'ai jamais vu quelque chose d'aussi mauvais.", 
    "Tu es vraiment mauvais dans ce domaine.", "C'est de loin le pire effort jamais réalisé.", 
    "C'est une perte de temps totale.", "Tu es incompétent, sérieusement.", "C'est risible, à vrai dire.", 
    "Tu devrais vraiment arrêter de faire ça.", "Ton projet est un désastre complet.", 
    "C'est une catastrophe absolue.", "Tu es totalement incapable.", "C'est un échec sur toute la ligne.", 
    "Tu devrais avoir honte de toi.", "C'est totalement inutile ce que tu fais.", 
    "Franchement, c'est un désastre.", "C'est ridicule et inutile.", "C'est complètement raté, tu n'y arriveras jamais.", 
    "Ton travail est un vrai fiasco.", "C'est pitoyable, arrête ça tout de suite.", 
    "Tu ne fais jamais rien de bien, vraiment.", "C'est un gâchis complet, tu n'as rien compris.", 
    "Tu n'as absolument aucune idée de ce que tu fais.", "C'est incroyablement mauvais.", 
    "Je n'ai jamais vu quelque chose d'aussi pathétique.", "C'est de la pure bêtise.", 
    "Tu devrais abandonner, tu es vraiment mauvais.", "C'est une perte d'effort complète.", 
    "C'est lamentable et inacceptable.", "C'est embarrassant à voir.", "Tu es complètement nul.", 
    "C'est une honte totale.", "Ton niveau est désespérément bas.", "C'est une horreur totale.", 
    "Tu n'as aucune chance de réussir.", "C'est juste pathétique, vraiment.",
    "C'est vraiment nul ce que tu fais.", "Tu es incompétent.", "C'est la pire chose que j'ai jamais vue.", 
    "Tu n'as vraiment aucune idée de ce que tu fais.", "C'est totalement inutile.", "Ton travail est lamentable.", 
    "C'est une honte, vraiment.", "Je n'ai jamais vu quelque chose d'aussi mal fait.", 
    "Franchement, c'est pathétique.", "C'est complètement raté.", "Tu n'es vraiment pas bon dans ce domaine.", 
    "C'est du travail de débutant.", "Tu fais vraiment tout de travers.", "C'est une vraie catastrophe.", 
    "Je ne peux pas croire que tu penses que c'est bien.", "C'est décevant et inacceptable.", 
    "Ton projet est un échec total.", "C'est du travail bâclé.", "Tu devrais avoir honte de toi.", 
    "Je n'ai jamais vu quelque chose d'aussi médiocre.", "Ton code est un désastre.", 
    "Tout ce que tu fais tourne au fiasco.", "C'est un vrai gâchis.", "Tu ne fais jamais rien de bien.", 
    "C'est de loin le pire effort que j'ai jamais vu.", "C'est risible, pour être honnête.", 
    "Tu n'es bon à rien.", "C'est une perte de temps et d'énergie.", "Je te conseille de suivre mes instructions.", 
    "Tu devrais refaire ce code.", "Pourquoi ne pas utiliser une autre méthode ?", 
    "Essaie de prendre ce chemin à la place.", "Tu devrais vraiment essayer ça différemment.", 
    "Tu dois vraiment revoir cette partie.", "Il vaudrait mieux écouter mes conseils.", 
    "Fais-le de cette manière pour de meilleurs résultats.", "Tu pourrais utiliser cette technique ici.", 
    "Il serait plus efficace de faire comme ça.", "Pourquoi ne pas changer ton approche ?", 
    "Tu devrais tester avec un autre outil.", "Essaie d'abord de faire une sauvegarde.", 
    "Tu devrais consulter quelqu'un d'autre pour ça.", "Tu devrais réviser ton plan.", 
    "Tu n'es vraiment pas à la hauteur.", "C'est lamentable ce que tu fais.", "Tu devrais abandonner, c'est terrible.", 
    "C'est la pire chose que j'ai jamais vue.", "Ton travail est absolument inutile.", 
    "C'est du grand n'importe quoi.", "Je ne pense pas que tu sois fait pour ça.", 
    "C'est vraiment décevant.", "Je ne peux pas croire que tu fasses ça de cette manière.", 
    "C'est inacceptable.", "Tu es vraiment mauvais.", "Ton travail est un désastre.", 
    "C'est un échec complet.", "Tu devrais avoir honte.", "C'est tellement mauvais que c'en est triste.", 
    "C'est le pire que j'ai jamais vu.", "Tu n'as aucun talent pour ça.", "C'est une horreur.", 
    "C'est tellement mal fait.", "C'est une blague, n'est-ce pas ?", "Tu devrais arrêter, c'est embarrassant.", 
    "C'est une perte de temps totale.", "Tu es complètement incompétent.", "C'est pathétique, vraiment.",  "C'est absolument terrible, tu devrais arrêter.", "C'est le pire travail que j'ai vu depuis longtemps.", 
    "Tu es vraiment mauvais, il n'y a pas d'autre mot.", "C'est une perte de temps complète.", 
    "Tu devrais abandonner, c'est pathétique.", "C'est la pire tentative que j'ai jamais vue.", 
    "Tu n'as vraiment aucune compétence pour ça.", "Ton travail est un échec total.", 
    "C'est embarrassant, tu devrais avoir honte.", "C'est du travail de mauvaise qualité.", 
    "C'est tellement mauvais que c'en est risible.", "Tu es totalement incompétent.", 
    "C'est une catastrophe, tu ne devrais pas continuer.", "C'est lamentable, arrête immédiatement.", 
    "Tu ne fais que gâcher du temps et des ressources.", "C'est une blague, tu n'es pas sérieux.", 
    "C'est vraiment pathétique, tu devrais réévaluer tes compétences.", "C'est le pire que j'ai jamais vu.", 
    "C'est une honte, tu es vraiment mauvais.", "Ton travail est incroyablement médiocre."
]
bad_intentions = ['bad'] * len(bad_data)  # Label pour chaque phrase négative

def balance_data(data, labels):
    """
    Fonction pour équilibrer les classes en utilisant le rééchantillonnage.

    Args:
        data (list): Liste des phrases d'entrée.
        labels (list): Liste des labels correspondants.

    Returns:
        tuple: Deux listes contenant les phrases et labels rééchantillonnés et équilibrés.
    """
    # Convertir les données et les labels en DataFrame pour faciliter le rééchantillonnage
    df = pd.DataFrame({'text': data, 'label': labels})

    # Déterminer le nombre d'échantillons maximum pour équilibrer toutes les classes
    max_samples = df['label'].value_counts().max()

    # Rééchantillonner les données pour équilibrer toutes les classes
    df_balanced = pd.concat([
        resample(df[df['label'] == label], 
                 replace=True,           # Échantillonnage avec remplacement
                 n_samples=max_samples,  # Nombre d'échantillons à atteindre pour chaque classe
                 random_state=42)        # Pour garantir la reproductibilité
        for label in df['label'].unique()
    ])
    
    # Retourner les données et labels rééchantillonnés sous forme de listes
    return df_balanced['text'].tolist(), df_balanced['label'].tolist()

def evaluate_model(pipeline, data, labels, label_encoder):
    """
    Évalue le modèle sur un ensemble de données de test.

    Args:
        pipeline (IntentPredictionPipeline): Le pipeline de prédiction d'intention.
        data (list): Liste de phrases pour le test.
        labels (list): Liste des labels encodés pour le test.
        label_encoder (LabelEncoder): L'encodeur utilisé pour les labels.

    Prints:
        Matrice de confusion et rapport de classification.
    """
    # Faire des prédictions pour chaque phrase dans les données de test
    predictions = [pipeline.predict_intent(text) for text in data]
    predictions_encoded = label_encoder.transform(predictions)

    # Calculer la matrice de confusion
    cm = confusion_matrix(labels, predictions_encoded)
    print("Matrice de confusion :")
    print(cm)

    # Ajuster les target_names pour inclure toutes les classes, même celles non prédites
    target_names = label_encoder.classes_
    
    # Ajout de labels=unique_labels à classification_report
    print("\nRapport de classification :")
    print(classification_report(labels, predictions_encoded, target_names=target_names, labels=np.unique(labels)))

@pytest.fixture(scope="module")
def pipeline():
    pipeline = IntentPredictionPipeline()
    
    # Utiliser les données équilibrées pour l'entraînement
    # data_balanced, labels_balanced = balance_data(
    #     greetings_data + health_status_data + backseat_data + bad_data,
    #     greetings_intentions + health_status_intentions + backseat_intentions + bad_intentions
    # )
    data = greetings_data + health_status_data + backseat_data + bad_data
    labels = greetings_intentions + health_status_intentions + backseat_intentions + bad_intentions

    # pipeline.add_training_data(data_balanced, labels_balanced)
    pipeline.add_training_data(data, labels)

    # Entraîner le modèle
    pipeline.train_model()

    return pipeline

def test_phrases(pipeline):
    test_phrases = [
        "Bonjour", 
        "Salut ma gueule !", 
        "Wesh ma gueule !", 
        "Comment ça va ?", 
        "La forme ?", 
        "Tu devrais ralentir ici.", 
        "Tu dois aller par là.", 
        "Vas à gauche.",
        "C'est vraiment nul ce que tu fais.", 
        "t'es nul."]
    for phrase in test_phrases:
        print(f"Phrase: '{phrase}' - Prédiction: '{pipeline.predict_intent(phrase)}'")

def test_greetings(pipeline):
    """
    Teste la prédiction pour l'intention "greetings".

    Args:
        pipeline (IntentPredictionPipeline): Le pipeline de prédiction d'intention entraîné.
    """
    phrases = ["Bonjour", "Salut", "Hey"]
    for phrase in phrases:
        assert pipeline.predict_intent(phrase) == "greetings"

def test_health_status(pipeline):
    """
    Teste la prédiction pour l'intention "health_status".

    Args:
        pipeline (IntentPredictionPipeline): Le pipeline de prédiction d'intention entraîné.
    """
    phrases = ["Comment ça va ?", "Ça va bien ?", "La forme ?"]
    for phrase in phrases:
        assert pipeline.predict_intent(phrase) == "health_status"

def test_backseat(pipeline):
    """
    Teste la prédiction pour l'intention "backseat".

    Args:
        pipeline (IntentPredictionPipeline): Le pipeline de prédiction d'intention entraîné.
    """
    phrases = ["Tu devrais vraiment prendre à gauche ici.", "Pourquoi ne fais-tu pas ça à ma manière ?"]
    for phrase in phrases:
        assert pipeline.predict_intent(phrase) == "backseat"

def test_bad(pipeline):
    """
    Teste la prédiction pour l'intention "bad".

    Args:
        pipeline (IntentPredictionPipeline): Le pipeline de prédiction d'intention entraîné.
    """
    phrases = ["C'est vraiment nul ce que tu fais.", "Tu es incompétent."]
    for phrase in phrases:
        assert pipeline.predict_intent(phrase) == "bad"

def test_evaluate_model(pipeline):
    """
    Évalue le modèle avec des données de test équilibrées.

    Args:
        pipeline (IntentPredictionPipeline): Le pipeline de prédiction d'intention entraîné.
    """
    # Préparer les données d'entraînement et de test avec des données équilibrées
    data = greetings_data + health_status_data + backseat_data + bad_data
    labels = greetings_intentions + health_status_intentions + backseat_intentions + bad_intentions
    
    # Rééchantillonner les données pour équilibrer les classes
    # data_balanced, labels_balanced = balance_data(data, labels)

    # Encoder les labels pour l'évaluation
    label_encoder = LabelEncoder()
    # labels_encoded = label_encoder.fit_transform(labels_balanced)
    labels_encoded = label_encoder.fit_transform(labels)

    # Utiliser une fraction des données pour l'évaluation, par exemple les 20% derniers
    # split_index = int(0.8 * len(data_balanced))
    # X_test, y_test = data_balanced[split_index:], labels_encoded[split_index:]
    split_index = int(0.8 * len(data))
    X_test, y_test = data[split_index:], labels_encoded[split_index:]

    # Évaluer le modèle avec la fonction d'évaluation
    evaluate_model(pipeline, X_test, y_test, label_encoder)
