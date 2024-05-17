from statsmodels.stats.diagnostic import linear_rainbow, linear_reset, het_breuschpagan, het_white
from scipy.stats import shapiro, ttest_1samp
from statsmodels.stats.stattools import durbin_watson

def text_vif():
    explanation = """
    Le VIF (variance inflation factor / facteur d'inflation de la variance) est 
    une mesure de l'importance de la colinéarité entre les variables 
    explicatives dans un modèle de régression linéaire. 
    Il indique à quel point la variance d'un coefficient de régression est 
    augmentée en raison de la corrélation entre les variables indépendantes. 
    Un VIF supérieur à 10 est souvent considéré comme indiquant une colinéarité 
    problématique (on ne considère pas le VIF de l'intercept).
    """.replace('\n    ', '\n')
    return explanation.strip()


def text_levier():
    explanation = """
    Un point levier est un point dont la coordonnée sur l’axe X est 
    significativement différente de celles des autres points. La notion de 
    point levier renvoie à la distance d’un point du centre de gravité du 
    nuage de point et par conséquent est distincte de la notion de valeur 
    aberrante. En fait, un point levier est atypique au niveau des variables 
    explicatives et l’on doit se poser la question de la considération d’un tel 
    point : erreur de mesure, inhomogénéité de la population, ...
    """.replace('\n    ', '\n')
    return explanation.strip()
    
def text_influence():
    explanation = """
    Un point influent est un point qui exerce une influence significative sur 
    l’équation de la droite de régression. On entend par cela que l’équation de 
    la droite de régression change de façon importante lorsque l’on supprime 
    ce point. Un point influent se caractérise par un levier important et 
    un résidu atypique (significativement plus grand en valeur absolue).

    Une distance de Cook / Mesure DFFITS importante peut être le résultat soit 
    d’un résidu standardisé grand (point aberrant), soit d’un levier 
    important, soit des deux.
    """.replace('\n    ', '\n')
    return explanation.strip()


def text_test_linearity(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        rainbow_test = linear_rainbow(model)
        reset_test = linear_reset(model)

        explanation = """
        Le test Rainbow est un test de linéarité utilisé pour vérifier si la relation 
        entre les régresseurs et la variable dépendante dans le modèle est linéaire.
        L'hypothèse nulle est que l'ajustement du modèle en utilisant l'échantillon 
        complet est le même que celui en utilisant un sous-ensemble central. 
        L'hypothèse alternative est que les ajustements sont différents.
        Ce test suppose que les résidus sont homoscédastiques et peut rejeter une 
        spécification linéaire correcte si les résidus sont hétéroscédastiques. 
        Si la p-valeur du test Rainbow est inférieure à un certain seuil 
        (généralement 0.05), cela suggère que la relation n'est pas linéaire.
        
        Le test d'erreur de spécification d'équation de régression de Ramsey (RESET) 
        est un test de spécification générale pour le modèle de régression linéaire. 
        Plus précisément, il teste si les combinaisons non linéaires des variables 
        explicatives aident à expliquer la variable réponse 
        (H0 : coefficients associés nuls). 
        Si la p-valeur du test Reset est inférieure à un certain seuil 
        (généralement 0.05), cela suggère que la régression pourrait ne 
        pas être correctement spécifiée.
        """.replace('\n        ', '\n')
        
        rainbow_result = f"""
        Résultat du test Rainbow:
        Statistique du test: {rainbow_test[0]}
        P-valeur: {rainbow_test[1]}
        """

        reset_result = f"""
        Résultat du test Reset:
        Statistique du test: {reset_test.statistic}
        P-valeur: {reset_test.pvalue}
        """

        return explanation.strip() + "\n\n" + rainbow_result.strip() + "\n\n" + reset_result.strip()
    
    
def text_test_normal(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        shapiro_test = shapiro(model.resid)
        t_statistic, p_value = ttest_1samp(model.resid, 0)

        explanation = """
        Le test de Shapiro-Wilk teste l'hypothèse nulle selon laquelle l'échantillon 
        de données suit une distribution normale. 
        Si la p-valeur du test de Shapiro-Wilk est inférieure à un certain seuil 
        (généralement 0.05), cela suggère que les données ne suivent pas une 
        distribution normale.

        Le test de Student teste l'hypothèse nulle selon laquelle la moyenne des 
        résidus est égal à zéro. 
        Si la p-valeur du test de Student est inférieure à un certain seuil 
        (généralement 0.05), cela suggère que la moyenne des résidus est 
        significativement différente de zéro.
        """.replace('\n        ', '\n')

        shapiro_result = f"""
        Résultat du test de Shapiro-Wilk:
        Statistique du test: {shapiro_test[0]}
        P-valeur: {shapiro_test[1]}
        """

        student_result = f"""
        Résultat du test de Student:
        Statistique du test: {t_statistic}
        P-valeur: {p_value}
        """

        return explanation.strip() + "\n\n" + shapiro_result.strip() + "\n\n" + student_result.strip()


def text_test_homo(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        breusch_pagan_test = het_breuschpagan(model.resid, model.model.exog)
        white_test = het_white(model.resid, model.model.exog)

        explanation = """
        Le test de Breusch-Pagan est utilisé pour vérifier si l'hypothèse 
        d'homoscédasticité (variance constante des erreurs) est violée dans un 
        modèle de régression linéaire. 
        Si la p-valeur du test de Breusch-Pagan est inférieure à un certain 
        seuil (généralement 0.05), on rejette l'hypothèse nulle d'homoscédasticité au
        profit d'une possible hétéroscédasticité.

        Le test de White est utilisé pour vérifier l'homoscédasticité
        des résidus dans un modèle de régression. Le test de White peut donc être un 
        test d'hétéroscédasticité (si aucun terme croisé n'est introduit dans la 
        procédure) ou de spécification, ou les deux à la fois (si les termes croisés 
        sont introduits dans la procédure).
        Si la p-valeur du test de White est inférieure à un certain seuil 
        (généralement 0.05), on rejette l'hypothèse nulle d'homoscédasticité.
        """.replace('\n        ', '\n')

        breusch_pagan_result = f"""
        Résultat du test de Breusch-Pagan:
        Statistique du test: {breusch_pagan_test[2]}
        P-valeur: {breusch_pagan_test[3]}
        """

        white_result = f"""
        Résultat du test de White:
        Statistique du test: {white_test[2]}
        P-valeur: {white_test[3]}
        """
        
        return explanation.strip() + "\n\n" + breusch_pagan_result.strip() + "\n\n" + white_result.strip()

def text_test_auto(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        dw_test_stat = durbin_watson(model.resid)

        explanation = """
        Le test de Durbin-Watson est utilisé pour vérifier l'autocorrélation d'ordre 
        1 des résidus dans un modèle de régression. La statistique de test est 
        approximativement égale à 2*(1-r), où r est l'autocorrélation de 
        l'échantillon des résidus. Ainsi, pour r = 0, ce qui indique aucune 
        autocorrélation, la statistique de test est égale à 2. 
        Plus la statistique est proche de 0, plus il y a de preuves de 
        autocorrélation positive. Plus la statistique est proche de 4, plus 
        il y a de preuves de autocorrélation négative.

        Le test de Ljung-Box teste l'auto-corrélation d'ordre supérieur à 1. 
        Il s'agit d'un test asymptotique qui n'a donc qu'une puissance très faible 
        dans le cadre de petits échantillons. L'hypothèse nulle (H0) stipule qu'il 
        n'y a pas auto-corrélation des erreurs d'ordre 1 à k (fixé à 10). 
        """.replace('\n        ', '\n')

        dw_result = f"""
        Résultat du test de Durbin-Watson:
        Statistique du test: {dw_test_stat}
        """

        lb_result = "Résultat du test de Ljung-Box:"
        
        return explanation.strip() + "\n\n" + dw_result.strip() + "\n\n" + lb_result