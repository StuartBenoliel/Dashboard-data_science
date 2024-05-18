import faicons as fa
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from shared import app_dir, data
from shinywidgets import output_widget, render_plotly
from shiny import App, reactive, render, ui

import re
from plots import *
from texts import *
from data_frame import *


column_names = data.columns.tolist()
column_types = data.dtypes
quantitative_vars = [col for col, dtype in zip(column_names, column_types) if dtype != "object"]
qualitative_vars = [col for col, dtype in zip(column_names, column_types) if dtype == "object"]
quant = {col: col for col in quantitative_vars}
quali = {col: col for col in qualitative_vars}

ICONS = {"ellipsis": fa.icon_svg("ellipsis")}

init_var_x = column_names[0]
init_var_y = column_names[1]

init_value_var_x = (data[init_var_x].min(), data[init_var_x].max())
init_value_var_y = (data[init_var_y].min(), data[init_var_y].max())

def extract_elements(expression):
    # Utilise une expression régulière pour capturer les éléments entre les opérateurs *
    pattern = re.compile(r'(\w+)\s*\*{1,2}\s*(\w+)')
    matches = pattern.findall(expression)
    
    # Aplatir la liste de tuples en une liste de chaînes de caractères
    elements = [item for sublist in matches for item in sublist]
    if len(elements) == 0:
        return [expression]
    return elements

def extract_column_name(expression, data):
    # Trouver tous les mots entre parenthèses
    variables = re.findall(r'\((.*?)\)', expression)
    if len(variables) == 0:
        variables = expression
    else:
        variables = variables[0]
    variables = extract_elements(variables)
    for variable in variables:
        if variable in data.columns:
            return variable
    return expression

app_ui = ui.page_navbar(
    ui.nav_panel("Vue d'ensemble",
        ui.page_sidebar(
            ui.sidebar(
                ui.input_file("file", "Upload a csv file", accept=[".csv", ".txt"]),
                ui.input_selectize("var_x", "Variable X:",
                    {"Variable quantitative :": quant, "Variable qualitative :": quali}, selected=init_var_x
                ),
                ui.output_ui("slider_x"),
                ui.input_selectize("var_y", "Variable Y:",
                    {"Variable quantitative :": quant, "Variable qualitative :": quali}, selected=init_var_y
                ),
                ui.output_ui("slider_y"),
                ui.input_selectize("var_z", "Variable Catégorielle:",
                    {"1" : {"" : None}, "Variable qualitative :": quali}
                ),
                ui.panel_conditional("input.var_z != '' ", 
                    ui.input_checkbox_group("var_z_modalite", "Modalités:", [], selected=[], inline=True)
                ),
                ui.input_action_button("reset", "Reset"),
            ),
            ui.layout_columns(
                ui.navset_card_underline(
                    ui.nav_panel("Table", ui.output_data_frame("table")),
                    ui.nav_panel("Résumé", ui.output_data_frame("summary_table")),
                    title="Données"
                ),
                ui.card(
                    ui.card_header("Corrélations entre variables quantitatives"), 
                    output_widget("corrplot"), full_screen=True
                ),
                ui.card(
                    ui.card_header("Histogramme Variable X",
                        ui.popover(ICONS["ellipsis"],
                            ui.input_radio_buttons("histogram_color", None,
                                ["Aucune"]+ qualitative_vars, inline=True,
                            ), title="Couleur:"
                        ), class_="d-flex justify-content-between align-items-center"
                    ),
                    output_widget("histogram"),
                    ui.output_ui("bins"), full_screen=True
                ),
                ui.card(
                    ui.card_header("Scatterplot Variable X vs Variable Y",
                        ui.popover(ICONS["ellipsis"],
                            ui.input_radio_buttons("scatter_color", None,
                                ["Aucune"]+ qualitative_vars, inline=True,
                            ), title="Couleur:",
                        ), class_="d-flex justify-content-between align-items-center",
                    ),
                    output_widget("scatterplot"), 
                    ui.input_checkbox("smoother", "Ajouter LOWESS"), full_screen=True,
                ),
                ui.card(
                    ui.card_header("Violinplot Variable X Vs Variable Catégorielle",
                        ui.popover(ICONS["ellipsis"],
                            ui.input_radio_buttons("violin_color", None,
                                ["Aucune"]+ qualitative_vars, inline=True,
                            ), title="Couleur:",
                        ), class_="d-flex justify-content-between align-items-center",
                    ),
                    output_widget("violinplot"), full_screen=True,
                ),
                col_widths=[6, 6, 6, 6, 6],
            )
        )
    ),
    ui.nav_panel("Régression", 
        ui.layout_columns(
            ui.input_selectize("type_reg", "Type de régression:",
                ["Régression linéaire", "Modèles linéaires généralisés (GLM)"], selected="Régression linéaire"
            ),
            ui.tooltip(
                ui.input_text("equation", "Entrer l'équation:", " ~ "), 
                "Exemple : Y  ~ np.log(X_1) + standardize(X_2) + I(X_3**2)", placement="right"
            ),
            ui.panel_conditional("input.type_reg === 'Modèles linéaires généralisés (GLM)' ", 
                ui.layout_columns(
                    ui.input_selectize("type_loi", "Loi du model:",
                        {"1" : {"Variable binaire" : {"Bernoulli" : "Bernoulli"}}, 
                         "2" : {"Variable catégorielle" : {"Multinomiale" : "Multinomiale"}}, 
                         "3" : {"Variable de comptage" : {"Poisson" : "Poisson", "Binomiale négative" : "Binomiale négative"}}
                        }, selected = "Bernoulli"
                    ),
                    ui.input_selectize("fc_liens", "Fonctions de liens:",
                        {"Logit": "Logit (régression logistique)", "Probit": "Probit", "Cloglog": "Cloglog", "Loglog": "Loglog"},
                    ),
                )
            ),
            ui.output_ui("reg_card"),
            col_widths=[3, 3, 6, 12],
        ),
    ),
    title="Dashboard Data-Science", fillable=True
)

def server(input, output, session):

    # Nav_panel : Vue d'ensemble
    upload_state = reactive.value(False)

    @render.ui
    @reactive.event(input.var_x)
    def slider_x():
        if input.var_x() in type_var()["quant"]:
            min_value_x = dataf()[input.var_x()].min()
            max_value_x = dataf()[input.var_x()].max()
            return ui.input_slider("var_x_slider", "Variable X range", min=min_value_x,
                max=max_value_x, value=(min_value_x, max_value_x))

    @render.ui
    @reactive.event(input.var_y)
    def slider_y():
        if input.var_y() in type_var()["quant"]:
            min_value_y = dataf()[input.var_y()].min()
            max_value_y = dataf()[input.var_y()].max()
            return ui.input_slider("var_y_slider", "Variable Y range", min=min_value_y,
                max=max_value_y, value=(min_value_y, max_value_y))
        
    @render.ui
    @reactive.event(input.var_x)
    def bins():
        if input.var_x() in type_var()["quant"]:
            return ui.input_slider("bins", "Nombre de bins:", min=1, max=30, value=15)

    @reactive.effect
    @reactive.event(input.file)
    def handle_file():
        upload_state.set(True)
        column_names = dataf().columns.tolist()
        print(column_names)
        init_var_x = column_names[0]
        init_var_y = column_names[1]
        quant = {col: col for col in type_var()["quant"]}
        quali = {col: col for col in type_var()["quali"]}
        init_value_var_x = (dataf()[init_var_x].min(), dataf()[init_var_x].max())
        init_value_var_y = (dataf()[init_var_y].min(), dataf()[init_var_y].max())

        ui.update_selectize("var_x", choices= {"Variable quantitative :": quant, "Variable qualitative :": quali}, selected= init_var_x)
        ui.update_slider("var_x_slider", min=init_value_var_x[0], max=init_value_var_x[1], value=init_value_var_x)
        ui.update_selectize("var_y", choices= {"Variable quantitative :": quant, "Variable qualitative :": quali}, selected= init_var_y)
        ui.update_slider("var_y_slider", min=init_value_var_y[0], max=init_value_var_y[1], value=init_value_var_y)
        ui.update_selectize("var_z", choices= {"1" : {"" : None}, "Variable qualitative :": quali})
        for color in ["histogram_color", "scatter_color", "violin_color"]:
            ui.update_radio_buttons(color, choices= ["Aucune"]+ type_var()["quali"], selected='Aucune')

    @reactive.effect
    @reactive.event(input.var_z)
    def update_checkbox_options():
        if input.var_z():
            checkbox_options = dataf()[input.var_z()].unique().tolist()
            ui.update_checkbox_group("var_z_modalite", choices=checkbox_options, selected=checkbox_options)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        upload_state.set(False)
        ui.update_selectize("var_x", choices= {"Variable quantitative :": quant, "Variable qualitative :": quali},selected=init_var_x)
        ui.update_slider("var_x_slider", min=init_value_var_x[0], max=init_value_var_x[1],value=init_value_var_x)
        ui.update_selectize("var_y", choices= {"Variable quantitative :": quant, "Variable qualitative :": quali},selected=init_var_y)
        ui.update_slider("var_y_slider", min=init_value_var_y[0], max=init_value_var_y[1],value=init_value_var_y)
        ui.update_selectize("var_z",selected="")
        ui.update_checkbox_group("var_z_modalite", choices=[] , selected=[])
        for color in ["histogram_color", "scatter_color", "violin_color"]:
            ui.update_radio_buttons(color, choices= ["Aucune"]+ type_var()["quali"], selected='Aucune')
    
    @reactive.calc
    def dataf():
        if upload_state.get():
            sep = [" ", ";", "\t", ","]
            i = 0
            df = pd.read_csv(input.file()[0]['datapath'], sep=sep[i])
            while len(df.columns.tolist()) == 1:
                i+=1
                df = pd.read_csv(input.file ()[0]['datapath'], sep=sep[i])
        else:   
            df = data
        return df
    
    @reactive.calc
    def type_var():
        df = dataf()
        column_names = df.columns.tolist()
        column_types = df.dtypes
        quantitative_vars = [col for col, dtype in zip(column_names, column_types) if dtype != "object"]
        qualitative_vars = [col for col, dtype in zip(column_names, column_types) if dtype == "object"]
        return {"quant": quantitative_vars, "quali" : qualitative_vars}
        
    @reactive.calc
    def dataf_filtre():
        idx1 = pd.Series(True, index=dataf().index)
        idx2 = pd.Series(True, index=dataf().index)
        idx3 = pd.Series(True, index=dataf().index)

        if input.var_x() in type_var()["quant"]:
            var_x_range = input.var_x_slider()
            idx1 = dataf()[input.var_x()].between(var_x_range[0], var_x_range[1])

        if input.var_y() in type_var()["quant"]:
            var_y_range = input.var_y_slider()
            idx2 = dataf()[input.var_y()].between(var_y_range[0], var_y_range[1])

        if input.var_z() in type_var()["quali"]:
            idx3 = dataf()[input.var_z()].isin(input.var_z_modalite())

        return dataf()[idx1 & idx2 & idx3]
    
    @render.data_frame
    def table():
        return render.DataGrid(dataf_filtre(), filters=True)

    @render.data_frame
    def summary_table():
        return render.DataGrid(table_summary(dataf()))
    
    @render_plotly
    def corrplot():
        return plot_corr(dataf()[type_var()["quant"]], type_var()["quant"])

    @render_plotly
    def scatterplot():
        if input.var_x() in dataf_filtre().columns.tolist() and input.var_y() in dataf_filtre().columns.tolist():
            return plot_scatter(dataf_filtre(), input.var_x(), input.var_y(), input.scatter_color(), input.smoother())
    
    @render_plotly
    def histogram():
        if input.var_x() in dataf_filtre().columns.tolist() and input.var_y() in dataf_filtre().columns.tolist():
            return plot_histogram(dataf_filtre(), input.var_x(), input.bins(), input.histogram_color())
    
    @render_plotly
    def violinplot():
        if input.var_x() in dataf_filtre().columns.tolist():
            if input.var_z() in dataf_filtre().columns.tolist():
                return plot_violin(dataf_filtre(), input.var_x(), input.var_z(), input.violin_color())
            return plot_violin(dataf_filtre(), input.var_x(), color = input.violin_color())
        
    # Nav_panel : Régression

    @reactive.effect
    @reactive.event(input.type_loi)
    def handle_lien_selection():
        if input.type_loi() == "Bernoulli":
            ui.update_selectize("fc_liens", 
                choices = {"Logit": "Logit (régression logistique)", "Probit": "Probit", "Cloglog": "Cloglog", "Loglog": "Loglog"})
        if input.type_loi() == "Multinomiale":
            ui.update_selectize("fc_liens", choices = {"Logit": "Logit (régression logistique)"})
        if input.type_loi() in ["Poisson" , "Binomiale négative"]:
            ui.update_selectize("fc_liens", choices = {"Log": "Log (régression log-linéaire)"})
    
    @render.ui  
    def reg_card():
        if "OLS" in reg():
            model = reg()['OLS']
            X = set(extract_column_name(expression, data) for expression in model.model.exog_names[1:])
            if len(model.model.exog_names[1:]) > 1:
                T_F = 0
            else:
                nom_x = model.model.exog_names[1]
                nom_x_b = extract_column_name(nom_x, data)
                T_F = int(nom_x != nom_x_b)

            return ui.layout_columns(
                ui.card(
                    ui.card_header("Résumé statistique du modèle:"),
                    ui.output_text_verbatim("summary_model"), full_screen=True
                ),
                ui.navset_card_underline(
                    ui.nav_panel("Graphique", ui.output_plot("linearplot")),
                    ui.nav_panel("Test", ui.output_text_verbatim("test_linearity")),
                    title="Hypothèse de linéarité",
                ),
                ui.navset_card_underline(
                    ui.nav_panel("Graphique", ui.output_plot("normalplot")),
                    ui.nav_panel("Test", ui.output_text_verbatim("test_normal")),
                    title="Hypothèse de normalité",
                ),
                ui.navset_card_underline(
                    ui.nav_panel("Valeur ajusté", ui.output_plot("homoplot_1")),
                    ui.nav_panel("Régresseur", ui.output_plot("homoplot_2")),
                    ui.nav_panel("Test", ui.output_text_verbatim("test_homo")),
                    title="Hypothèse d'homoscédasticité",
                ),
                ui.navset_card_underline(
                    ui.nav_panel("Graphique", ui.output_plot("acfplot")),
                    ui.nav_panel("Test", ui.output_text_verbatim("test_auto"), ui.output_data_frame("box_table")),
                    title="Hypothèse d'indépendance / décorrélation",
                ),
                ui.card(
                    ui.card_header("Non-colinéarité"),
                    ui.output_text_verbatim("vif"),
                    ui.output_data_frame("vif_table"), full_screen=True
                ),
                ui.card(
                    ui.card_header("Points aberrants / atypiques"),
                    output_widget("aberrantplot"), full_screen=True
                ),
                ui.card(
                    ui.card_header("Points leviers"),
                    ui.output_text_verbatim("levier"),
                    output_widget("levierplot"), full_screen=True
                ),
                ui.navset_card_underline(
                    ui.nav_panel("Influence", ui.output_text_verbatim("influence")),
                    ui.nav_panel("Distance de Cook", output_widget("cookplot")),
                    ui.nav_panel("Mesure DFFITS", output_widget("dffitsplot")),
                    title="Mesure d'influence"
                ),
                ui.panel_conditional(f"{len(X)} === 1", 
                    ui.card(
                        ui.card_header("Courbe:"),
                        output_widget("regplot"),
                        ui.panel_conditional(f"{T_F} === 1 ", 
                            ui.input_checkbox("transfo", "Transformation axe")
                        ), full_screen=True
                    ),
                ),
                col_widths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
            )
        
        elif any(value in reg() for value in ["Logist", "Poisson", "Bin_neg"]):
            model = list(reg().values())[0]
            X = set(extract_column_name(expression, data) for expression in model.model.exog_names[1:])
            if len(model.model.exog_names[1:]) > 1:
                T_F = False
            else:
                nom_x = model.model.exog_names[1]
                nom_x_b = extract_column_name(nom_x, data)
                T_F = (nom_x != nom_x_b)
            #mfx = model.get_margeff()
            #print(mfx.summary())
            return ui.layout_columns(
                ui.navset_card_underline(
                    ui.nav_panel("Résumé", ui.output_text_verbatim("summary_model"), 
                        ui.output_text_verbatim("test_model")),
                    ui.nav_panel("Odds ratio", ui.output_data_frame("odds_table")),
                    title="Résultats du modèle:"
                ),
                ui.panel_conditional(f"{len(X)} === 1", 
                    ui.card(
                        ui.card_header("Courbe:"),
                        output_widget("logplot"),
                        ui.layout_columns(
                            ui.input_slider("classe", "Nombre de classes:",
                                min=1, max=10, value=5),
                            ui.panel_conditional(f"'{T_F}' === 'True' ", 
                                ui.input_checkbox("transfo", "Transformation axe")
                            ), full_screen=True
                        )
                    )
                ),
                ui.panel_conditional(f"'{list(reg().keys())[0]}' === 'Logist' ",
                    ui.navset_card_underline(
                        ui.nav_panel("Graphique", ui.output_plot("normalplot")),
                        ui.nav_panel("Test", ui.output_text_verbatim("test_normal")),
                        title="Hypothèse de normalité des résidus de déviance",
                    ),
                ),
                ui.panel_conditional(f"'{list(reg().keys())[0]}' === 'Logist' ",
                    ui.card(
                        ui.card_header("Points aberrants / atypiques"),
                        output_widget("aberrantplot"), full_screen=True
                    )
                ),
                ui.panel_conditional(f"'{list(reg().keys())[0]}' === 'Logist' ",
                    ui.card(
                        ui.card_header("Points leviers"),
                        ui.output_text_verbatim("levier"),
                        output_widget("levierplot"), full_screen=True
                    ),
                ),
                ui.panel_conditional(f"'{list(reg().keys())[0]}' === 'Logist' ",
                    ui.navset_card_underline(
                        ui.nav_panel("Influence", ui.output_text_verbatim("influence")),
                        ui.nav_panel("Distance de Cook", output_widget("cookplot")),
                        title="Mesure d'influence"
                    ),
                ),
                col_widths=[6, 6, 6, 6, 6, 6]
            )
        elif "MNlogit" in reg():
            return ui.layout_columns(
                ui.card(
                    ui.card_header("Résumé statistique du modèle:"),
                    ui.output_text_verbatim("summary_model"), full_screen=True
                ),
                col_widths=[6]
            )
    
    @reactive.calc
    def reg():
        try:
            if input.type_reg() == "Régression linéaire":
                model = smf.ols(f'{input.equation()}', data=dataf()).fit()
                dico = {'OLS': model}
            else:
                data_copy = dataf().copy()
                var_y = input.equation().split('~')[0].strip()
                data_copy[var_y] = dataf()[var_y].astype('category').cat.codes

                if input.type_loi() == "Bernoulli":
                    if input.fc_liens() == "Logit":
                        model = smf.logit(f'{input.equation()}', data=data_copy).fit()
                    if input.fc_liens() == "Probit":
                        model = smf.probit(f'{input.equation()}', data=data_copy).fit()
                    if input.fc_liens() == "Cloglog":
                        model = smf.glm(f'{input.equation()}', 
                            family=sm.families.Binomial(sm.families.links.CLogLog()),
                            data=data_copy).fit()
                    if input.fc_liens() == "Loglog":
                        model = smf.glm(f'{input.equation()}', 
                            family=sm.families.Binomial(sm.families.links.LogLog()),
                            data=data_copy).fit()
                        
                    dico = {'Logist': model}
            
                elif input.type_loi() == "Multinomiale":
                    model = smf.mnlogit(f'{input.equation()}', data=data_copy).fit()
                    dico = {'MNlogit': model}

                elif input.type_loi() == "Poisson":
                    model = smf.poisson(f'{input.equation()}', data=data_copy).fit()
                    dico = {'Poisson': model}

                elif input.type_loi() == "Binomiale négative":
                    model = smf.negativebinomial(f'{input.equation()}', data=data_copy).fit()
                    dico = {'Bin_neg': model}

        except Exception as e:
                #print(e)
                dico={}
                pass

        return dico
    
    @render.data_frame
    def box_table():
        if "OLS" in reg():
            return render.DataGrid(table_box(reg()))
    
    @render.data_frame
    def vif_table():
        if "OLS" in reg():
            return render.DataGrid(table_vif(reg()))
        
    @render.data_frame
    def odds_table():
        if any(value in reg() for value in ['Logist', 'Poisson', 'Bin_neg']):
            return render.DataGrid(table_odds(reg()))
    
    @render_plotly
    def regplot():
        return plot_reg(reg(), dataf(), input.transfo())
    
    @render_plotly
    def logplot():
        return plot_log(reg(), dataf(), input.classe(), input.transfo())
    
    @render_plotly
    def aberrantplot():
        return plot_aberrant(reg())
    
    @render_plotly
    def levierplot():
        return plot_levier(reg())
    
    @render_plotly
    def cookplot():
        return plot_cook(reg())
    
    @render_plotly
    def dffitsplot():
        return plot_dffits(reg())
    
    @render.plot
    def linearplot(): 
        return plot_linear(reg())
        
    @render.plot
    def normalplot():
        return plot_normal(reg())

    @render.plot
    def homoplot_1():
        return plot_homo_1(reg())
    
    @render.plot
    def homoplot_2():
        return plot_homo_2(reg())
    
    @render.plot
    def acfplot():
        return plot_auto(reg())
    
    @render.text
    def summary_model():
        if len(reg().values()) != 0:
            return list(reg().values())[0].summary2()
    
    @render.text
    def vif():
        return text_vif()
    
    @render.text
    def levier():
        return text_levier()
    
    @render.text
    def influence():
        return text_influence()
    
    @render.text
    def test_linearity():
        return text_test_linearity(reg())
        
    @render.text
    def test_normal():
        return text_test_normal(reg())
    
    @render.text
    def test_homo():
        return text_test_homo(reg())
    
    @render.text
    def test_auto():
        return text_test_auto(reg())
    
    @render.text
    def test_model():
        return text_test_model()

app = App(app_ui, server)
app.run()