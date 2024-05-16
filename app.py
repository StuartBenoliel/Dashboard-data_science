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
quant = {"Variable quantitative :": {col: col for col in quantitative_vars}}
quali = {"Variable qualitative :": {col: col for col in qualitative_vars}}

ICONS = {"ellipsis": fa.icon_svg("ellipsis")}

init_var_x = column_names[0]
init_var_y = column_names[1]

init_value_var_x = (data[init_var_x].min(), 
                       data[init_var_x].max())
init_value_var_y = (data[init_var_y].min(), 
                       data[init_var_y].max())

def extract_column_name(expression, data):
    # Trouver tous les mots entre parenthèses
    variables = re.findall(r'\((.*?)\)', expression)
    for variable in variables:
        if variable in data.columns:
            return variable
    return expression

app_ui = ui.page_navbar(
    ui.nav_panel("Vue d'ensemble",
        ui.page_sidebar(
            ui.sidebar(
                ui.input_selectize("var_x", "Variable X:",
                    {"1": quant, "2": quali}, selected=init_var_x
                ),
                ui.panel_conditional(f"{quantitative_vars}.includes(input.var_x)", 
                    ui.input_slider("var_x_slider","Variable X range", min=init_value_var_x[0],
                        max=init_value_var_x[1], value=init_value_var_x)
                ),
                ui.input_selectize("var_y", "Variable Y:",
                    {"1": quant, "2": quali}, selected=init_var_y
                ),
                ui.panel_conditional(f"{quantitative_vars}.includes(input.var_y)", 
                    ui.input_slider("var_y_slider","Variable Y range", min=init_value_var_y[0],
                        max=init_value_var_y[1], value=init_value_var_y)
                ),
                ui.input_selectize("var_z", "Variable Catégorielle:",
                    {"1" : {"" : None}, "2": quali}
                ),
                ui.panel_conditional("input.var_z != '' ", 
                    ui.input_checkbox_group("var_z_modalite", "Modalités:", [], selected=[], inline=True)
                ),
                ui.input_action_button("reset", "Reset filter"),
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
                    ui.panel_conditional(f"{quantitative_vars}.includes(input.var_x)", 
                        ui.input_slider("bins", "Nombre de bins:", min=1, max=30, value=15)
                    ), full_screen=True
                ),
                ui.card(
                    ui.card_header("Scatterplot Variable X vs Variable Y",
                        ui.popover(ICONS["ellipsis"],
                            ui.input_radio_buttons("scatter_color", None,
                                ["Aucune"]+ qualitative_vars, inline=True,
                            ), title="Couleur:",
                        ), class_="d-flex justify-content-between align-items-center",
                    ),
                    output_widget("scatterplot"), full_screen=True,
                ),
                ui.card(
                    ui.card_header("Violinplot Variable X Vs Variable Y",
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
                ["Régression linéaire", "Régression logistique"], selected="Régression linéaire"
            ),
            ui.tooltip(
                ui.input_text("equation", "Entrer l'équation:", " ~ "), 
                "Exemple : Y  ~ np.log(X_1) + X_2 + np.square(X_2)", placement="right"
            ),
            ui.panel_conditional("input.type_reg === 'Régression logistique' ", 
                ui.layout_columns(
                    ui.input_selectize("type_loi", "Loi du model:",
                        {"1" : {"Variable binaire" : {"Bernoulli" : "Bernoulli"}}, 
                         "2" : {"Variable catégorielle" : {"Multinomiale" : "Multinomiale"}}, 
                         "3" : {"Variable de comptage" : {"Poisson" : "Poisson", "Binomiale négative" : "Binomiale négative"}}
                        }, selected = "Bernoulli"
                    ),
                    ui.input_selectize("fc_liens", "Fonctions de liens:",
                        {"Logit": "Logit", "Probit": "Probit", "Cloglog": "Cloglog", "Loglog": "Loglog"},
                        selected = "Logit"
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
        
    @reactive.effect
    @reactive.event(input.var_x)
    def handle_variable_x_selection():
        if input.var_x() in quantitative_vars:
            min_value_x = data[input.var_x()].min()
            max_value_x = data[input.var_x()].max()
            ui.update_slider("var_x_slider", min=min_value_x, max=max_value_x, value=(min_value_x, max_value_x))

    @reactive.effect
    @reactive.event(input.var_y)
    def handle_variable_y_selection():
        if input.var_y() in quantitative_vars:
            min_value_y = data[input.var_y()].min()
            max_value_y = data[input.var_y()].max()
            ui.update_slider("var_y_slider", min=min_value_y, max=max_value_y, value=(min_value_y, max_value_y))

    @reactive.effect
    @reactive.event(input.var_z)
    def update_checkbox_options():
        if input.var_z():
            checkbox_options = data[input.var_z()].unique().tolist()
            ui.update_checkbox_group("var_z_modalite", choices=checkbox_options, selected=checkbox_options)

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        ui.update_selectize("var_x",selected=init_var_x)
        ui.update_slider("var_x_slider", min=init_value_var_x[0], max=init_value_var_x[1],value=init_value_var_x)
        ui.update_selectize("var_y",selected=init_var_y)
        ui.update_slider("var_y_slider", min=init_value_var_y[0], max=init_value_var_y[1],value=init_value_var_y)
        ui.update_selectize("var_z",selected="")
        ui.update_checkbox_group("var_z_modalite", choices=[] , selected=[])
        for color in ["histogram_color", "scatter_color", "violin_color"]:
            ui.update_radio_buttons(color, selected='Aucune')
        
    @reactive.calc
    def data_filtre():
        idx1 = pd.Series(True, index=data.index)
        idx2 = pd.Series(True, index=data.index)
        idx3 = pd.Series(True, index=data.index)

        if input.var_x() in quantitative_vars:
            var_x_range = input.var_x_slider()
            idx1 = data[input.var_x()].between(var_x_range[0], var_x_range[1])

        if input.var_y() in quantitative_vars:
            var_y_range = input.var_y_slider()
            idx2 = data[input.var_y()].between(var_y_range[0], var_y_range[1])

        if input.var_z():
            idx3 = data[input.var_z()].isin(input.var_z_modalite())

        return data[idx1 & idx2 & idx3]
    
    @render.data_frame
    def table():
        return render.DataGrid(data_filtre(), filters=True)

    @render.data_frame
    def summary_table():
        return render.DataGrid(table_summary(data_filtre()))
    
    @render_plotly
    def corrplot():
        return plot_corr(data[quantitative_vars], quantitative_vars)

    @render_plotly
    def scatterplot():
        return plot_scatter(data_filtre(), input.var_x(), input.var_y(), input.scatter_color())
    
    @render_plotly
    def histogram():
        return plot_histogram(data_filtre(), input.var_x(), input.bins(), input.histogram_color())
    
    @render_plotly
    def violinplot():
        return plot_violin(data_filtre(), input.var_x(), input.var_y(), input.violin_color())
        
    # Nav_panel : Régression

    @reactive.effect
    @reactive.event(input.type_loi)
    def handle_lien_selection():
        if input.type_loi() == "Binomiale":
            ui.update_selectize("fc_liens", 
                choices = {"Logit": "Logit", "Probit": "Probit", "Cloglog": "Cloglog", "Loglog": "Loglog"},
                selected = "Logit")
        if input.type_loi() == "Multinomiale":
            ui.update_selectize("fc_liens", choices = {"Logit": "Logit"}, selected = "Logit")
        if input.type_loi() in ["Poisson" , "Binomiale négative"]:
            ui.update_selectize("fc_liens", choices = {"?": "?"}, selected = "?")
    
    @render.ui  
    def reg_card():
        if "OLS" in reg():
            model = reg()['OLS']
            X = model.model.exog[:, 1:]
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
                    ui.nav_panel("Distance de Cook", output_widget("cookplot")),
                    ui.nav_panel("Mesure DFFITS", output_widget("dffitsplot")),
                    title="Mesure d'influence",
                ),
                ui.panel_conditional(f"{X.shape[1]} === 1", 
                    ui.card(
                        ui.card_header("Courbe:"),
                        output_widget("regplot"), full_screen=True
                    )
                ),
                col_widths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
            )
        
        elif any(value in reg() for value in ["Logist", "Poisson", "Bin_neg"]):
            model = list(reg().values())[0]
            X = model.model.exog[:, 1:]
            return ui.layout_columns(
                ui.card(
                    ui.card_header("Résumé statistique du modèle:"),
                    ui.output_text_verbatim("summary_model"), full_screen=True
                ), 
                ui.panel_conditional(f"{X.shape[1]} === 1", 
                    ui.card(
                        ui.card_header("Courbe:"),
                        output_widget("logplot"),
                        ui.input_slider("classe", "Nombre de classes:",
                            min=1, max=20, value=5
                        ), full_screen=True
                    )
                ),
                ui.card(
                    ui.card_header("Odds ratio (rapport de cotes):"),
                    ui.output_data_frame("odds_table"), full_screen=True
                ),
                col_widths=[6, 6, 6]
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
                model = smf.ols(f'{input.equation()}', data=data).fit()
                dico = {'OLS': model}
            else:
                data_copy = data.copy()
                var_y = input.equation().split('~')[0].strip()
                data_copy[var_y] = data[var_y].astype('category').cat.codes

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
        return plot_reg(reg())
    
    @render_plotly
    def logplot():
        return plot_log(reg(), input.classe())
    
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

app = App(app_ui, server)
app.run()