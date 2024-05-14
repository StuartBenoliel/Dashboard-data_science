import faicons as fa
import plotly.express as px
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from shared import app_dir, data
from shinywidgets import output_widget, render_plotly
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import matplotlib.gridspec as gridspec
from shiny import App, reactive, render, ui
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import linear_rainbow, linear_reset, het_breuschpagan, het_white, acorr_ljungbox
from scipy.stats import shapiro, ttest_1samp
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.outliers_influence import OLSInfluence
from scipy.stats import f
import plotly.graph_objects as go

# Obtenez les noms des colonnes
column_names = data.columns.tolist()

# Obtenez les types de données des colonnes
column_types = data.dtypes

# Séparez les noms des colonnes en variables quantitatives et qualitatives
quantitative_vars = [col for col, dtype in zip(column_names, column_types) if dtype != "object"]
qualitative_vars = [col for col, dtype in zip(column_names, column_types) if dtype == "object"]

# Créez les options pour les choix 1 et 2
choices_1 = {"Variable quantitative :": {col: col for col in quantitative_vars}}
choices_2 = {"Variable qualitative :": {col: col for col in qualitative_vars}}

ICONS = {
    "ellipsis": fa.icon_svg("ellipsis")
}

init_var_x= column_names[0]
init_var_y= column_names[1]
init_value_var_x = (data[init_var_x].min(), 
                       data[init_var_x].max())
init_value_var_y = (data[init_var_y].min(), 
                       data[init_var_y].max())

# Add page title and sidebar
app_ui = ui.page_navbar(
    ui.nav_panel(
        "Vue d'ensemble",
        ui.page_sidebar(
            ui.sidebar(
                ui.input_selectize(
                    "var_x",
                    "Variable X:",
                    {"1": choices_1, "2": choices_2},
                    selected=init_var_x,
                ),
                ui.output_ui("slider_x"),
                ui.input_selectize(
                    "var_y",
                    "Variable Y:",
                    {"1": choices_1, "2": choices_2},
                    selected=init_var_y
                ),
                ui.output_ui("slider_y"),
                ui.input_selectize(
                    "var_z",
                    "Variable Catégorielle:",
                    {"1" : {"" : None}, "2": choices_2},
                ),
                ui.output_ui("modalite_z"),
                ui.input_action_button("reset", "Reset filter"),
                open="desktop",
            ),
            ui.layout_columns(
                ui.card(
                    ui.card_header("Données"), 
                    ui.output_data_frame("table"), 
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Résumés variables quantitatives"), 
                    ui.output_data_frame("table_summary"), 
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Corrélations entre variables quantitatives"), 
                    output_widget("corrplot"), 
                    full_screen=True
                ),
                ui.card(
                    ui.card_header(
                        "Histogramme Variable X",
                        ui.popover(
                            ICONS["ellipsis"],
                            ui.input_radio_buttons(
                                "histogram_color",
                                None,
                                ["Aucune"]+ qualitative_vars,
                                inline=True,
                            ),
                            title="Couleur:",
                            placement="top",
                        ),
                        class_="d-flex justify-content-between align-items-center",
                    ),
                    output_widget("histogram"),
                    ui.output_ui("slider_bins"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header(
                        "Scatterplot Variable X vs Variable Y",
                        ui.popover(
                            ICONS["ellipsis"],
                            ui.input_radio_buttons(
                                "scatter_color",
                                None,
                                ["Aucune"]+ qualitative_vars,
                                inline=True,
                            ),
                            title="Couleur:",
                            placement="top",
                        ),
                        class_="d-flex justify-content-between align-items-center",
                    ),
                    output_widget("scatterplot"),
                    full_screen=True,
                ),
                ui.card(
                    ui.card_header(
                        "Violinplot Variable X Vs Variable Y",
                        ui.popover(
                            ICONS["ellipsis"],
                            ui.input_radio_buttons(
                                "violin_color",
                                None,
                                ["Aucune"]+ qualitative_vars,
                                inline=True,
                            ),
                            title="Couleur:",
                            placement="top",
                        ),
                        class_="d-flex justify-content-between align-items-center",
                    ),
                    output_widget("violinplot"),
                    full_screen=True,
                ),
                col_widths=[12, 12, 12, 12],
            )
        )
    ),
    ui.nav_panel("Régression", 
                 ui.layout_columns(
                    ui.input_selectize(
                        "type_reg",
                        "Type de régression:",
                        ["Régression linéaire", "Régression logistique"],
                        selected="Régression linéaire",
                    ),
                    ui.tooltip(
                        ui.input_text("equation", "Entrer l'équation:", " ~ "), 
                        "Exemple : Y  ~ np.log(X_1) + X_2 + np.square(X_2)",
                        placement="right",
                    ),
                    ui.output_ui("lois"),
                    ui.output_ui("liens"),
                    ui.output_ui("reg_card"),
                    col_widths=[3, 3, 3, 3, 12],
                ),
    ),
    title="Dashboard Data-Science",
    fillable=True,
)


def server(input, output, session):

    # Nav_panel : Vue d'ensemble
    @render.ui
    @reactive.event(input.var_x)
    def slider_x():
        if input.var_x() in quantitative_vars:
            return ui.input_slider(
                        "var_x_slider",
                        "Variable X range",
                        min=init_value_var_x[0],
                        max=init_value_var_x[1],
                        value=init_value_var_x,
                    )

    @render.ui
    @reactive.event(input.var_y)
    def slider_y():
        if input.var_y() in quantitative_vars:
            return ui.input_slider(
                        "var_y_slider",
                        "Variable Y range",
                        min=init_value_var_y[0],
                        max=init_value_var_y[1],
                        value=init_value_var_y,
                    )
        
    @render.ui
    @reactive.event(input.var_x)
    def slider_bins():
        if input.var_x() in quantitative_vars:
            return ui.input_slider(
                            "bins",
                            "Nombre de bins:",
                            min=1,  # Nombre minimum de bins
                            max=50,  # Nombre maximum de bins
                            value=20,  # Nombre de bins initial
                        )

    @render.ui
    @reactive.event(input.var_z) 
    def modalite_z():
        if input.var_z() != '':
            return ui.input_checkbox_group(
                        "var_z_modalite",
                        "Modalités:",
                        [],
                        selected=[],
                        inline=True,
                    )
        
    @reactive.effect
    @reactive.event(input.var_x)
    def handle_variable_x_selection():
        selected_variable_x = input.var_x()
        min_value_x = data[selected_variable_x].min()
        max_value_x = data[selected_variable_x].max()
        ui.update_slider("var_x_slider", min=min_value_x, max=max_value_x, value=(min_value_x, max_value_x))

    @reactive.effect
    @reactive.event(input.var_y)
    def handle_variable_y_selection():
        selected_variable_y = input.var_y()
        min_value_y = data[selected_variable_y].min()
        max_value_y = data[selected_variable_y].max()
        ui.update_slider("var_y_slider", min=min_value_y, max=max_value_y, value=(min_value_y, max_value_y))

    @reactive.effect
    @reactive.event(input.var_z)
    def update_checkbox_options():
        if input.var_z():
            checkbox_options = data[input.var_z()].unique().tolist()
            ui.update_checkbox_group("var_z_modalite", choices=checkbox_options, selected=checkbox_options)
        else:
            ui.update_checkbox_group("var_z_modalite", choices=[], selected=[])

    @reactive.effect
    @reactive.event(input.reset)
    def _():
        ui.update_selectize("var_x",selected=init_var_x)
        ui.update_slider("var_x_slider", min=init_value_var_x[0], max=init_value_var_x[1],value=init_value_var_x)
        ui.update_selectize("var_y",selected=init_var_y)
        ui.update_slider("var_y_slider", min=init_value_var_y[0], max=init_value_var_y[1],value=init_value_var_y)
        ui.update_selectize("var_z",selected="")
        ui.update_checkbox_group("var_z_modalite", choices=[] ,selected=[])
        for e in ["histogram_color", "scatter_color", "violin_color"]:
            ui.update_radio_buttons(e, selected='Aucune')
        
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
        return render.DataGrid(data_filtre(),filters=True)

    @render.data_frame
    def table_summary():
        summary = data_filtre().describe().transpose()
        summary['Variable'] = summary.index
        summary = summary[['Variable'] + [col for col in summary.columns if col != 'Variable']]

        return render.DataGrid(summary)
    
    @render_plotly
    def corrplot():
        correlations = data[quantitative_vars].corr()
        correlations['nom'] = correlations.index
        X = pd.melt(correlations, id_vars='nom', var_name='Variable', value_name='Cor')
        X['nom'] = pd.Categorical(X['nom'], categories=quantitative_vars[::-1], ordered=True)
        X['Variable'] = pd.Categorical(X['Variable'], categories=quantitative_vars[::-1], ordered=True)
        

        fig = px.imshow(X.pivot(index='Variable', columns='nom', values='Cor'),
                color_continuous_scale=[[0,"#636EFA"], [0.5, 'white'], [1, '#F24B4B']],
                labels=dict(x='Variable', y='nom'))
        
        for i, row in enumerate(X.pivot(index='Variable', columns='nom', values='Cor').values):
            for j, val in enumerate(row):
                fig.add_annotation(text=f"{val:.3f}", x=list(quantitative_vars[::-1])[i], y=list(quantitative_vars[::-1])[j], showarrow=False)
        
        fig.update_layout(
                        xaxis_title="",
                        yaxis_title="")
        
        return fig

    @render_plotly
    def scatterplot():
        color = input.scatter_color()
        return px.scatter(
            data_filtre(),
            x=input.var_x(),
            y=input.var_y(),
            color=None if color == "Aucune" else color,
            template="plotly_white",
        )
    
    @render_plotly
    def histogram():
        color = input.histogram_color()

        fig = px.histogram(
            data_filtre(),
            x=input.var_x(),
            nbins=input.bins(),
            color= None if color == "Aucune" else color,
            template="plotly_white",
            histnorm='percent'
        )
        fig.update_traces(marker_line_color="black", marker_line_width=2)
        if color == "Aucune":
            fig.update_traces(marker_color="#F24B4B")

        return fig
    
    @render_plotly
    def violinplot():
        color = input.violin_color()
        return px.violin(
            data_filtre(),
            y=input.var_y(),
            x=input.var_x(),
            color=None if color == "Aucune" else color,
            box=True,
            points='all',
            template="plotly_white"
        )
        
    # Nav_panel : Régression
    @render.ui
    @reactive.event(input.type_reg)
    def lois():
        if input.type_reg() == "Régression logistique":
            return ui.input_selectize(
                    "type_loi",
                    "Loi du model:",
                    { "1" : {"Variable binaire" : {"Bernoulli" : "Bernoulli"}}, "2" : {"Variable catégorielle" : {"Multinomiale" : "Multinomiale"}}, 
                     "3" : {"Variable de comptage" : {"Poisson" : "Poisson", "Binomiale négative" : "Binomiale négative"}}},
                    selected="Bernoulli",
                )

    @render.ui
    @reactive.event(input.type_loi)
    def liens():
        if input.type_loi() == "Bernoulli":
            return ui.input_selectize(
                    "fc_liens",
                    "Fonctions de liens:",
                    {"Logit": "Logit", "Probit": "Probit", "Cloglog": "Cloglog", "Loglog": "Loglog"},
                    selected = "Logit",
                )
        if input.type_loi() == "Multinomiale":
            return ui.input_selectize(
                    "fc_liens",
                    "Fonctions de liens:",
                    {"Logit": "Logit"},
                    selected = "Logit",
                )
        else:
            return ui.input_selectize(
                    "fc_liens",
                    "Fonctions de liens:",
                    {"1": "?"},
                    selected = "?",
                )
        
    @render.ui
    @reactive.event(input.linearity_radio) 
    def linearity_section():
        if input.linearity_radio() == 'plot':
            return ui.output_plot("linearplot")
        
        elif input.linearity_radio() == 'test':
            return ui.output_text_verbatim("test_linearity")
        
    @render.ui
    @reactive.event(input.homo_radio) 
    def homo_section():
        if input.homo_radio() == 'plot_1':
            return ui.output_plot("homoplot_1")
        
        elif input.homo_radio() == 'plot_2':
            return ui.output_plot("homoplot_2")
        
        elif input.homo_radio() == 'test':
            return ui.output_text_verbatim("test_homo")
    
    @render.ui
    @reactive.event(input.normal_radio) 
    def normal_section():
        if input.normal_radio() == 'plot':
            return ui.output_plot("normalplot")
        
        elif input.normal_radio() == 'test':
            return ui.output_text_verbatim("test_normal")
        
    @render.ui
    @reactive.event(input.auto_radio) 
    def auto_section():
        if input.auto_radio() == 'plot':
            return ui.output_plot("acfplot")
        
        elif input.auto_radio() == 'test':
            return ui.output_text_verbatim("test_auto"), ui.output_data_frame("box_table")
        
    @render.ui
    @reactive.event(input.influence_radio) 
    def influence_section():
        if input.influence_radio() == 'plot_1':
            return output_widget("cookplot")
        
        elif input.influence_radio() == 'plot_2':
            return output_widget("dffitsplot")
    
    @render.ui    
    def reg_card():
        try:
            if "OLS" in reg():
                return ui.layout_columns(
                        ui.card(
                            ui.card_header("Résumé statistique du modèle:"),
                            ui.output_text_verbatim("summary_model"),
                            full_screen=True
                        ),
                        ui.card(
                            ui.card_header(ui.input_radio_buttons(id="linearity_radio",
                                                    label="Hypothèse de linéarité:",
                                                    choices={"plot": "Graphique", "test": "Test"},
                                                    inline=True,
                                                    selected="plot" 
                                            )
                            ),
                            ui.output_ui("linearity_section"),
                            full_screen=True
                        ), 
                        ui.card(
                            ui.card_header(ui.input_radio_buttons(id="normal_radio",
                                                    label="Hypothèse de normalité:",
                                                    choices={"plot": "Graphique", "test": "Test"},
                                                    inline=True,
                                                    selected="plot" 
                                            )
                            ),
                            ui.output_ui("normal_section"),
                            full_screen=True
                        ), 
                        ui.card(
                            ui.card_header(ui.input_radio_buttons(id="homo_radio",
                                                    label="Hypothèse d'homoscédasticité:",
                                                    choices={"plot_1": "Valeur ajusté", "plot_2": "Régresseur", "test": "Test"},
                                                    inline=True,
                                                    selected="plot_1" 
                                            )
                            ),
                            ui.output_ui("homo_section"),
                            full_screen=True
                        ), 
                        ui.card(
                            ui.card_header(ui.input_radio_buttons(id="auto_radio",
                                                    label="Hypothèse d'indépendance (autocorrélation):",
                                                    choices={"plot": "Graphique", "test": "Test"},
                                                    inline=True,
                                                    selected="plot" 
                                            )
                            ),
                            ui.output_ui("auto_section"),
                            full_screen=True
                        ), 
                        ui.card(
                            ui.card_header("Non-colinéarité:"),
                            ui.output_text_verbatim("vif"),
                            ui.output_data_frame("vif_table"),
                            full_screen=True
                        ),
                        ui.card(
                            ui.card_header("Points aberrants / atypiques:"),
                            output_widget("aberrantplot"),
                            full_screen=True
                        ),
                        ui.card(
                            ui.card_header("Points leviers:"),
                            ui.output_text_verbatim("levier"),
                            output_widget("levierplot"),
                            full_screen=True
                        ),
                        ui.card(
                            ui.card_header(ui.input_radio_buttons(id="influence_radio",
                                                    label="Mesure d'influence:",
                                                    choices={"plot_1": "Distance de Cook", "plot_2": "Mesure DFFITS"},
                                                    inline=True,
                                                    selected="plot_1" 
                                            )
                            ),
                            ui.output_text_verbatim("influence"),
                            ui.output_ui("influence_section"),
                            full_screen=True
                        ),
                        ui.output_ui("reg_plot_card"),
                        col_widths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6]
                )
            
            if "Logist" in reg():
                model = reg()['Logist']
                odds_ratios = pd.DataFrame(
                        {
                            "OR": model.params,
                            "Lower CI": model.conf_int()[0],
                            "Upper CI": model.conf_int()[1],
                        }
                    )
                odds_ratios = np.exp(odds_ratios)

                odds_ratios= odds_ratios.style.set_table_styles(
                [{'selector': 'td', 'props': [('padding', '5px 10px')]}]
                )

                return ui.layout_columns(
                        ui.card(
                            ui.card_header("Résumé statistique du modèle:"),
                            ui.output_text_verbatim("summary_model"),
                            full_screen=True
                        ), 
                        ui.output_ui("log_plot_card"),
                        ui.card(
                            ui.card_header("Odds ratio (cotes):"),
                            odds_ratios,
                            full_screen=True
                        ),

                        col_widths=[6, 6],
                )
        except Exception as e:
            print(e)
            pass
    
    @render.ui    
    def reg_plot_card():
        model = reg()['OLS']
        X = model.model.exog[:, 1:]
        if X.shape[1] == 1:
            return ui.card(
                        ui.card_header("Courbe:"),
                        output_widget("regplot"),
                        full_screen=True
                    )
        
    @render.ui    
    def log_plot_card():
        model = reg()['Logist']
        X = model.model.exog[:, 1:]
        if X.shape[1] == 1:
            return ui.card(
                        ui.card_header("Courbe:"),
                        output_widget("logplot"),
                        ui.input_slider(
                        "classe",
                        "Nombre de classes:",
                        min=1,  # Nombre minimum de bins
                        max=20,  # Nombre maximum de bins
                        value=5,  # Nombre de bins initial
                        ),
                        full_screen=True
                    )
    
    @reactive.calc
    def reg():
        if input.type_reg() == "Régression linéaire":
            model = smf.ols(f'{input.equation()}', data=data).fit()
            dico = {'OLS': model}

        else:
            data_copy = data.copy()
            var_y = input.equation().split('~')[0].strip()
            data_copy[var_y] = data[var_y].astype('category').cat.codes

            if input.type_loi() == "Bernoulli":
                #model = smf.glm(f'{input.equation()}',family = sm.families.Binomial(), data=data_copy).fit()
                if input.fc_liens() == "Logit":
                    model = smf.logit(f'{input.equation()}', data=data_copy).fit()
                if input.fc_liens() == "Probit":
                    model = smf.probit(f'{input.equation()}', data=data_copy).fit()
                dico = {'Logist': model}
        
            elif input.type_loi() == "Multinomiale":
                model = smf.mnlogit(f'{input.equation()}', data=data_copy).fit()
                dico = {'MNlogit': model}

            elif input.type_loi() == "Poisson":
                model = smf.poisson(f'{input.equation()}', data=data_copy).fit()
                dico = {'Poisson': model}

        return dico
    
    @render.text
    def summary_model():
        return list(reg().values())[0].summary2()
    
    @render.text
    def vif():
        explanation = """
        Le VIF (variance inflation factor / facteur d'inflation de la variance) est 
        une mesure de l'importance de la colinéarité entre les variables 
        explicatives dans un modèle de régression linéaire. 
        Il indique à quel point la variance d'un coefficient de régression est 
        augmentée en raison de la corrélation entre les variables indépendantes. 
        Un VIF supérieur à 10 est souvent considéré comme indiquant une colinéarité 
        problématique (on ne considère pas le VIF de l'intercept).
        """.replace('\n        ', '\n')
        return explanation.strip()
    
    @render.text
    def levier():
        explanation = """
        Un point levier est un point dont la coordonnée sur l’axe X est 
        significativement différente de celles des autres points. La notion de 
        point levier renvoie à la distance d’un point du centre de gravité du 
        nuage de point et par conséquent est distincte de la notion de valeur 
        aberrante. En fait, un point levier est atypique au niveau des variables 
        explicatives et l’on doit se poser la question de la considération d’un tel 
        point : erreur de mesure, inhomogénéité de la population, ...
        """.replace('\n        ', '\n')
        return explanation.strip()
    
    @render.text
    def influence():
        explanation = """
        Un point influent est un point qui exerce une influence significative sur 
        l’équation de la droite de régression. On entend par cela que l’équation de 
        la droite de régression change de façon importante lorsque l’on supprime 
        ce point. Un point influent se caractérise par un levier important et 
        un résidu atypique (significativement plus grand en valeur absolue).

        Une distance de Cook / Mesure DFFITS importante peut être le résultat soit 
        d’un résidu standardisé grand (point aberrant), soit d’un levier 
        important, soit des deux.
        """.replace('\n        ', '\n')
        return explanation.strip()
    
    @render.data_frame
    def box_table():
        model = reg()['OLS']
        acorr_result = acorr_ljungbox(model.resid)
        test_df = pd.DataFrame({
            'Lag': range(1,11),
            'Statistique de test': acorr_result.lb_stat,
            'p-valeur': acorr_result.lb_pvalue
        })
        return render.DataGrid(test_df)
    
    @render.data_frame
    def vif_table():
        model = reg()['OLS']
        X = model.model.exog
        vif = pd.DataFrame()
        vif['VIF'] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
        vif['variable'] = model.model.exog_names
        return render.DataGrid(vif)
    
    @render_plotly
    def regplot():
        model = reg()['OLS']
        X = model.model.exog[:, 1:]
        df = pd.DataFrame({
            model.model.exog_names[1]: X.flatten(),
            model.model.endog_names: model.model.endog
        })
        x = np.arange(np.min(X), np.max(X), 0.1)

        fig = px.scatter(df, x=model.model.exog_names[1], 
                         y=model.model.endog_names, trendline="ols")
        fig.update_traces(marker=dict(color='white', line=dict(color='black', width=1)))
        fig.update_traces(line=dict(color='black'))

        # Calculer les intervalles de confiance de la droite de régression
        conf_int = model.conf_int(alpha=0.05)

        # Ajouter l'intervalle de confiance de la droite de régression
        fig.add_traces(go.Scatter(x=x, 
                               y=conf_int.loc['Intercept', 0] + conf_int.loc[model.model.exog_names[1], 0] * x,
                               mode='lines',
                               line=dict(color='red', dash='dash'), opacity=0.5,
                               name='IC droite de régression à 95%'))
    
        fig.add_traces(go.Scatter(x=x, 
                                y=conf_int.loc['Intercept', 1] + conf_int.loc[model.model.exog_names[1], 1] * x,
                                mode='lines',
                                line=dict(color='red', dash='dash'), opacity=0.5,
                                name='IC droite de régression à 95%',
                                showlegend=False))

        # Calculer l'intervalle de confiance d'une valeur prédite
        predict_ci = model.get_prediction().summary_frame(alpha=0.05)

        # Ajouter l'intervalle de confiance d'une valeur prédite
        fig.add_traces(go.Scatter(x=df[model.model.exog_names[1]], 
                                y=predict_ci['obs_ci_lower'],
                                mode='lines',
                                line=dict(color='blue', dash='dash', width=1), opacity=0.5,
                                name='IC valeur prédite à 95%'))

        fig.add_traces(go.Scatter(x=df[model.model.exog_names[1]], 
                                y=predict_ci['obs_ci_upper'],
                                mode='lines',
                                line=dict(color='blue', dash='dash', width=1), opacity=0.5,
                                name='IC valeur prédite à 95%',
                                showlegend=False))

        fig.update_layout(
                        template="plotly_white",
                        legend=dict(
                          orientation="h",
                          x=0.5,
                          y=1.1,
                          xanchor='center',
                          yanchor='middle'
                      ))

        return fig
    
    @render_plotly
    def logplot():
        model = reg()['Logist']
        X = model.model.exog[:, 1:]
        df = pd.DataFrame({
            model.model.exog_names[1]: X.flatten(),
            model.model.endog_names: model.model.endog
        })
        df = df.sort_values(by=model.model.exog_names[1])
        x = np.arange(np.min(X), np.max(X), 0.1)
        p_hat = np.exp(model.params[0] + x * model.params[1]) / (1 + np.exp(model.params[0] + x * model.params[1]))

        nb_bins = input.classe()
        bin_edges = np.linspace(df[model.model.exog_names[1]].min(), df[model.model.exog_names[1]].max(), nb_bins + 1)
        bins = pd.cut(df[model.model.exog_names[1]], bins=bin_edges, labels=False, include_lowest=True)
        df['bin'] = bins

        # Calculer la moyenne de model.model.endog_names pour chaque bin
        means = df.groupby('bin')[model.model.endog_names].mean()

        data_emp = pd.DataFrame({
            'borne_inf': [df[df['bin'] == i][model.model.exog_names[1]].min() for i in range(0, nb_bins)],
            'borne_sup': [df[df['bin'] == i][model.model.exog_names[1]].max() for i in range(0, nb_bins)],
            'proportion': means.values
        })
        fig = px.scatter(df, x=model.model.exog_names[1], y=model.model.endog_names)
        fig.add_traces(go.Scatter(x=x, y=p_hat, name='Probabilité prédite', showlegend=True))
        fig.update_traces(line=dict(color="#3854A6"))
        fig.update_traces(marker=dict(color='white', line=dict(color='black', width=1)))
        fig.add_trace(go.Scatter(x=data_emp['borne_inf'], y=data_emp['proportion'], mode='markers', 
                                 showlegend=False, marker=dict(color='#F24B4B', size=6),
                                 name="Probabilité prédite par regroupement"))
        fig.add_trace(go.Scatter(x=data_emp['borne_sup'], y=data_emp['proportion'], mode='markers',
                                  showlegend=False, marker=dict(color='white', line=dict(color='#F24B4B', width=1)),
                                  name="Probabilité prédite par regroupement"))
        
        affichage = True
        for i in range(len(data_emp)):
            fig.add_trace(go.Scatter(x=[data_emp['borne_inf'][i], data_emp['borne_sup'][i]], 
                                    y=[data_emp['proportion'][i], data_emp['proportion'][i]], 
                                    mode='lines', line=dict(color='#F24B4B'), 
                                    name="Probabilité prédite par regroupement", 
                                    showlegend=affichage))
            affichage= False
            
        fig.update_layout(
                        template="plotly_white",
                        legend=dict(
                          orientation="h",
                          x=0.5,
                          y=1.1,
                          xanchor='center',
                          yanchor='middle'
                        ))

        return fig
    
    @render.plot
    def linearplot():  
        model = reg()['OLS']
        residuals = model.resid
        fitted_values = model.fittedvalues
        local = lowess(residuals, fitted_values)

        fig, ax = plt.subplots()
        ax.scatter(fitted_values, residuals, edgecolors='k', alpha=0.6)
        ax.plot(local[:,0], local[:,1], color = 'black')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set_xlabel('Values ajustées / Fitted Values')
        ax.set_ylabel('Résidus / Residuals')
            
        return fig
        
    @render.plot
    def normalplot():
        model = reg()['OLS']
        studentized_residuals = model.get_influence().resid_studentized_external
        
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        # QQ plot des résidus standardisés
        ax0 = plt.subplot(gs[0, 0])
        stats.probplot(studentized_residuals, dist="norm", plot=ax0)
        ax0.get_lines()[0].set_markeredgecolor('k')
        ax0.get_lines()[0].set_alpha(0.6)
        ax0.set_title('Normal Q-Q Plot des résidus studentisés')
        ax0.set_xlabel('Quantiles théoriques')
        ax0.set_ylabel("Quantiles observés")
        # Histogramme des résidus standardisés avec estimation de densité
        ax1 = plt.subplot(gs[0, 1])
        sns.histplot(studentized_residuals, kde=True, ax=ax1, bins=30, edgecolor='k', alpha=0.6)
        ax1.set_xlabel('Résidus studentisés / Studentized Residuals par VC')
        ax1.set_ylabel('Fréquence')
        
        return fig

    @render.plot
    def homoplot_1():
        model = reg()['OLS']
        studentized_residuals = model.get_influence().resid_studentized_external
        sqrt_absolute_residuals = np.sqrt(np.abs(studentized_residuals))
        local = lowess(sqrt_absolute_residuals, model.fittedvalues)

        fig, ax = plt.subplots()
        ax.scatter(model.fittedvalues, sqrt_absolute_residuals, edgecolors='k', alpha=0.6)
        ax.plot(local[:,0], local[:,1], color = 'black')
        ax.set_xlabel("Valeur ajustée")
        ax.set_ylabel("Racine carrée des résidus studentisés")

        return fig
    
    @render.plot
    def homoplot_2():
        import math
        model = reg()['OLS']
        X = model.model.exog[:, 1:]
        studentized_residuals = model.get_influence().resid_studentized_external
        num_exog_vars = X.shape[1]
        num_cols = math.ceil(num_exog_vars**0.5)
        num_rows = num_cols -1

        if num_cols*num_rows < num_exog_vars:
            num_rows = num_cols

        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols)

        if num_exog_vars == 1:
            local = lowess(studentized_residuals, X[:, 0])
            axs.scatter(X[:, 0], studentized_residuals, edgecolors='k', alpha=0.6)
            axs.plot(local[:, 0], local[:, 1], color='black')
            axs.set_xlabel(f'{model.model.exog_names[1]}')
            axs.set_ylabel("Résidus studentisés")

        elif num_exog_vars == 2:
            for i in range(num_exog_vars):
                local = lowess(studentized_residuals, X[:, i])
                axs[i].scatter(X[:, i], studentized_residuals, edgecolors='k', alpha=0.6)
                axs[i].plot(local[:, 0], local[:, 1], color='black')
                axs[i].set_xlabel(f'{model.model.exog_names[i+1]}')
                axs[i].set_ylabel("Résidus studentisés")

        else:
            for i in range(num_rows):
                for j in range(num_cols):
                    idx = i * num_cols + j
                    if idx < num_exog_vars:
                        local = lowess(studentized_residuals, X[:, idx])
                        axs[i, j].scatter(X[:, idx], studentized_residuals, edgecolors='k', alpha=0.6)
                        axs[i, j].plot(local[:, 0], local[:, 1], color='black')
                        axs[i, j].set_xlabel(f'{model.model.exog_names[idx+1]}')
                        axs[i, j].set_ylabel("Résidus studentisés", fontsize=8)
                    else:
                        axs[i, j].axis('off')
        return fig
    
    @render.plot
    def acfplot():
        model = reg()['OLS']

        fig, ax = plt.subplots()
        plot_acf(model.resid, ax=ax)
        plt.title("Autocorrelation des résidus")
        plt.xlabel("Lag")
        plt.ylabel("ACF")

        return fig

    @render_plotly
    def aberrantplot():
        model = reg()['OLS']
        n = len(data)
        x = np.linspace(1, n, n)
        y = np.repeat(2, n)
        studentized_residuals = model.get_influence().resid_studentized_external
        local = lowess(studentized_residuals, x)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=studentized_residuals, mode='markers', marker=dict(color='white', line=dict(color='black', width=1)), showlegend=False, name=''))
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', line=dict(color='red', dash='dash'), showlegend=False, name=''))
        fig.add_trace(go.Scatter(x=x, y=-y, mode='lines', line=dict(color='red', dash='dash'), showlegend=False, name=''))
        fig.add_trace(go.Scatter(x=local[:, 0], y=local[:, 1], mode='lines', line=dict(color='black'), showlegend=False, name=''))
        fig.update_layout(
                        xaxis_title="Individu",
                        yaxis_title="Résidu studentisé par VC",
                        template="plotly_white")

        return fig
    
    @render_plotly
    def levierplot():
        model = reg()['OLS']
        n = len(data)
        x = np.linspace(1,n,n)
        hii = OLSInfluence(model).hat_diag_factor
        yhw = np.repeat(2*len(model.model.exog_names)/n,n)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=hii,showlegend=False, name=''))
        fig.add_trace(go.Scatter(x=x, y=yhw, mode='lines', line=dict(color='red', dash='dash'), name='Hoaglin-Welsh'))
        fig.update_layout(
                        xaxis_title="Individu",
                        yaxis_title="hii = poids obs i sur sa propre estimation",
                        template="plotly_white",
                        legend=dict(
                          orientation="h",
                          x=0.5,
                          y=1.1,
                          xanchor='center',
                          yanchor='middle'
                      ))

        return fig
    
    @render_plotly
    def cookplot():
        model = reg()['OLS']
        n = len(data)
        x = np.linspace(1,n,n)
        deg_freedom1 = len(model.model.exog_names)
        deg_freedom2 = n - deg_freedom1
        quantile = f.ppf(0.1, deg_freedom1, deg_freedom2)
        quantile_b = f.ppf(0.5, deg_freedom1, deg_freedom2)
        yok = np.repeat(quantile, n)
        yok_b = np.repeat(quantile_b, n)
        cd, _ = OLSInfluence(model).cooks_distance 

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=cd,showlegend=False, name=''))
        fig.add_trace(go.Scatter(x=x, y=yok, mode='lines', line=dict(color='red', dash='dash'), name='seuil 0.1'))
        fig.add_trace(go.Scatter(x=x, y=yok_b, mode='lines', line=dict(color='red', dash='dot'), name='seuil 0.5'))
        fig.update_layout(
                        xaxis_title="Individu",
                        yaxis_title="distance de Cook",
                        template="plotly_white",
                        legend=dict(
                          orientation="h",
                          x=0.5,
                          y=1.1,
                          xanchor='center',
                          yanchor='middle'
                      ))

        return fig
    
    @render_plotly
    def dffitsplot():
        model = reg()['OLS']
        n = len(data)
        x = np.linspace(1,n,n)
        obs, dffits_seuil = OLSInfluence(model).dffits
        yok = np.repeat(dffits_seuil,n)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=abs(obs),showlegend=False, name=''))
        fig.add_trace(go.Scatter(x=x, y=yok, mode='lines', line=dict(color='red', dash='dash'),showlegend=False, name=''))
        fig.update_layout(
                        xaxis_title="Individu",
                        yaxis_title="Écart de Welsh-Kuh",
                        template="plotly_white")

        return fig
    
    @render.text
    def test_linearity():
        model = reg()['OLS']
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
        
    
    @render.text
    def test_normal():
        model = reg()['OLS']
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
    
    @render.text
    def test_homo():
        model = reg()['OLS']
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
    
    @render.text
    def test_auto():
        model = reg()['OLS']
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

app = App(app_ui, server)
app.run()