import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import f
import math
from statsmodels.stats.outliers_influence import OLSInfluence
from statsmodels.nonparametric.smoothers_lowess import lowess

from statsmodels.graphics.tsaplots import plot_acf
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


def plot_reg(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        X = model.model.exog[:, 1:]
        if X.shape[1] == 1:
            nom_x = model.model.exog_names[1]
            df = pd.DataFrame({
                nom_x: X.flatten(),
                model.model.endog_names: model.model.endog
            })

            fig = px.scatter(df, x=nom_x, 
                            y=model.model.endog_names, trendline="ols")
            fig.update_traces(marker=dict(color='white', line=dict(color='black', width=1)))
            fig.update_traces(line=dict(color='black'))

            pred_ci = model.get_prediction().summary_frame(alpha=0.05)

            fig.add_trace(go.Scatter(x=np.concatenate([df[nom_x], df[nom_x][::-1]]),
                            y=np.concatenate([pred_ci['mean_ci_upper'], pred_ci['mean_ci_lower'][::-1]]),
                            fillcolor="red",
                            name='IC régression à 95%',
                            line=dict(color='red',dash='dash'),
                            showlegend=True))
            
            fig.add_trace(go.Scatter(x=np.concatenate([df[nom_x], df[nom_x][::-1]]),
                            y=np.concatenate([pred_ci['obs_ci_upper'], pred_ci['obs_ci_lower'][::-1]]),
                            fillcolor="#3854A6",
                            name='IC prédiction à 95%',
                            line=dict(color="#3854A6",dash='dash'),
                            showlegend=True))

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
    

def plot_log(dict_reg, bins):
    if any(value in dict_reg.keys() for value in ['Logist', 'Poisson', 'Bin_neg']):
        model = list(dict_reg.values())[0]
        X = model.model.exog[:, 1:]
        nom_x = model.model.exog_names[1]
        df = pd.DataFrame({
            nom_x: X.flatten(),
            model.model.endog_names: model.model.endog
        })

        df = df.sort_values(by=nom_x)
        nb_bins = bins
        bin_edges = np.linspace(df[nom_x].min(), df[nom_x].max(), nb_bins + 1)
        bins = pd.cut(df[nom_x], bins=bin_edges, labels=False, include_lowest=True)
        df['bin'] = bins

        # Calculer la moyenne de model.model.endog_names pour chaque bin
        means = df.groupby('bin')[model.model.endog_names].mean()

        data_emp = pd.DataFrame({
            'borne_inf': [df[df['bin'] == i][nom_x].min() for i in range(0, nb_bins)],
            'borne_sup': [df[df['bin'] == i][nom_x].max() for i in range(0, nb_bins)],
            'proportion': means.values
        })

        fig = px.scatter(df, x=nom_x, y=model.model.endog_names)

        pred_ci = model.get_prediction().summary_frame(alpha=0.05)
        pred_ci.columns = ['predicted', 'se', 'ci_lower', 'ci_upper']

        fig.add_traces(go.Scatter(x=df[nom_x], y=pred_ci.predicted, name='Probabilité prédite'))
        
        fig.add_trace(go.Scatter(x=np.concatenate([df[nom_x], df[nom_x][::-1]]),
                        y=np.concatenate([pred_ci.ci_upper, pred_ci.ci_lower[::-1]]),
                        fill='toself', opacity=0.3,
                        fillcolor="#3854A6",
                        name="IC régression 95%",
                        line=dict(dash='dash'),
                        showlegend=True))

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
    
def plot_linear(dict_reg): 
    if "OLS" in dict_reg: 
        model = dict_reg['OLS']
        residuals = model.resid
        fitted_values = model.fittedvalues
        local = lowess(residuals, fitted_values)

        fig, ax = plt.subplots()
        ax.scatter(fitted_values, residuals, edgecolors='k', alpha=0.6)
        ax.plot(local[:,0], local[:,1], color = 'black')
        ax.axhline(y=0, color='r', linestyle='--')
        ax.set(xlabel='Values ajustées / Fitted Values', 
                ylabel='Résidus / Residuals')
        
        return fig
    

def plot_normal(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        studentized_residuals = model.get_influence().resid_studentized_external
        
        fig = plt.figure()
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1])
        # QQ plot des résidus standardisés
        ax0 = plt.subplot(gs[0, 0])
        stats.probplot(studentized_residuals, dist="norm", plot=ax0)
        ax0.set_title('Normal Q-Q Plot des résidus studentisés')
        ax0.set(xlabel='Quantiles théoriques', 
                ylabel='Quantiles observés')
        # Histogramme des résidus standardisés avec estimation de densité
        ax1 = plt.subplot(gs[0, 1])
        sns.histplot(studentized_residuals, kde=True, ax=ax1, bins=30, edgecolor='k', alpha=0.6)
        ax1.set(xlabel='Résidus studentisés / Studentized Residuals par VC', 
                ylabel='Fréquence')
        
        return fig

def plot_homo_1(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        studentized_residuals = model.get_influence().resid_studentized_external
        sqrt_absolute_residuals = np.sqrt(np.abs(studentized_residuals))
        local = lowess(sqrt_absolute_residuals, model.fittedvalues)

        fig, ax = plt.subplots()
        ax.scatter(model.fittedvalues, sqrt_absolute_residuals, edgecolors='k', alpha=0.6)
        ax.plot(local[:,0], local[:,1], color = 'black')
        ax.set(xlabel="Valeur ajustée",
                ylabel="Racine carrée des résidus studentisés")

        return fig

def plot_homo_2(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
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
            axs.set(xlabel=f'{model.model.exog_names[1]}',
                    ylabel="Résidus studentisés")

        elif num_exog_vars == 2:
            for i in range(num_exog_vars):
                local = lowess(studentized_residuals, X[:, i])
                axs[i].scatter(X[:, i], studentized_residuals, edgecolors='k', alpha=0.6)
                axs[i].plot(local[:, 0], local[:, 1], color='black')
                axs[i].set(xlabel=f'{model.model.exog_names[i+1]}',
                            ylabel="Résidus studentisés")

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


def plot_auto(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']

        fig, ax = plt.subplots()
        plot_acf(model.resid, ax=ax)
        plt.title("Autocorrelation des résidus")
        plt.xlabel("Lag")
        plt.ylabel("ACF")

        return fig

def plot_aberrant(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        n = model.model.exog.shape[0]
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

def plot_levier(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        n = model.model.exog.shape[0]
        x = np.linspace(1,n,n)
        hii = OLSInfluence(model).hat_diag_factor
        yhw = np.repeat(2*len(model.model.exog_names)/n,n)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=x, y=hii,showlegend=False, name=''))
        fig.add_trace(go.Scatter(x=x, y=yhw, mode='lines', line=dict(color='red', dash='dash'), name='Hoaglin-Welsh'))
        fig.update_layout(
                        xaxis_title="Individu",
                        yaxis_title="Levier",
                        template="plotly_white",
                        legend=dict(
                        orientation="h",
                        x=0.5,
                        y=1.1,
                        xanchor='center',
                        yanchor='middle'
                    ))

        return fig

def plot_cook(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        n = model.model.exog.shape[0]
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

def plot_dffits(dict_reg):
    if "OLS" in dict_reg:
        model = dict_reg['OLS']
        n = model.model.exog.shape[0]
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
    


def plot_corr(df, nom_vars):
    correlations = df.corr()
    correlations['nom'] = correlations.index
    X = pd.melt(correlations, id_vars='nom', var_name='Variable', value_name='Cor')
    X['nom'] = pd.Categorical(X['nom'], categories=nom_vars[::-1], ordered=True)
    X['Variable'] = pd.Categorical(X['Variable'], categories=nom_vars[::-1], ordered=True)

    fig = px.imshow(X.pivot(index='Variable', columns='nom', values='Cor'),
            color_continuous_scale=[[0,"#636EFA"], [0.5, 'white'], [1, '#F24B4B']],
            labels=dict(x='Variable', y='nom'))
    
    for i, row in enumerate(X.pivot(index='Variable', columns='nom', values='Cor').values):
        for j, val in enumerate(row):
            fig.add_annotation(text=f"{val:.3f}", x=list(nom_vars[::-1])[i], y=list(nom_vars[::-1])[j], showarrow=False)
    
    fig.update_layout(xaxis_title="",yaxis_title="")
    
    return fig

def plot_scatter(df, x, y, color):
    return px.scatter(
        df,
        x=x,
        y=y,
        color=None if color == "Aucune" else color,
        template="plotly_white",
    )

def plot_histogram(df, x, bins, color):
    fig = px.histogram(
        df,
        x=x,
        nbins=bins,
        color= None if color == "Aucune" else color,
        template="plotly_white",
        histnorm='percent'
    )
    fig.update_traces(marker_line_color="black", marker_line_width=2)
    if color == "Aucune":
        fig.update_traces(marker_color="#F24B4B")

    return fig

def plot_violin(df, x, y=None, color=None):
    return px.violin(
        df,
        x=x,
        y=y,
        color=None if color == "Aucune" else color,
        box=True,
        points='all',
        template="plotly_white"
    )