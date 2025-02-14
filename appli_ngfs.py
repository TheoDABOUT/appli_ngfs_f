############## MODELISATION DE LA PD FORWORD-LOOKING AVEC LES SCENARIOS NGFS ##############

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from scipy.stats import pearsonr
from scipy.stats import spearmanr
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from statsmodels.tools.tools import add_constant
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.stats.diagnostic as diag
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import jarque_bera
from scipy.stats import shapiro
from statsmodels.stats.diagnostic import linear_harvey_collier
from statsmodels.stats.diagnostic import breaks_cusumolsresid
import matplotlib.pyplot as plt
import itertools
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import statsmodels.tsa.stattools as smt
import statsmodels.graphics.tsaplots as tsaplots
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy import stats
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from statsmodels.tsa.api import AutoReg



##########################################################################################################################
###################################################### TRAITEMENT DES DATASETS ###########################################
##########################################################################################################################

# Charger les datasets
df = pd.read_excel("Dataset PD.xlsx", sheet_name="pass")
futur = pd.read_excel("forward_looking_ngfs.xlsx")

# Préparer les données pour le graphique "passé"
def quarter_to_date(quarter):
    q, year = quarter.split()
    quarter_map = {"Q1": "01", "Q2": "04", "Q3": "07", "Q4": "10"}
    return pd.to_datetime(f"{year}-{quarter_map[q]}-01")

df['Date'] = df['Date'].apply(quarter_to_date)
df.set_index('Date', inplace=True)
df['PD'] = np.log(df['PD'] / (1 - df['PD']))




# Normalisation des colonnes
columns = [col for col in df.columns ]
columns2 = [col for col in df.columns if col not in ["PD"]]



##########################################################################################################################
###################################################### ANALYSE DES STAT & SEAS #############################################
##########################################################################################################################

df_brut = pd.DataFrame(df)

def differenciate_series(y, max_diff=10):
    diff_count = 0
    while diff_count < max_diff:
        # Appliquer la différenciation et supprimer les valeurs manquantes
        y_diff = y.diff().dropna()

        # Effectuer le test ADF sur la série différenciée
        adf_test = adfuller(y_diff, regression='c')
        st.write(f"\n📊 **Résultats du Test ADF après {diff_count + 1} différenciation(s)**")
        st.write(f"📌 Statistique ADF : {adf_test[0]:.4f}")
        st.write(f"📌 p-value : {adf_test[1]:.4f}")
        
        # Vérifier la stationnarité avec le test ADF sur la série différenciée
        if adf_test[1] < 0.05:
            st.write(f"✅ Série stationnaire après {diff_count + 1} différenciation(s).")
            return y_diff  # Retourner la série stationnarisée après différenciation
        
        # Si la série n'est pas stationnaire, continuer la différenciation
        diff_count += 1
        y = y_diff  # Continuer avec la série différenciée pour l'itération suivante
    
    # Si après 10 différenciations la série n'est toujours pas stationnaire
    st.write("❌ Série non stationnaire après 10 différenciations.")
    return y_diff  # Retourner la dernière version différenciée

def test_adf_models(df, column_name):
    if column_name in df.columns:
        y = df[column_name].dropna()
    else:
        st.error(f"❌ Erreur : La colonne '{column_name}' est introuvable dans le DataFrame.")
        return
    
    trend = pd.Series(range(1, len(y) + 1), index=y.index)
    y_lagged = y.shift(1)
    X_ct = pd.DataFrame({"LAGGED": y_lagged, "CONST": 1, "TREND": trend}).iloc[1:]
    y_ct = y.iloc[1:]
    
    if len(X_ct) == len(y_ct):
        model_ct = sm.OLS(y_ct, X_ct).fit()
        st.write("\n📊 **Résultats de la régression OLS (avec constante et tendance)**")
        st.write(model_ct.summary())
        
        if model_ct.pvalues['TREND'] > 0.05:
            st.write("\n🔍 La tendance n'est pas significative, passage au modèle 2 (avec constante).")
            test_adf_models_with_constant(df, column_name)
            return
        
        adf_result_ct = adfuller(y, regression='ct')
        st.write("\n📊 **Résultats du Test ADF en Niveau (avec constante et tendance)**")
        st.write(f"📌 Statistique ADF : {adf_result_ct[0]:.4f}")
        st.write(f"📌 p-value : {adf_result_ct[1]:.4f}")
        
        if adf_result_ct[1] < 0.05:
            st.write("✅ Le processus est un TS.  Il convient de le stationnariser en retranchant la tendance de la série Y par la méthode des MCO")
            X = sm.add_constant(np.arange(1, len(y) + 1))
            model = sm.OLS(y, X).fit()
            trend = model.predict(X)
            Y_stationary = y - trend
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(y.index, y, label='Série d\'origine', color='blue')
            ax1.plot(y.index, trend, label='Tendance estimée', linestyle='--', color='orange')
            ax1.set_xlabel('Temps')
            ax1.set_ylabel('Série d\'origine / Tendance', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.legend(loc='upper left')
            
            ax2 = ax1.twinx()
            ax2.plot(y.index, Y_stationary, label='Série stationnarisée', color='red')
            ax2.set_ylabel('Série stationnarisée', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
            
            plt.title('Série d\'origine, Tendance Estimée et Série Stationnarisée')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.write("❌ Le processus est non-stationnaire (DS).")
            y_diff = differenciate_series(y)
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(y.index, y, label='Série en niveau', color='blue')
            ax1.set_xlabel('Temps')
            ax1.set_ylabel('Série en niveau', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.legend(loc='upper left')
            
            ax2 = ax1.twinx()
            ax2.plot(y_diff.index, y_diff, label='Série différenciée', color='orange', linestyle='--')
            ax2.set_ylabel('Série différenciée', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            
            plt.title('Série en Niveau et Série Différenciée (DS)')
            plt.tight_layout()
            st.pyplot(fig)

def test_adf_models_with_constant(df, column_name):
    if column_name in df.columns:
        y = df[column_name].dropna()
    else:
        st.error(f"❌ Erreur : La colonne '{column_name}' est introuvable dans le DataFrame.")
        return
    
    y_lagged = y.shift(1)
    X_c = pd.DataFrame({"LAGGED": y_lagged, "CONST": 1}).iloc[1:]
    y_c = y.iloc[1:]
    
    if len(X_c) == len(y_c):
        model_c = sm.OLS(y_c, X_c).fit()
        st.write("\n📊 **Résultats de la régression OLS (avec constante)**")
        st.write(model_c.summary())
        
        if model_c.pvalues['CONST'] > 0.05:
            st.write("\n🔍 La constante n'est pas significative, passage au modèle 1 (sans constante ni tendance).")
            test_adf_models_without_constant(df, column_name)
            return
        
        adf_result_c = adfuller(y, regression='c')
        st.write("\n📊 **Résultats du Test ADF en Niveau (avec constante)**")
        st.write(f"📌 Statistique ADF : {adf_result_c[0]:.4f}")
        st.write(f"📌 p-value : {adf_result_c[1]:.4f}")
        
        if adf_result_c[1] < 0.05:
            st.write("✅ Le processus est stationnaire.")

        else:
            st.write("❌ Le processus est non-stationnaire (DS).")
            y_diff = differenciate_series(y)
            y_diff = y_diff.dropna()
            
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(y.index, y, label='Série en niveau', color='blue')
            ax1.set_xlabel('Temps')
            ax1.set_ylabel('Série en niveau', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            ax1.legend(loc='upper left')
            
            ax2 = ax1.twinx()
            ax2.plot(y_diff.index, y_diff, label='Série différenciée', color='orange', linestyle='--')
            ax2.set_ylabel('Série différenciée', color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')
            ax2.legend(loc='upper right')
            
            plt.title('Série en Niveau et Série Différenciée (DS)')
            plt.tight_layout()
            st.pyplot(fig)


def test_adf_models_without_constant(df, column_name):
    """
    Test de stationnarité ADF sans constante sur une série temporelle avec Streamlit.
    """
    if column_name not in df.columns:
        st.error(f"❌ Erreur : La colonne '{column_name}' est introuvable dans le DataFrame.")
        return

    y = df[column_name].dropna()

    # 📌 Test ADF en niveau sans constante
    adf_result_n = adfuller(y, regression='n')

    st.write("\n📊 **Résultats du Test ADF en Niveau (sans constante)**")
    st.write(f"📌 Statistique ADF :** {adf_result_n[0]:.4f}")
    st.write(f"📌 p-value :** {adf_result_n[1]:.4f}")

    if adf_result_n[1] < 0.05:
        st.success("✅ Le processus en niveau (sans constante) est stationnaire.")
        return
    
    st.write("❌ Le processus en niveau est non-stationnaire DS.")

    # 📌 Appliquer la différenciation automatique
    y_diff = differenciate_series(y)

    # 📊 Visualisation de la série en niveau et différenciée
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(y.index, y, label='Série en niveau', color='blue')
    ax1.set_xlabel('Temps')
    ax1.set_ylabel('Série en niveau', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(y_diff.index, y_diff, label='Série différenciée', color='orange', linestyle='--')
    ax2.set_ylabel('Série différenciée', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    ax2.legend(loc='upper right')

    plt.title('Série en Niveau et Série Différenciée (DS)')
    plt.tight_layout()
    
    st.pyplot(fig)

def differenciate_series(y, max_diff=10):
    diff_count = 0
    current_y = y.copy()
    while diff_count < max_diff and len(current_y) > 1:  # Empêcher les séries vides
        current_y = current_y.diff().dropna()
        diff_count += 1
        if len(current_y) < 1:  # Éviter les séries vides après diff
            break
        adf_test = adfuller(current_y, regression='c')
        if adf_test[1] < 0.05:
            break
    return current_y if not current_y.empty else y  # Retourner l'original si vide

def stationarize_series(y):
    y = y.dropna()
    if len(y) < 2:
        return y
    
    # Modèle avec tendance et constante
    y_lagged = y.shift(1).dropna()
    trend = pd.Series(range(1, len(y_lagged)+1), index=y_lagged.index)
    X_ct = pd.DataFrame({'LAGGED': y_lagged, 'CONST': 1, 'TREND': trend})
    y_ct = y.iloc[1:]
    
    if len(X_ct) == len(y_ct):
        try:
            model_ct = sm.OLS(y_ct, X_ct).fit()
            trend_p = model_ct.pvalues.get('TREND', 1)
            if trend_p <= 0.05:
                adf_result_ct = adfuller(y, regression='ct')
                if adf_result_ct[1] < 0.05:
                    X_trend = sm.add_constant(np.arange(1, len(y)+1))
                    model_trend = sm.OLS(y, X_trend).fit()
                    trend_estimated = model_trend.predict(X_trend)
                    return y - trend_estimated
                else:
                    return differenciate_series(y)
            else:
                # Modèle avec constante
                X_c = pd.DataFrame({'LAGGED': y_lagged, 'CONST': 1})
                y_c = y.iloc[1:]
                model_c = sm.OLS(y_c, X_c).fit()
                const_p = model_c.pvalues.get('CONST', 1)
                if const_p <= 0.05:
                    adf_result_c = adfuller(y, regression='c')
                    if adf_result_c[1] < 0.05:
                        return y
                    else:
                        return differenciate_series(y)
                else:
                    # Modèle sans constante
                    adf_result_n = adfuller(y, regression='n')
                    if adf_result_n[1] < 0.05:
                        return y
                    else:
                        return differenciate_series(y)
        except Exception as e:
            print(f"Erreur lors de la stationnarisation : {e}")
            return y  # Retourner l'original en cas d'erreur

def generer_dataframe_stationarise(df):
    df_stationarise = pd.DataFrame()
    for col in df.columns:
        y = df[col].dropna()
        if len(y) < 2:  # Ignorer les colonnes avec < 2 observations
            continue
        y_stationary = stationarize_series(y)
        if not y_stationary.empty:
            df_stationarise[col] = y_stationary
    return df_stationarise.dropna(how='all', axis=1)  # Supprimer colonnes vides

def test_seasonality(data, columns):
    """Effectue un test de saisonnalité pour chaque série à l'aide de la décomposition STL."""
    results = []

    for col in columns:
        st.write(f"Analyse de la saisonnalité pour la série: {col}")
        serie = data[col].dropna()

        # Décomposer la série temporelle
        decomposition = seasonal_decompose(serie, model='additive', period=12)  # Période de 12 pour mensuel
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()

        # Affiche les résultats de la décomposition
        st.write(" Composants de la décomposition saisonnière :")
        
        # Afficher les graphiques de chaque composant
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle(f'Décomposition Saisonnière pour {col}')
        
        # Composant tendance
        axes[0].plot(trend)
        axes[0].set_title('Composant Tendance')
        axes[0].set_xlabel('Temps')
        axes[0].set_ylabel('Valeurs')
        
        # Composant saisonnier
        axes[1].plot(seasonal)
        axes[1].set_title('Composant Saisonnière')
        axes[1].set_xlabel('Temps')
        axes[1].set_ylabel('Valeurs')
        
        # Composant résiduel
        axes[2].plot(residual)
        axes[2].set_title('Composant Résiduel')
        axes[2].set_xlabel('Temps')
        axes[2].set_ylabel('Valeurs')
        
        st.pyplot(fig)

        # ACF (Autocorrelation Function)
        st.write("#### Fonction d'autocorrélation (ACF) :")
        fig_acf = plt.figure(figsize=(10, 6))
        tsaplots.plot_acf(serie, lags=24, ax=fig_acf.gca())  # 24 lags par exemple
        st.pyplot(fig_acf)

        # Test de Kruskal-Wallis pour les différences entre les trimestres
        st.write("#### Test de Kruskal-Wallis :")
        serie_quarterly = serie.resample('Q').mean()  # Resample par trimestre
        quarterly_groups = [serie_quarterly[serie_quarterly.index.quarter == quarter] for quarter in range(1, 5)]

        # Effectuer le test de Kruskal-Wallis
        kruskal_stat, kruskal_p_value = stats.kruskal(*quarterly_groups)
        st.write(f"- Statistique de Kruskal-Wallis : {kruskal_stat}")
        st.write(f"- Valeur p : {kruskal_p_value}")
        
        is_seasonal = kruskal_p_value < 0.05

        if is_seasonal:
            st.write("La série présente une saisonnalité significative (on rejette H0).")
        else:
            st.write("La série ne présente pas de saisonnalité significative (on ne rejette pas H0).")

        # Stocker les résultats dans le tableau
        results.append({
            "Série": col,
            "Présence de saisonnalité": "Oui" if is_seasonal else "Non",
            "Valeur p (Kruskal-Wallis)": kruskal_p_value,
            "Moyenne saisonnière": seasonal.mean()
        })

    # Créer un DataFrame des résultats et l'afficher
    summary_df = pd.DataFrame(results)
    st.write("### Tableau récapitulatif de la saisonnalité :")
    st.dataframe(summary_df)

    return summary_df

#########################################################################################################################
###################################################### TRAITEMENT FUTUR ###########################################
##########################################################################################################################

df_stationnaire = generer_dataframe_stationarise(df)
df_stationnaire = df_stationnaire.dropna()


# Préparer les données pour le graphique "futur"
df_futur = pd.melt(futur, id_vars=['Model', 'Scenario', 'Region', 'Variable'], var_name='Year', value_name='Value')
df_futur = df_futur.pivot_table(index=['Model', 'Scenario', 'Region', 'Year'], columns='Variable', values='Value', aggfunc='first').reset_index()
df_futur = df_futur[df_futur['Year'] != 'Unit']
df_futur['Model_Scenario'] = df_futur['Model'] + ' ' + df_futur['Scenario']
df_futur.rename(columns={'Year': 'Date'}, inplace=True)
df_futur['Date'] = pd.to_datetime(df_futur['Date'], format='%Y')
df_futur.set_index('Date', inplace=True)
df_futur = df_futur[['Model_Scenario'] + columns2]
# Données futur réel
df_2014 = df_stationnaire[df_stationnaire.index == "2014-10-01"]
gdp_2014_q4 = df_2014['Gross Domestic Product (GDP)'].values[0]
df_futur['Gross Domestic Product (GDP)'] = gdp_2014_q4 * (1 + df_futur['Gross Domestic Product (GDP)'] / 100)
exp_2014_q4 = df_2014['Exports (goods and services)'].values[0]
df_futur['Exports (goods and services)'] = exp_2014_q4 * (1 + df_futur['Exports (goods and services)'] / 100)
gov_2014_q4 = df_2014['Gov. consumption'].values[0]
df_futur['Gov. consumption'] = gov_2014_q4 * (1 + df_futur['Gov. consumption'] / 100)
invest_2014_q4 = df_2014['Investment (private sector)'].values[0]
df_futur['Investment (private sector)'] = invest_2014_q4 * (1 + df_futur['Investment (private sector)'] / 100)
imp_2014_q4 = df_2014['Imports (goods and services)'].values[0]
df_futur['Imports (goods and services)'] = imp_2014_q4 * (1 + df_futur['Imports (goods and services)'] / 100)
rpdi_2014_q4 = df_2014['Real personal disposable income'].values[0]
df_futur['Real personal disposable income'] = rpdi_2014_q4 * (1 + df_futur['Real personal disposable income'] / 100)
hp_2014_q4 = df_2014['House prices (residential)'].values[0]
df_futur['House prices (residential)'] = hp_2014_q4 * (1 + df_futur['House prices (residential)'] / 100)
##############
df_2017 = df_stationnaire[df_stationnaire.index == "2017-10-01"]
er_2017_q4 = df_2017['Effective exchange rate'].values[0]
df_futur['Effective exchange rate'] = er_2017_q4 * (1 + df_futur['Effective exchange rate'] / 100)
##############
df_2021 = df_stationnaire[df_stationnaire.index == "2021-10-01"]
unem_2021_q4 = df_2021['Unemployment rate ; %'].values[0]
df_futur['Effective exchange rate'] = unem_2021_q4  + df_futur['Unemployment rate ; %'] 
cbir_2021_q4 = df_2021['Central bank Intervention rate (policy interest rate) ; %'].values[0]
df_futur['Central bank Intervention rate (policy interest rate) ; %'] = cbir_2021_q4  + df_futur['Central bank Intervention rate (policy interest rate) ; %'] 
infl_2021_q4 = df_2021['Inflation rate ; %'].values[0]
df_futur['Inflation rate ; %'] = infl_2021_q4  + df_futur['Inflation rate ; %'] 
# 
op_2021_q4 = df_2021['Oil price ; US$ per barrel'].values[0]
df_futur['Oil price ; US$ per barrel'] = op_2021_q4 * (1 + df_futur['Oil price ; US$ per barrel'] / 100)
er_2021_q4 = df_2021['Exchange rate FRA Franc; per US$'].values[0]
df_futur['Exchange rate FRA Franc; per US$'] = er_2021_q4 * (1 + df_futur['Exchange rate FRA Franc; per US$'] / 100)

result = pd.concat([df, df_futur])
result['Model_Scenario'] = result['Model_Scenario'].fillna('Data pass')
result = result[['Model_Scenario'] + [col for col in df.columns if col != 'PD']]

moyennes_2020 = df.loc["2020"].mean()
moyennes_2021 = df.loc["2021"].mean()


# Création des valeurs retardées pour df_futur (hors Model_Scenario)
scenarios = df_futur["Model_Scenario"].unique()
df_futur["PD"] = None
df_futur_retarde = df_futur.copy()
variables_macro = [col for col in df_futur.columns if col not in ["Model_Scenario"]]


for scenario in scenarios:
    mask = df_futur["Model_Scenario"] == scenario
    df_scenario = df_futur.loc[mask].copy()

    for column in variables_macro:
        df_scenario[f"{column}_T1"] = df_scenario[column].shift(1)
        df_scenario[f"{column}_T2"] = df_scenario[column].shift(2)

        # Remplacement des premières valeurs de chaque scénario
        df_scenario.at[df_scenario.index[0], f"{column}_T1"] = moyennes_2021[column]
        df_scenario.at[df_scenario.index[0], f"{column}_T2"] = moyennes_2020[column]
        if len(df_scenario) > 1:
            df_scenario.at[df_scenario.index[1], f"{column}_T2"] = moyennes_2021[column]

    # Affectation correcte des nouvelles colonnes
    for col in df_scenario.columns:
        df_futur_retarde.loc[mask, col] = df_scenario[col].values


#RL
df_futur_retarde2 = df_futur_retarde.copy()
df_futur_retarde4 = df_futur_retarde.copy()

#AR_X
df_futur_retarde1 = df_futur_retarde.copy()
df_futur_retarde3 = df_futur_retarde.copy()

# Stationnarise.

def generer_dataframe_stationarise_futur(df):
    series_list = []  # Liste pour stocker les séries transformées
    
    for col in df.columns:
        y = df[col].dropna()
        if len(y) < 2:  # Ignorer les colonnes avec < 2 observations
            continue
        y_stationary = stationarize_series(y)
        if not y_stationary.empty:
            series_list.append(y_stationary.rename(col))  # Renommer pour garder le nom d'origine
    
    # Concaténer toutes les séries en une seule opération
    df_stationarise = pd.concat(series_list, axis=1)
    
    return df_stationarise.dropna(how='all', axis=1)  # Supprimer colonnes vides


df_futur_retarde1 = df_futur_retarde1.pivot(columns="Model_Scenario", values=[col for col in df_futur_retarde1.columns if col != "Model_Scenario"])
df_futur_retarde1.columns = [f"{col[0]}_{col[1]}" for col in df_futur_retarde1.columns]

df_futur_retarde1 = df_futur_retarde1.apply(pd.to_numeric, errors='coerce')
df_futur_retarde1 = df_futur_retarde1.dropna(axis=1)  

df_futur_stationarise= generer_dataframe_stationarise_futur(df_futur_retarde1)
#df_futur_stationarise = df_futur_stationarise.dropna()

######################### Trimestrialise #########################


df_futur_stationarise = df_futur_stationarise.reset_index()
dates_trimestrielles = pd.date_range(start=df_futur_stationarise["Date"].min(), 
                                     end=df_futur_stationarise["Date"].max(), 
                                     freq='QE-DEC')

dates_trimestrielles += pd.DateOffset(days=1)
df_trimestriel = pd.DataFrame({"Date": dates_trimestrielles})

# Appliquer l'interpolation pour chaque colonne sauf la date
for col in df_futur_stationarise.columns[1:]:
    values = df_futur_stationarise[col].values
    interpolated_values = []

    for i in range(len(values) - 1):
        T_pi_4 = values[i + 1]  # Valeur de fin d'année suivante
        T_pi_4_prev = values[i]  # Valeur de fin d'année actuelle
        
        for j in range(1, 5):  # Trimestres 1 à 4
            value = (T_pi_4 - T_pi_4_prev) * (j / 4) + T_pi_4_prev
            interpolated_values.append(value)

    df_trimestriel[col] = interpolated_values

df_trimestriel.set_index('Date', inplace=True)





df_futur_modelisation = df_futur_retarde.copy()
df_futur_modelisation = df_futur_modelisation.drop(columns=["PD", "PD_T1", "PD_T2"])


##########################################################################################################################
###################################################### APPLICATION #######################################################
##########################################################################################################################



# Configurer le mode large
st.set_page_config(layout="wide")

# Appliquer un style global
st.markdown(
    """
    <style>
    .css-1d391kg {  
        max-width: 90%;  
        margin: auto;  
    }
    .css-10trblm {  
        text-align: center;  
    }
    </style>
    """, unsafe_allow_html=True
)

# Titre principal
st.title("Dashboard for modeling risk parameters ('Forward-looking') based on NGFS climate scenarios.")
st.write("You will find the main steps for modeling")

with st.sidebar:
    st.image("NEONRISK_LOGO.png", 
             caption="NEON RISK", width=150)
    st.image("NGFS_LOGO.jpg", 
             caption="NGFS", width=150)


with st.expander("Data processing"):

    # Section d'analyse de la stationnarité
    
    st.subheader("Stationarity analysis")

    selected_columns = st.multiselect("Sélectionnez les colonnes pour l'analyse de la stationnarité :   ", options=df.columns, default=df.columns[:3])

    if st.button("Analyze stationarity "):
        for column_name in selected_columns:
            st.write(f"\n🔍 **Analyse de la colonne : {column_name}**")
            test_adf_models(df, column_name)
        
    # Section d'analyse de la saisonnalité
    st.subheader("Seasonality analysis")
    
    columns_seasonality = st.multiselect(
        "Sélectionnez les colonnes pour l'analyse de saisonnalité :  ", options=df.columns, default=df.columns[:3]
    )
    if st.button("Analyze seasonality "):
        st.write("## Résultats de l'analyse de saisonnalité :")
        test_seasonality(df, columns_seasonality)


    # Utiliser le DataFrame stationnarisé

    st.header("Number of LAGs")

    choice_lag = st.number_input("Number of LAGs", 1, 5, value=2)
    lag_steps = 4  # Intervalle de décalage


    df_stationnaire_modelisation = df_stationnaire.copy()
    df_modelisation = df.copy()




    columns_sans_pd = [col for col in df.columns if col != "PD"]
    for column in columns_sans_pd:
        for lag in range(1, choice_lag + 1):
            shift_value = lag * lag_steps
            df_stationnaire_modelisation[f"{column}_T{lag}"] = df_stationnaire_modelisation[column].shift(shift_value)
            df_modelisation[f"{column}_T{lag}"] = df_modelisation[column].shift(shift_value)

    df_stationnaire_modelisation = df_stationnaire_modelisation.dropna()
    df_modelisation = df_modelisation.dropna()
    df = df.dropna()


        

    st.header("Data separation (Train/Test)")

    df_stationnaire_modelisation.index = pd.to_datetime(df_stationnaire_modelisation.index)

    # Génération des dates avec un pas de 3 mois
    min_date = df_stationnaire_modelisation.index.min()
    max_date = df_stationnaire_modelisation.index.max()
    date_range = pd.date_range(start=min_date, end=max_date, freq='3MS').tolist()  # Convertir en liste

    # Slider avec des dates
    selection_date = st.select_slider(
        "Sélection de la date de séparation :",
        options=date_range,
        value=pd.Timestamp('2018-01-01')
    )

    # Séparation des données
    train_stationnaire = df_stationnaire_modelisation[df_stationnaire_modelisation.index < selection_date]
    test_stationnaire = df_stationnaire_modelisation[df_stationnaire_modelisation.index >= selection_date]



    # Affichage des graphiques
    st.header("Correlation analysis")
    st.write("**Paramètres d'analyse :**")
    
    vif_columns = [col for col in train_stationnaire.columns if col not in ["PD"]]

    col1, col2, col3, col4, col5 = st.columns([0.5, 1, 0.5, 2, 2])

    with col1:
        # Filtre P-value
        filter_p_value = st.checkbox("P-value < 0.05 ", value=True)
    
    with col2:
        # Seuil de corrélation
        correlation_threshold = st.slider(
            "Corrélation min absolue ",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1
        )

    with col3:
        test_choice = st.radio(
            "Méthode de corrélation :",  
            ("pearson", "spearman"), 
            index=0 
        )
    
    # Sélection des colonnes
    with col4:
        vif_threshold = st.slider(
            "Seuil maximal de VIF",
            min_value=1.0,
            max_value=10.0,
            value=5.0,
            step=0.1
        )
    with col5:
        st.write(" ")

    col1, col2, col3 = st.columns([2, 2, 2])

    with col1:
        st.subheader("Correlations      -->")
        variables_explicatives = [col for col in train_stationnaire.columns if col != "PD"]
        results = {}

        # Calcul de la corrélation et de la p-value pour chaque variable

        for var in variables_explicatives:
            if test_choice == "pearson":
                corr, p_value = stats.pearsonr(train_stationnaire["PD"], train_stationnaire[var])
            else:  # Spearman
                corr, p_value = stats.spearmanr(train_stationnaire["PD"], train_stationnaire[var])

            results[var] = {"Corrélation": corr, "p-value": p_value}

        # Conversion en DataFrame pour affichage
        df_results = pd.DataFrame(results).T

        if filter_p_value:
            df_results = df_results[(df_results["p-value"] < 0.05) & 
                                    ((df_results["Corrélation"] > correlation_threshold) | 
                                    (df_results["Corrélation"] < -correlation_threshold))]
        else:
            df_results = df_results[(df_results["Corrélation"].abs() > correlation_threshold)]
        
        variables_retenues_correlation = df_results.index.tolist()

        st.dataframe(df_results)

    with col2:
        st.subheader("VIF  -->")
        def compute_vif(df_selected, variables, threshold):
            """
            Calcule le VIF pour un DataFrame donné et filtre les variables en fonction d'un seuil.
            """
            vif_filtered = variables.copy()  # Copie des variables initiales
            vif_data = pd.DataFrame()

            while True:
                # Calcul du VIF
                X_vif = df_selected[vif_filtered].dropna()  # Suppression des NaN
                X_const_vif = add_constant(X_vif)  # Ajout de la constante

                vif_df = pd.DataFrame()
                vif_df["Variable"] = X_const_vif.columns
                vif_df["VIF"] = [variance_inflation_factor(X_const_vif.values, i) for i in range(X_const_vif.shape[1])]

                # Suppression de la constante
                vif_df = vif_df[vif_df["Variable"] != "const"]

                # Trier par VIF décroissant
                vif_df = vif_df.sort_values(by='VIF', ascending=False)

                # Vérifier si toutes les variables sont sous le seuil
                if vif_df["VIF"].max() <= threshold:
                    vif_data = vif_df  # Sauvegarde du dernier état du VIF
                    break  # Arrêter si toutes les variables respectent le seuil

                # Supprimer la variable avec le VIF le plus élevé
                max_vif_variable = vif_df.iloc[0]["Variable"]
                vif_filtered.remove(max_vif_variable)

            return vif_data, vif_filtered

        # Si les variables de VIF sont disponibles
        vif_filtered_columns = []  # Initialisation vide par défaut

        if variables_retenues_correlation:
            st.write("Variance Inflation Factor (VIF) des variables retenues après corrélation")

            # Calcul du VIF après élimination des variables trop corrélées
            vif_data, vif_filtered_columns = compute_vif(df_stationnaire_modelisation, variables_retenues_correlation, vif_threshold)

            # Affichage du tableau final des VIF
            st.write(vif_data)


    with col3:
        st.subheader("Lasso regression")
        X_train = train_stationnaire[vif_filtered_columns]
        y_train = train_stationnaire['PD']
        X_test = test_stationnaire[vif_filtered_columns]
        y_test = test_stationnaire['PD']

        # Normalisation des données
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Régression Lasso optimisée par validation croisée
        st.write("Régression Lasso optimisée (Validation croisée) :")
        lasso_cv = LassoCV(alphas=np.logspace(-4, 0, 50), cv=5, max_iter=10000)
        lasso_cv.fit(X_train_scaled, y_train)
        y_pred_cv = lasso_cv.predict(X_test_scaled)

        mse_cv = mean_squared_error(y_test, y_pred_cv)
        r2_cv = r2_score(y_test, y_pred_cv)
        n_non_zero_cv = sum(lasso_cv.coef_ != 0)
        best_alpha = lasso_cv.alpha_

        st.write(f"Meilleur alpha par validation croisée : {best_alpha:.4f}")
        #st.write(f"MSE : {mse_cv:.2f}")
        st.write(f"R² : {r2_cv:.2f}")
        st.write("Nombre de coefficients non nuls :", n_non_zero_cv)

        # Affichage des coefficients du modèle optimisé
        st.write("Coefficients du modèle optimisé :")
        coef_df = pd.DataFrame({
            "Variable": X_train.columns,
            "Coefficient": lasso_cv.coef_
        })
        st.write(coef_df)


    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.write("Variables retenues :", variables_retenues_correlation)
    with col2:
        st.write("Variables retenues :", vif_filtered_columns)
    with col3:
        selected_features = X_train.columns[lasso_cv.coef_ != 0]
        st.write("Variables retenues :", selected_features.tolist())


with st.expander("Data analysis"):

    col1, col2 = st.columns([4 , 2])
    with col1:
        st.subheader("Pass")
    with col2:
        st.subheader("Futur")

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.write("Initial past database:")
        st.dataframe(df_modelisation)
    with col2:
        st.write("Stationary past database:")
        st.dataframe(df_stationnaire_modelisation)
    with col3:
        st.write("Forward-looking scenario database:")
        st.dataframe(df_futur_modelisation)

    #Normaisé pour les graphiques !
    df_norm=df_modelisation.copy()
    df_stationnaire_norm=df_stationnaire_modelisation.copy()
    df_futur_norm=df_futur_modelisation.copy()
    for column in df_norm.columns:
        df_norm[f"{column}_Norm"] = (df_norm[column] - df_norm[column].min()) / (df_norm[column].max() - df_norm[column].min())
    for column in df_stationnaire_norm.columns:
        df_stationnaire_norm[f"{column}_Norm"] = (df_stationnaire_norm[column] - df_stationnaire_norm[column].min()) / (df_stationnaire_norm[column].max() - df_stationnaire_norm[column].min())
    for column in [col for col in df_futur_norm.columns if col != "Model_Scenario"]:
        df_futur_norm[f"{column}_Norm"] = (df_futur_norm[column] - df_futur_norm[column].min()) / (df_futur_norm[column].max() - df_futur_norm[column].min())



    selected_columns = st.multiselect(
            "Choisissez les indicateurs à afficher :", 
            options=df_stationnaire_modelisation.columns, 
            default=selected_features
        )

    col1, col2, col3 = st.columns([2, 2, 2])

    
    with col1:
        if selected_columns:
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(df_norm.index, df_norm["PD_Norm"], marker='o', label='Taux defaut (Normalisé)')
            for column in selected_columns:
                ax.plot(df_norm.index, df_norm[f"{column}_Norm"], marker='o', label=f'{column} (Normalisé)')
            ax.set_title('Évolution normalisée des indicateurs économiques', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Valeur normalisée', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True)
            st.pyplot(fig, use_container_width=True)

    with col2:
        if selected_columns:
            fig, ax = plt.subplots(figsize=(16, 8))
            ax.plot(df_stationnaire_norm.index, df_stationnaire_norm["PD_Norm"], marker='o', label='Taux defaut (Normalisé)')
            for column in selected_columns:
                ax.plot(df_stationnaire_norm.index, df_stationnaire_norm[f"{column}_Norm"], marker='o', label=f'{column} (Normalisé)')
            ax.set_title('Évolution normalisée des indicateurs économiques', fontsize=14)
            ax.set_xlabel('Date', fontsize=12)
            ax.set_ylabel('Valeur normalisée', fontsize=12)
            ax.legend(fontsize=12)
            ax.grid(True)
            st.pyplot(fig, use_container_width=True)


    with col3:

        selected_model_scenarios = st.multiselect(
            "Choisissez les scénarios :", 
            options=df_futur_norm['Model_Scenario'].unique(),
            default=['NiGEM NGFS v1.24.2[REMIND-MAgPIE 3.3-4.8] Below 2?C']
        )

        selected_columns_futur = [col for col in selected_columns if col not in ["PD", "PD_T1","PD_T2"]]

        if selected_model_scenarios and selected_columns_futur:
            filtered_df = df_futur_norm[df_futur_norm['Model_Scenario'].isin(selected_model_scenarios)]
            fig, ax = plt.subplots(figsize=(16, 8))

            for column in selected_columns_futur:  # Iterate over selected columns
                sns.lineplot(
                    data=filtered_df, 
                    x=filtered_df.index, 
                    y=f"{column}_Norm", 
                    hue='Model_Scenario', 
                    marker='o', 
                    ax=ax, 
                    label=column  # Add column name as a label
                )

            ax.set_title(f"Évolution normalisée des variables: {', '.join(selected_columns_futur)}", fontsize=14)
            ax.set_xlabel("Année", fontsize=12)
            ax.set_ylabel("Valeur", fontsize=12)
            ax.legend(title='Variable / Model Scenario', fontsize=10)
            ax.grid(True)
            st.pyplot(fig, use_container_width=True)
            


st.header("Modeling & Prediction (ARX - RL)")
st.write("ARX (Times Series) and Linear Regression (Classic Method) models are interpretable models.")
    
with st.expander("Modeling"):

    # Créez deux colonnes côte à côte
    col1, col2 = st.columns([3, 3])  # Ajustez les proportions si nécessaire


    with col1:
        st.subheader("AR-X")

        choix_var_arx = st.radio(
        "Choix des variables 👇",
        ["Resultats Selection Variables", "All"],
        horizontal=True)

        if choix_var_arx == "Resultats Selection Variables":
            features = selected_features
        else:
            features = variables_explicatives

        X_train = train_stationnaire[features]
        y_train = train_stationnaire['PD']
        X_test = test_stationnaire[features]
        y_test = test_stationnaire['PD']

        results_test = []

        # Boucle pour générer toutes les combinaisons possibles (par exemple 2 features)
        for r in range(1, 3):  # Ajustez r pour choisir la taille des combinaisons
            for combination in itertools.combinations(features, r):
                for lags in range(1, 3):
                    # Ajouter une constante pour le modèle
                    X_train = add_constant(X_train)
                    X_test = add_constant(X_test)

                    # Sous-ensembles des données avec les features sélectionnées
                    X_train_subset = X_train[list(combination)]
                    X_test_subset = X_test[list(combination)]

                    start = y_test.index[0]
                    end = y_test.index[-1]

                    # Modèle
                    model_arx = AutoReg(endog=y_train, lags=lags, exog=X_train_subset).fit()
                    y_pred = model_arx.predict(start=start, end=end , exog_oos=X_test_subset)

                    # P-values des coefficients
                    pvalues = model_arx.pvalues
                    pvalues_greater_than_0_05 = (pvalues > 0.05).sum()
                    pvalues_less_than_or_equal_to_0_05 = (pvalues <= 0.05).sum()

                    # Tests statistiques
                    #bg_test = diag.acorr_breusch_godfrey(model_arx, nlags=4)
                    lb_test = acorr_ljungbox(model_arx.resid, lags=[10], return_df=True)
                    #bp_test = het_breuschpagan(model_arx.resid, X_train)
                    jb_test = jarque_bera(model_arx.resid)
                    shapiro_test = shapiro(model_arx.resid)

                    # Métriques
                    r2 = r2_score(y_test, y_pred)
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    mape = (abs((y_test - y_pred) / y_test).mean()) * 100

                    # Résultats
                    results_test.append({
                        'Combination': combination,
                        'LAGs': lags,
                        'R2': r2,
                        'MSE': mse,
                        'RMSE': rmse,
                        'MAE': mae,
                        'MAPE': mape,
                        'P-value > 0.05': pvalues_greater_than_0_05,
                        'P-value <= 0.05': pvalues_less_than_or_equal_to_0_05,
                        #'Breusch-Godfrey': bg_test[1],
                        'Ljung-Box': lb_test['lb_pvalue'].values[0],
                        #'Breusch-Pagan': bp_test[1],
                        'Jarque-Bera': jb_test[1],
                        'Shapiro-Wilk': shapiro_test[1]
                    })

        # Conversion des résultats en DataFrame
        results_test = pd.DataFrame(results_test)
        results_test = results_test.sort_values(by='R2', ascending=False)
        best_model = results_test.iloc[0]['Combination']
        best_lag = results_test.iloc[1]['LAGs']


        # Streamlit app
        st.subheader("Cross-validation")
        st.write("Résultats des métriques de performance pour différentes combinaisons de features :")

        # Afficher les résultats sous forme de table
        st.dataframe(results_test)



    with col2:
        st.subheader("Lineaire Regression")

        choix_var_rl = st.radio(
        "Choix des variables 👇",
        ["Resultats Selection Variables ", "All"],
        horizontal=True)

        if choix_var_rl == "Resultats Selection Variables":
            features = selected_features
        else:
            features = variables_explicatives

        X_train = train_stationnaire[features]  # Variables explicatives pour l'entraînement
        y_train = train_stationnaire['PD']  # Variable cible pour l'entraînement
        X_test = test_stationnaire[features]  # Variables explicatives pour le test
        y_test = test_stationnaire['PD']  # Variable cible pour le test

        results = []

        # Boucle pour générer toutes les combinaisons possibles (2 features ici)
        for r in range(2, 3):  # Ajustez r pour modifier le nombre de features
            for combination in itertools.combinations(features, r):
                # Sous-ensembles des données avec les features sélectionnées
                X_train_subset = X_train[list(combination)]
                X_test_subset = X_test[list(combination)]

                # Modèle de régression linéaire
                model_rl = LinearRegression()
                model_rl.fit(X_train_subset, y_train)

                # Prédictions
                y_pred = model_rl.predict(X_test_subset)

                # Calcul des métriques
                r2 = r2_score(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                mape = (abs((y_test - y_pred) / y_test).mean()) * 100

                # Résultats
                results.append({
                    'Combination': combination,
                    'R2': r2,
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'MAPE': mape
                })

        # Conversion des résultats en DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='R2', ascending=False)
        best_model_rl = results_df.iloc[0]['Combination']


        # Streamlit app
        st.subheader("Cross-validation")
        st.write("Résultats des métriques de performance pour différentes combinaisons de features :")

        # Afficher les résultats sous forme de table
        st.dataframe(results_df)  # Option dynamique avec filtrage et redimensionnement



    # Créez deux colonnes côte à côte
    col1, col2 = st.columns([3, 3])  # Ajustez les proportions si nécessaire


    with col1:

        selected_columns_arx = st.multiselect(
            "Choisissez les indicateurs à afficher :", 
            options=features, 
            default=best_model
        )

        choice_lag = st.number_input("Number of LAGs ", 1, 5, value=best_lag)

        # Séparer les variables explicatives (X) et la variable cible (y) pour l'ensemble d'entraînement
        X_train = train_stationnaire[selected_columns_arx]  # Variables explicatives pour l'entraînement
        y_train = train_stationnaire['PD']  # Variable cible pour l'entraînement
        # Variables explicatives (X) et cible (y) pour l'ensemble de test
        X_test = test_stationnaire[selected_columns_arx]  # Variables explicatives pour le test
        y_test = test_stationnaire['PD']  # Variable cible pour le test

        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        start = y_test.index[0]
        end = y_test.index[-1]
        

        # Modèle
        model_arx = AutoReg(endog=y_train, lags=choice_lag, exog=X_train).fit()
        y_pred = model_arx.predict(start=start, end=end , exog_oos=X_test)

        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)

        # Afficher l'équation du modèle
        coeffs = model_arx.params
        equation = "TD = " + " + ".join(f"{coeff:.4f}*{col}" for col, coeff in coeffs.items())
        st.write("**Équation du modèle :**")
        st.text(equation)

        st.write("**Summary du modèle :**")
        st.write(model_arx.summary())

        # Afficher les performances du modèle
        st.write("**Performances du modèle :**")
        st.write(f"Erreur quadratique moyenne (MSE): {mse:.2f}")
        st.write(f"R² (coefficient de détermination): {r2:.2f}")


        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))

        # Utiliser les dates comme index pour le graphique
        ax.plot(train_stationnaire.index, y_train, label="TD (Train)", color="blue", alpha=0.7)
        ax.plot(test_stationnaire.index, y_test, label="TD (Test)", color="orange", alpha=0.7)
        ax.plot(test_stationnaire.index, y_pred, label="TD (Prédictions)", color="green", linestyle="--")

        # Ajouter un titre et des labels
        ax.set_title("Comparaison des Taux de défauts : Train, Test et Prédictions", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("TD")
        ax.legend()

        # Améliorer l'affichage des dates
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)

    with col2:

        selected_columns_rl = st.multiselect(
            "Choisissez les indicateurs à afficher pour RL :", 
            options=features, 
            default=best_model_rl
        )

        # Séparer les variables explicatives (X) et la variable cible (y) pour l'ensemble d'entraînement
        X_train = train_stationnaire[selected_columns_rl]  # Variables explicatives pour l'entraînement
        y_train = train_stationnaire['PD']  # Variable cible pour l'entraînement
        # Variables explicatives (X) et cible (y) pour l'ensemble de test
        X_test = test_stationnaire[selected_columns_rl]  # Variables explicatives pour le test
        y_test = test_stationnaire['PD']  # Variable cible pour le test
        # Créer et entraîner le modèle de régression linéaire
        model_rl = LinearRegression()
        model_rl.fit(X_train, y_train)
        # Prédictions sur l'ensemble de test
        y_pred = model_rl.predict(X_test)
        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred)  # Erreur quadratique moyenne
        r2 = r2_score(y_test, y_pred)  # Coefficient de détermination (R^2)

        global_model_equation = {
        "intercept": model_rl.intercept_,
        "coefficients": dict(zip(selected_columns_rl, model_rl.coef_))
        }

        # Afficher l'équation du modèle
        equation = f"TD = {model_rl.intercept_:.2f}"
        for coef, feature in zip(model_rl.coef_, selected_columns_rl):
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.2f}*{feature.strip()}"
        st.write("**Équation du modèle :**")
        st.text(equation)


        # Afficher les performances du modèle
        st.write("**Performances du modèle :**")
        st.write(f"Erreur quadratique moyenne (MSE): {mse:.2f}")
        st.write(f"R² (coefficient de détermination): {r2:.2f}")


        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))

        # Utiliser les dates comme index pour le graphique
        ax.plot(train_stationnaire.index, y_train, label="TD (Train)", color="blue", alpha=0.7)
        ax.plot(test_stationnaire.index, y_test, label="TD (Test)", color="orange", alpha=0.7)
        ax.plot(test_stationnaire.index, y_pred, label="TD (Prédictions)", color="green", linestyle="--")

        # Ajouter un titre et des labels
        ax.set_title("Comparaison des Taux de défauts : Train, Test et Prédictions", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("TD")
        ax.legend()

        # Améliorer l'affichage des dates
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)

st.write("Ici, vous pouvez visualiser les prédictions en ajustant les scenarios du NGFS.")
    
with st.expander("Prediction"):    
    
    selected_model_scenarios = st.multiselect(
        "Choisissez les scénarios pour des prédictions:", 
        options=df_futur['Model_Scenario'].unique(),
        default=['NiGEM NGFS v1.24.2[REMIND-MAgPIE 3.3-4.8] Below 2?C', 'NiGEM NGFS v1.24.2[REMIND-MAgPIE 3.3-4.8] Net Zero 2050']
        )
        
    # Créez deux colonnes côte à côte

    col1, col2 = st.columns([3, 3])  # Ajustez les proportions si nécessaire

    with col1:

        st.write("Trimestrialiser pour predire.")

        #df_pass = df[['PD']]

        #results = []

        #for scenario in selected_model_scenarios :
        #    selected_columns_arx = [colonne + '_' + scenario for colonne in selected_columns_arx]
        #    df_trimestriel = df_trimestriel[selected_columns_arx]

#            start = df_trimestriel.index[0]
#             end = df_trimestriel.index[-1]
#            y_pred_futur = model_arx.predict(start=start, end=end , exog_oos=df_trimestriel)

#            results.append({
#                    f'PD {scenario}': y_pred_futur
#                })

#        plt.figure(figsize=(10, 6))
#        plt.plot(df_pass.index, df_pass['PD'], marker='o', color='b', label='PD pass')
#        for scenario in selected_model_scenarios :
#            plt.plot(df_selected_rl2.index, df_selected_rl2[f'PD_{scenario}'], marker='o', label=scenario) 
#        plt.title('Evolution of PD over time')
#        plt.xlabel('Year')
#        plt.ylabel('PD')
#        plt.legend()
#        plt.grid(True)
#        st.pyplot(plt)




        
    with col2:

        df_futur_retarde2 = df_futur_retarde2.pivot(columns="Model_Scenario", values=[col for col in df_futur_retarde2.columns if col != "Model_Scenario"])
        df_futur_retarde2.columns = [f"{col[0]}_{col[1]}" for col in df_futur_retarde2.columns]
        df_pass = df[['PD']]


        for scenario in  df_futur["Model_Scenario"].unique() :
            for i in range(len(df_futur_retarde2)):
                # Si on est dans l'année de la prédiction, on prédit la PD
                if pd.isna(df_futur_retarde2.iloc[i][f'PD_{scenario}']):
                    # Préparer les données d'entrée (X) pour la prédiction de la PD
                    X_current = np.array([df_futur_retarde2.iloc[i][[f"{col}_{scenario}" for col in selected_columns_rl]].values])
                    #X_current = sm.add_constant(X_current, has_constant='add')
                    predicted_PD = model_rl.predict(X_current)[0]
                    df_futur_retarde2.iloc[i, df_futur_retarde2.columns.get_loc(f'PD_{scenario}')] = predicted_PD
                
                # Mettre à jour PD_T1 et PD_T2 pour les années suivantes
                if i + 1 < len(df_futur_retarde2):  # Assurez-vous qu'on ne dépasse pas les indices
                    df_futur_retarde2.iloc[i + 1, df_futur_retarde2.columns.get_loc(f'PD_T1_{scenario}')] = df_futur_retarde2.iloc[i, df_futur_retarde2.columns.get_loc(f'PD_{scenario}')]
                
                if i + 2 < len(df_futur_retarde2):
                    df_futur_retarde2.iloc[i + 2, df_futur_retarde2.columns.get_loc(f'PD_T2_{scenario}')] = df_futur_retarde2.iloc[i, df_futur_retarde2.columns.get_loc(f'PD_{scenario}')]





        selected_columns_rl1 = selected_columns_rl + ['PD']
        cols_to_select = [f"{v}_{s}" for v in selected_columns_rl1 for s in selected_model_scenarios]
        # Sélectionner les colonnes correspondantes dans df_futur_retarde1
        df_selected_rl2 = df_futur_retarde2.loc[:, cols_to_select]

        equation = f"PD = {model_rl.intercept_:.2f}"
        for coef, feature in zip(model_rl.coef_, selected_columns_rl):
            sign = "+" if coef >= 0 else "-"
            equation += f" {sign} {abs(coef):.2f}*{feature.strip()}"
        st.write("**Équation du modèle :**")
        st.text(equation)

        #df_selected_rl2['PD'] = 1 / (1 + np.exp(-df_selected_rl2['PD']))
            
        # Tracer les résultats
        plt.figure(figsize=(10, 6))
        plt.plot(df_pass.index, df_pass['PD'], marker='o', color='b', label='PD pass')
        for scenario in selected_model_scenarios :
            plt.plot(df_selected_rl2.index, df_selected_rl2[f'PD_{scenario}'], marker='o', label=scenario) 
        plt.title('Evolution of PD over time')
        plt.xlabel('Year')
        plt.ylabel('PD')
        plt.legend()
        plt.grid(True)
        st.pyplot(plt)


        st.dataframe(df_selected_rl2)

    # Créez deux colonnes côte à côte
    #col1, col2 = st.columns([3, 3])  # Ajustez les proportions si nécessaire
    #with col1:
    #    st.write("Trimestrialiser pour predire.")

    #with col2:
        #df_pass = df[['PD']]
        #def predict_PD(i, df_futur_retarde4, model_rl):
        #        """Prédit la PD et met à jour le DataFrame."""
        #        if pd.isna(df_futur_retarde4.iloc[i]['PD']):
        #            X_current = np.array([df_futur_retarde4.iloc[i][selected_columns_rl].values])  
        #            predicted_PD = model_rl.predict(X_current)[0]
        #            df_futur_retarde4.iloc[i, df_futur_retarde4.columns.get_loc('PD')] = predicted_PD

        #def update_PD_future(i, df_futur_retarde4):
        #    """Met à jour PD_T1 et PD_T2 pour les années suivantes."""
        #    if i + 1 < len(df_futur_retarde4):
        #        df_futur_retarde4.iloc[i + 1, df_futur_retarde4.columns.get_loc('PD_T1')] = df_futur_retarde4.iloc[i, df_futur_retarde4.columns.get_loc('PD')]
        #    if i + 2 < len(df_futur_retarde4):
        #        df_futur_retarde4.iloc[i + 2, df_futur_retarde4.columns.get_loc('PD_T2')] = df_futur_retarde4.iloc[i, df_futur_retarde4.columns.get_loc('PD')]

        # Boucle principale
        #for i in range(len(df_futur_retarde4) - 2):  # Évite l'IndexError
        #    scenario_i = df_futur_retarde4.iloc[i]['Model_Scenario']
        #    scenario_i1 = df_futur_retarde4.iloc[i+1]['Model_Scenario']
        #    scenario_i2 = df_futur_retarde4.iloc[i+2]['Model_Scenario']

        #    if scenario_i == scenario_i1 == scenario_i2:
        #        predict_PD(i, df_futur_retarde4, model_rl)
        #        update_PD_future(i, df_futur_retarde4)

        #    elif scenario_i == scenario_i1 != scenario_i2:
        #        predict_PD(i, df_futur_retarde4, model_rl)
        #        if i + 1 < len(df_futur_retarde4):
        #            df_futur_retarde4.iloc[i + 1, df_futur_retarde4.columns.get_loc('PD_T1')] = df_futur_retarde4.iloc[i, df_futur_retarde4.columns.get_loc('PD')]

        #    elif scenario_i != scenario_i1 == scenario_i2:
        #        predict_PD(i, df_futur_retarde4, model_rl)

        #df_futur_retarde4 = df_futur_retarde4[df_futur_retarde4['Model_Scenario'].isin(selected_model_scenarios)]
        # Sélectionner les colonnes demandées
        #df_selected_rl = df_futur_retarde4[['Model_Scenario'] + ['PD'] + selected_columns_rl]
        #df_selected_rl = df_selected_rl.pivot(columns="Model_Scenario", values=[col for col in df_selected_rl.columns if col != "Model_Scenario"])
        #df_selected_rl.columns = [f"{col[0]}_{col[1]}" for col in df_selected_rl.columns]

        #equation = f"PD = {model_rl.intercept_:.2f}"
        #for coef, feature in zip(model_rl.coef_, selected_columns_rl):
        #    sign = "+" if coef >= 0 else "-"
        #    equation += f" {sign} {abs(coef):.2f}*{feature.strip()}"
        #st.write("**Équation du modèle :**")
        #st.text(equation)
            
        # Tracer les résultats
        #plt.figure(figsize=(10, 6))
        #plt.plot(df_pass.index, df_pass['PD'], marker='o', color='b', label='PD pass')
        #for scenario in df_futur_retarde4['Model_Scenario'].unique():
        #    scenario_df = df_futur_retarde4[df_futur_retarde4['Model_Scenario'] == scenario]
        #    plt.plot(scenario_df.index, scenario_df['PD'], marker='o', label=scenario)
        #plt.title('Evolution of PD over time')
        #plt.xlabel('Year')
        #plt.ylabel('PD')
        #plt.legend()
        #plt.grid(True)
        #st.pyplot(plt)

        #st.dataframe(df_selected_rl)
 
st.header("Modeling & Prediction (Machine Learning)")

with st.expander("Modeling"):

    col1, col2 = st.columns([3 , 3])  # Colonnes principales
    with col1:
        st.subheader("NeuralProphet")
        st.write("Intégre des réseaux neuronaux.Conçu pour la prévision de séries temporelles avec tendances, saisonnalités, et effets exogènes.")
    with col2:
        st.subheader("SETAR")
        st.write("Un modèle non linéaire pour les séries temporelles, où l’évolution future dépend d’un seuil. Divise la série en plusieurs régimes (ex: haute et basse volatilité) avec des dynamiques différentes. Adapté aux séries avec des changements de comportement brutaux.")


    col1, col2, col3 = st.columns([2, 2, 2])  # Colonnes principales

    # Modèle 1 : SVR
    with col1:
        st.subheader("SVR")

        kernels = ['linear', 'poly', 'rbf', 'sigmoid']
        C_values = [0.1, 1, 10, 100]
        epsilon_values = [0.01, 0.1, 0.2]

        X_train = train_stationnaire[selected_features]  # Variables explicatives pour l'entraînement
        y_train = train_stationnaire['PD']  # Variable cible pour l'entraînement
        X_test = test_stationnaire[selected_features]  # Variables explicatives pour le test
        y_test = test_stationnaire['PD']  # Variable cible pour le test

        results = []

        # Boucle pour générer toutes les combinaisons possibles (2 features ici)
        for r in range(2, 3):  # Ajustez r pour modifier le nombre de features
            for combination in itertools.combinations(selected_features, r):
                for kernel in kernels:
                    for C in C_values:
                        for epsilon in epsilon_values:
                            # Sous-ensembles des données avec les features sélectionnées
                            X_train_subset = X_train[list(combination)]
                            X_test_subset = X_test[list(combination)]

                            # Modèle de régression linéaire
                            model_svr = SVR(kernel=kernel, C=C, epsilon=epsilon)
                            model_svr.fit(X_train_subset, y_train)

                            # Prédictions
                            y_pred = model_svr.predict(X_test_subset)

                            # Calcul des métriques
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            mape = (abs((y_test - y_pred) / y_test).mean()) * 100

                            # Résultats
                            results.append({
                                'Combination': combination,
                                'R2': r2,
                                'MSE': mse,
                                'RMSE': rmse,
                                'MAE': mae,
                                'MAPE': mape,
                                'Kernel': kernel,
                                'C': C,
                                'Epsilon': epsilon
                            })

        # Conversion des résultats en DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='R2', ascending=False)
        best_model_svr = results_df.iloc[0]['Combination']
        best_kernels_svr = results_df.iloc[6]['Kernel']
        best_C_svr = results_df.iloc[7]['C']
        best_epsilon_svr = results_df.iloc[8]['Epsilon']


        # Streamlit app
        st.subheader("Cross-validation")
        st.write("Résultats des métriques de performance pour différentes combinaisons de features :")

        # Afficher les résultats sous forme de table
        st.dataframe(results_df)  # Option dynamique avec filtrage et redimensionnement

        selected_columns_svr = st.multiselect(
            "Choisissez les indicateurs à afficher pour SVR :   ",
            options=train_stationnaire.columns,
            default=best_model_svr
        )

        # Colonnes internes pour les paramètres SVR
        col1_1, col1_2, col1_3 = st.columns([2, 2, 2])

        with col1_1:
            kernel = st.selectbox(
                "Kernels", 
                options=kernels,
                index=kernels.index(best_kernels_svr)
            )

        with col1_2:
            max_c = st.slider(
                "C",  
                min_value=0.01,  
                max_value=100.0,  
                value=float(best_C_svr),  
                step=0.1
            )

        with col1_3:
            max_epsilon = st.slider(
                "Epsilon",  
                min_value=0.01,  
                max_value=20.0,  
                value=float(best_epsilon_svr),  
                step=0.1
            )

        X_train = train_stationnaire[selected_columns_svr]  # Variables explicatives pour l'entraînement
        y_train = train_stationnaire['PD']  # Variable cible pour l'entraînement
        # Variables explicatives (X) et cible (y) pour l'ensemble de test
        X_test = test_stationnaire[selected_columns_svr]  # Variables explicatives pour le test
        y_test = test_stationnaire['PD']  # Variable cible pour le test
        # Créer et entraîner le modèle de régression linéaire

        model_svr = SVR(kernel=kernel, C=max_c, epsilon=max_epsilon)
        model_svr.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model_svr.predict(X_test)
        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred)  # Erreur quadratique moyenne
        r2 = r2_score(y_test, y_pred)  # Coefficient de détermination (R^2)


        # Afficher les performances du modèle
        st.write("**Performances du modèle :**")
        st.write(f"Erreur quadratique moyenne (MSE): {mse:.2f}")
        st.write(f"R² (coefficient de détermination): {r2:.2f}")


        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))

        # Utiliser les dates comme index pour le graphique
        ax.plot(train_stationnaire.index, y_train, label="PD (Train)", color="blue", alpha=0.7)
        ax.plot(test_stationnaire.index, y_test, label="PD (Test)", color="orange", alpha=0.7)
        ax.plot(test_stationnaire.index, y_pred, label="PD (Prédictions)", color="green", linestyle="--")
        # Ajouter un titre et des labels
        ax.set_title("Comparaison des PD : Train, Test et Prédictions", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("PD")
        ax.legend()

        # Améliorer l'affichage des dates
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)

    # Modèle 2 : RandomForestRegressor
    with col2:
        st.subheader("RandomForestRegressor")


        n_estimators = [100, 200, 300]
        max_depths = [1, 10, 20]

        X_train = train_stationnaire[selected_features]  # Variables explicatives pour l'entraînement
        y_train = train_stationnaire['PD']  # Variable cible pour l'entraînement
        X_test = test_stationnaire[selected_features]  # Variables explicatives pour le test
        y_test = test_stationnaire['PD']  # Variable cible pour le test

        results = []

        # Boucle pour générer toutes les combinaisons possibles (2 features ici)
        for r in range(2, 3):  # Ajustez r pour modifier le nombre de features
            for combination in itertools.combinations(selected_features, r):
                for n_estimator in n_estimators:
                    for max_depth in max_depths:
                        # Sous-ensembles des données avec les features sélectionnées
                        X_train_subset = X_train[list(combination)]
                        X_test_subset = X_test[list(combination)]

                        # Modèle de régression linéaire
                        model_rf = RandomForestRegressor(n_estimators=n_estimator,  max_depth=max_depth)
                        model_rf.fit(X_train_subset, y_train)

                        # Prédictions
                        y_pred = model_rf.predict(X_test_subset)

                        # Calcul des métriques
                        r2 = r2_score(y_test, y_pred)
                        mse = mean_squared_error(y_test, y_pred)
                        rmse = sqrt(mse)
                        mae = mean_absolute_error(y_test, y_pred)
                        mape = (abs((y_test - y_pred) / y_test).mean()) * 100

                        # Résultats
                        results.append({
                            'Combination': combination,
                            'R2': r2,
                            'MSE': mse,
                            'RMSE': rmse,
                            'MAE': mae,
                            'MAPE': mape,
                            'n_estimator': n_estimator,
                            'max_depth': max_depth
                        })

        # Conversion des résultats en DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='R2', ascending=False)
        best_model_rf = results_df.iloc[0]['Combination']
        best_n_estimator_rf = results_df.iloc[6]['n_estimator']
        best_max_depth_rf = results_df.iloc[7]['max_depth']


        # Streamlit app
        st.subheader("Cross-validation")
        st.write("Résultats des métriques de performance pour différentes combinaisons de features :")

        # Afficher les résultats sous forme de table
        st.dataframe(results_df)  # Option dynamique avec filtrage et redimensionnement

        selected_columns_rf = st.multiselect(
            "Choisissez les indicateurs à afficher pour RL :", 
            options=train_stationnaire.columns, 
            default=best_model_rf
        )

        # Colonnes internes pour les paramètres Random Forest
        col2_1, col2_2 = st.columns([2, 2])

        with col2_1:
            n_estimators_rf = st.slider(
                "Nombre d'arbres (n_estimators)",  
                min_value=100,  
                max_value=300,  
                value=int(best_n_estimator_rf),  
                step=100
            )

        with col2_2:
            max_depth_rf = st.slider(
                "Profondeur maximale (max_depth)",  
                min_value=1,  
                max_value=20,  
                value=int(best_max_depth_rf),  
                step=1
            )

        X_train = train_stationnaire[selected_columns_rf]  # Variables explicatives pour l'entraînement
        y_train = train_stationnaire['PD']  # Variable cible pour l'entraînement
        # Variables explicatives (X) et cible (y) pour l'ensemble de test
        X_test = test_stationnaire[selected_columns_rf]  # Variables explicatives pour le test
        y_test = test_stationnaire['PD']  # Variable cible pour le test
        # Créer et entraîner le modèle de régression linéaire

        model_rf = RandomForestRegressor(n_estimators=n_estimators_rf,  max_depth=max_depth_rf)
        model_rf.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model_rf.predict(X_test)
        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred)  # Erreur quadratique moyenne
        r2 = r2_score(y_test, y_pred)  # Coefficient de détermination (R^2)


        # Afficher les performances du modèle
        st.write("**Performances du modèle :**")
        st.write(f"Erreur quadratique moyenne (MSE): {mse:.2f}")
        st.write(f"R² (coefficient de détermination): {r2:.2f}")


        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))

        # Utiliser les dates comme index pour le graphique
        ax.plot(train_stationnaire.index, y_train, label="PD (Train)", color="blue", alpha=0.7)
        ax.plot(test_stationnaire.index, y_test, label="PD (Test)", color="orange", alpha=0.7)
        ax.plot(test_stationnaire.index, y_pred, label="PD (Prédictions)", color="green", linestyle="--")
        # Ajouter un titre et des labels
        ax.set_title("Comparaison des PD : Train, Test et Prédictions", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("PD")
        ax.legend()

        # Améliorer l'affichage des dates
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)


    # Modèle 3 : GradientBoostingRegressor
    with col3:
        st.subheader("GradientBoostingRegressor")

        
        n_estimators = [100, 200, 300]
        max_depths = [1, 10, 20]
        learning_rates= [0.01, 0.1, 1, 2]

        X_train = train_stationnaire[selected_features]  # Variables explicatives pour l'entraînement
        y_train = train_stationnaire['PD']  # Variable cible pour l'entraînement
        X_test = test_stationnaire[selected_features]  # Variables explicatives pour le test
        y_test = test_stationnaire['PD']  # Variable cible pour le test

        results = []

        # Boucle pour générer toutes les combinaisons possibles (2 features ici)
        for r in range(2, 3):  # Ajustez r pour modifier le nombre de features
            for combination in itertools.combinations(selected_features, r):
                for n_estimator in n_estimators:
                    for max_depth in max_depths:
                        for learning_rate in learning_rates:
                            # Sous-ensembles des données avec les features sélectionnées
                            X_train_subset = X_train[list(combination)]
                            X_test_subset = X_test[list(combination)]

                            # Modèle de régression linéaire
                            model_gb = GradientBoostingRegressor(n_estimators=n_estimator, max_depth=max_depth, learning_rate=learning_rate)

                            model_rf.fit(X_train_subset, y_train)

                            # Prédictions
                            y_pred = model_rf.predict(X_test_subset)

                            # Calcul des métriques
                            r2 = r2_score(y_test, y_pred)
                            mse = mean_squared_error(y_test, y_pred)
                            rmse = sqrt(mse)
                            mae = mean_absolute_error(y_test, y_pred)
                            mape = (abs((y_test - y_pred) / y_test).mean()) * 100

                            # Résultats
                            results.append({
                                'Combination': combination,
                                'R2': r2,
                                'MSE': mse,
                                'RMSE': rmse,
                                'MAE': mae,
                                'MAPE': mape,
                                'n_estimator': n_estimator,
                                'max_depth': max_depth,
                                'learning_rate': learning_rate
                            })

        # Conversion des résultats en DataFrame
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values(by='R2', ascending=False)
        best_model_bg = results_df.iloc[0]['Combination']
        best_n_estimator_bg = results_df.iloc[6]['n_estimator']
        best_max_depth_bg = results_df.iloc[7]['max_depth']
        best_learning_rate_bg = results_df.iloc[8]['learning_rate']


        # Streamlit app
        st.subheader("Cross-validation")
        st.write("Résultats des métriques de performance pour différentes combinaisons de features :")

        # Afficher les résultats sous forme de table
        st.dataframe(results_df)  # Option dynamique avec filtrage et redimensionnement

        selected_columns_bg = st.multiselect(
            "Choisissez les indicateurs à afficher pour BG :", 
            options=train_stationnaire.columns, 
            default=best_model_bg
        )

        # Colonnes internes pour les paramètres Gradient Boosting
        col3_1, col3_2, col3_3 = st.columns([1, 1, 1])

        with col3_1:
            n_estimators_gb = st.slider(
                "Nombre d'arbres(n_estimators)",  
                min_value=100,  
                max_value=300,  
                value=int(best_n_estimator_bg),  
                step=50
            )

        with col3_2:
            max_depth_gb = st.slider(
                "Profondeur maximale(max_depth)",  
                min_value=1,  
                max_value=20,  
                value=int(best_max_depth_bg),  
                step=1
            )

        with col3_3:
            learning_rate_gb = st.slider(
                "Taux d'apprentissage (learning_rate)",  
                min_value=0.01,  
                max_value=0.2,  
                value=float(best_learning_rate_bg),  
                step=0.01
            )

        X_train = train_stationnaire[selected_columns_bg]  # Variables explicatives pour l'entraînement
        y_train = train_stationnaire['PD']  # Variable cible pour l'entraînement
        # Variables explicatives (X) et cible (y) pour l'ensemble de test
        X_test = test_stationnaire[selected_columns_bg]  # Variables explicatives pour le test
        y_test = test_stationnaire['PD']  # Variable cible pour le test
        # Créer et entraîner le modèle de régression linéaire

        model_gb = GradientBoostingRegressor(n_estimators=n_estimators_gb, max_depth=max_depth_gb, learning_rate=learning_rate_gb)
        model_gb.fit(X_train, y_train)

        # Prédictions sur l'ensemble de test
        y_pred = model_gb.predict(X_test)
        # Évaluation du modèle
        mse = mean_squared_error(y_test, y_pred)  # Erreur quadratique moyenne
        r2 = r2_score(y_test, y_pred)  # Coefficient de détermination (R^2)


        # Afficher les performances du modèle
        st.write("**Performances du modèle :**")
        st.write(f"Erreur quadratique moyenne (MSE): {mse:.2f}")
        st.write(f"R² (coefficient de détermination): {r2:.2f}")


        # Création du graphique
        fig, ax = plt.subplots(figsize=(10, 6))

        # Utiliser les dates comme index pour le graphique
        ax.plot(train_stationnaire.index, y_train, label="PD (Train)", color="blue", alpha=0.7)
        ax.plot(test_stationnaire.index, y_test, label="PD (Test)", color="orange", alpha=0.7)
        ax.plot(test_stationnaire.index, y_pred, label="PD (Prédictions)", color="green", linestyle="--")
        # Ajouter un titre et des labels
        ax.set_title("Comparaison des PD : Train, Test et Prédictions", fontsize=14)
        ax.set_xlabel("Date")
        ax.set_ylabel("PD")
        ax.legend()

        # Améliorer l'affichage des dates
        plt.xticks(rotation=45)
        plt.grid(True)
        st.pyplot(fig)

with st.expander("Prediction"):
    st.write("Modeling & Prediction (Machine Learning)")