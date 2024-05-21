import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from statsmodels.miscmodels.ordinal_model import OrderedModel
import statsmodels.formula.api as smf
from shared import app_dir, data
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import f
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from patsy import dmatrices
data_copy = data.copy()
data_copy['age'] = data['age'].astype('category').cat.codes

# Ajuster un modèle de régression logistique multinomiale
model = OrderedModel.from_formula('age ~ id', data=data_copy).fit()

pred = model.predict().argmax(1)
df = pd.DataFrame({
    model.model.endog_names : model.model.endog,
    'y_pred' : pred
})
print(df['y_pred'], df[model.model.endog_names])
df['correcte'] = df['y_pred'] == df[model.model.endog_names]
correct_counts = df['correcte'].groupby(df['y_pred']).value_counts(True)
correct_proportions = correct_counts.loc[:, True]
print(correct_counts)
endog_categories = pd.Categorical(data[model.model.endog_names], ordered=True).categories
observed_counts = df[model.model.endog_names].value_counts(True).sort_index()
predicted_counts = df['y_pred'].value_counts(True).sort_index()
df = pd.DataFrame({'Freq': predicted_counts, 'Correct': correct_proportions})
df.fillna(0, inplace=True)
df.index=endog_categories[df.index]
print(df)
fig = plt.figure()
plt.scatter(endog_categories,observed_counts.values, color='white',edgecolors='k',label='Observé', zorder=2)
plt.vlines(df.index, 0,df['Freq']*df['Correct'], color = 'green',linestyle= "--", alpha=0.6, label='Prédiction juste', zorder=1 )
plt.vlines(df.index, df['Freq']*df['Correct'],df['Freq'], color = 'red',linestyle= "--", alpha=0.6, label='Prédiction fausse', zorder=1)
plt.xlabel(model.model.endog_names)
plt.ylabel('Fréquence')
plt.ylim(bottom=0)
plt.legend()
plt.show()
