#importar librerías
import pandas as pd #version. 1.3.4
import numpy as np #version. 1.20.3

#importar librerías para visualización
import seaborn as sns #version. 0.11.2
import matplotlib.pyplot as plt #version. 3.4.3
from sklearn.metrics import classification_report




#Representación gráfica de las variables

def grafica_subplots(df, columns=3):
    
    """
    Gráfica la distribución de las variables dentro de un dataframe, según su tipo de dato.

        Parametros:
            df: DataFrame a graficar
            columns: cantidad de columnas en las que se visualizaran los graficos en el jupyter, 
                    predeterminado en 3 columnas.
        
        Retorno:
            Gráfica la distribución de las variables.


    """
    
    rows = np.ceil(df.shape[1] / columns)
    height = rows * 3.5
    fig = plt.figure(figsize=(12, height))
 
    for n, i in enumerate(df.columns):
        
        if df[i].dtype in ('object', 'int64', 'uint8') :
           fig.add_subplot(rows, columns, n+1)
           ax = sns.countplot(x=i, data=df)
           plt.title(i)
           plt.xlabel('')
           for p in ax.patches:
               height = p.get_height()
               ax.text(p.get_x()+p.get_width()/2., height + .5,
                   '{:1.2f}'.format(height/len(df[i])), ha="center")
        if df[i].dtype == 'float64':
           fig.add_subplot(rows, columns, n+1)
           ax = sns.distplot(df[i])
           plt.title(i)
           plt.xlabel('')
            
    plt.tight_layout()
    plt.show()

    return


#Función que binariza variables categóricas
def binarizador (df):
	"""
    Binariza el dataframe ingresado utilizando get_dummies y las variables categoricas del DataFrame utilizado.
	    Parametros:
		    df: DataFrame
	    Retorno:
		    Devuelve un df binarizado.
	"""		
	#declaramos un df vacio a utilizar 
	df_binarizado = pd.DataFrame()

	#obtenemos una lista de las variables categóricas
	variables_categoricas = list(df.select_dtypes(object).columns)

	#binarizamos las variables categóricas        
	df_binarizado=pd.get_dummies(data=df, columns=variables_categoricas,drop_first=True)

	#reemplazamos el caracter '-' por '_' en los nombres de las variables
	#para evitar problemas futuros en el modelamiento
	df_binarizado=df_binarizado.rename(columns=lambda x: x.replace("-","_"))

	return df_binarizado

    	


#Observamos las diferentes columnas que hay entre los DataFrame:

def compara_columns_df (df_train,df_test):
    """
    Compara las columnas entre dos Data Frames
	    Parametros:
		    df_train: DataFrame de entrenamiento
            df_test: DataFrame de testeo
	
        Retorno:
		    Entrega los campos que difieren entre un DataFrame y el otro. 
	"""	
    #Diferencia entre DF de entrenamiento y DF de testeo:
    diferencia_1 = set(df_train.columns.values).difference(set(df_test.columns.values))

    #Diferencia entre DF de testeo y DF de entrenamiento:
    diferencia_2 = set(df_test.columns.values).difference(set(df_train.columns.values))

    print(f'Campos que difieren entre DataFrame de entrenamiento y DataFrame de testeo: \n {diferencia_1}')

    print(f'\nCampos que difieren entre DataFrame de testeo y DataFrame de entrenamiento: \n {diferencia_2}')






##CONTABILIZADOR DE NAN
def contabilizador_nan (df):
    """
    Cuenta los valores perdidos de un Data Frame
	    Parametros:
		    df: DataFrame 
            
        Retorno:
		    Devuelve el número total de valores perdidos. 
	"""	
    
    #Se cuentan los NaN en dataframe
    df.isna().sum()
    print(df.isna().sum())
    print(f'total de valores perdidos: {np.sum((df.isna().sum()).values)}')



##CONTADOR DE FILAS Y COLUMNAS DE DOS DATAFRAME
def contador_filas_columns(df_train,df_test):
    """
    Permite contar el numero de filas y de columnas del Data Frame de entrenamiento y el de testeo
	    Parametros:
		    df_train: DataFrame de entrenamiento
            df_test: DataFrame de testeo
            
        Retorno:
		    Devuelve la cantidad de registros en df de entrenamiento y el de testeo. 
	"""	
    
    #Observamos la cantidad de filas y de columnas que hay en los dos datos de los Data Frame ya procesados
    train_shape = df_train.shape
    test_shape = df_test.shape

    print(f'La cantidad de registros en df de entrenamiento:\n filas, columnas: {train_shape}')
    print(f'La cantidad de registros en df de testeo:\n filas, columnas: {test_shape}')



#GRAFICAR VARIABLE OBJETIVO
def graficar_var_objetivo(df, nombre_variable, num_objetivo, nombre_df): 
    """
    Permite graficar la variable objetivo de un Data Frame, puede
     ser utilizada cuando hay más de una variable objetivo y más de un Data Frame.

	    Parametros:
		    df: DataFrame 
            nombre_variable: Nombre de la variable objetivo
            num_objetivo: Número de la variable objetivo.
            nombre_df: Nombre del Data Frame en una palabra (Ej: entrenamiento, testeo)
            
        Retorno:
		    Grafico de la variable objetivo con un titulo acorde a su utilidad. 
	"""	    
    # Distribución del valor objetivo      
    print (df.value_counts())

    # Gráfico de distribción nueva variable objetivo
    ax = sns.countplot(df)
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x()+p.get_width()/2., height + .5,
            '{:1.2f}'.format(height/len(df)), ha="center")
    plt.title(f'Distribución {num_objetivo} variable objectivo: {nombre_variable} en DateFrame de {nombre_df}')
    plt.show()



#CREAR COLUMNA 'VIOLENT'
    """
    Permite crear una nueva columna llamada 'Violent' para el segundo vector objetivo.
	    Parametros:
		    df: DataFrame 
        Retorno:
		    Devuelve una nueva columna llamada 'violent'. 
	"""	
def column_violent (df):
    # Creación 2do vector objetivo 
    df['violent'] = 'N'
    for i, row in df.iterrows():
        if (row['pf_hands'] == 'Y') | (row['pf_wall'] == 'Y') | (
            row['pf_grnd'] == 'Y') | (row['pf_drwep'] == 'Y') | (
            row['pf_baton'] == 'Y') | (row['pf_hcuff'] == 'Y') | (
            row['pf_pepsp'] == 'Y') | (row['pf_ptwep'] == 'Y') | (
            row['pf_other'] == 'Y'):

            df.loc[i, 'violent'] = 'Y'



def compare_classifiers(estimators, X_test, y_test, n_cols=2):
    
    """
    Compara en forma gráfica las métricas de clasificación a partir de una lista de 
    tuplas con los modelos (nombre_modelo, modelo_entrendo) 
    """

    rows = np.ceil(len(estimators)/n_cols)
    height = 2 * rows
    width = n_cols * 5
    fig = plt.figure(figsize=(width, height))

    colors = ['dodgerblue', 'tomato', 'purple', 'orange']

    for n, clf in enumerate(estimators):

        y_hat = clf[1].predict(X_test)
        # Si las prediciones son probabilidades, binarizar
        if y_hat.dtype == 'float32':
            y_hat = [1 if i >= .5 else 0 for i in y_hat]

        dc = classification_report(y_test, y_hat, output_dict=True)

        plt.subplot(rows, n_cols, n + 1)

        for i, j in enumerate(['0', '1', 'macro avg']):

            tmp = {'0': {'marker': 'x', 'label': f'Class: {j}'},
                   '1': {'marker': 'x', 'label': f'Class: {j}'},
                   'macro avg': {'marker': 'o', 'label': 'Avg'}}

            plt.plot(dc[j]['precision'], [1], marker=tmp[j]['marker'], color=colors[i])
            plt.plot(dc[j]['recall'], [2], marker=tmp[j]['marker'], color=colors[i])
            plt.plot(dc[j]['f1-score'], [3], marker=tmp[j]['marker'],color=colors[i], label=tmp[j]['label'])
            plt.axvline(x=.5, ls='--')

        plt.yticks([1.0, 2.0, 3.0], ['Precision', 'Recall', 'f1-Score'])
        plt.title(clf[0])
        plt.xlim((0.1, 1.0))

        if (n + 1) % 2 == 0:
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            
    fig.tight_layout()
    
    return