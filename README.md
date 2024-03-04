# FraudDetect
Descripción y logo


## Créditos
[Pablo Santos Quirce](https://github.com/pabloquirce23)

[Pablo Oller Pérez](https://github.com/pabloquirce23)

[Alejandro Castillo](https://github.com/pabloquirce23)


## Video


## Presentación


## I. Justificación y descripción del proyecto
La funcionalidad de FraudDetect es detectar si las transacciones bancarias que les pasamos mediante tablas de PDF son fraude o no, además clasifica esas transacciones en diferentes agrupaciones en referencia al tipo de gasto que sean. También tiene integrado una funcionalidad de chatbot utilizando una API Key de OpenAI.

Para el entrenamiento de los modelos hemos utilizado un Dataset de Kaggle del cual podeis encontrar el enlace en el apartado de Bibliografía.


## II. Obtención de datos.

Para obtener los datos que utilizaremos para entrenar nuestros modelos hemos decidido emplear un Dataset en el que tenemos información sobre transacciones bancarias. En él tratamos con las siguientes columnas:

* **Time:** tiempo en segundos que pasaron desde la anterior transacción del Dataset.
* **Columnas desde V1 a V28:** datos de un Análisis de Componentes Principales (técnica estadística utilizada para simplificar la complejidad de conjuntos de datos de grandes dimensiones) sometidos a una reducción para proteger la información personal de los clientes de la entidad bancaria.
* **Amount:** Cantidad monetaria de la transacción bancaria.
* **Class:** Muestra si la transacción bancaria es fraudulenta o no. 1 = Fraude y 0 = No es Fraude.

En este fragmento de código podemos observar la carga de nuestro Dataset:

```
ccdf = pd.read_csv('/content/creditcard.csv')
```


## III. Limpieza de datos.

Para la primera limpieza de datos sólo necesitamos eliminar los valores nulos existentes en el Dataset, para ello detectamos los valores nulos que hay en nuestro Dataframe:

```
ccdf.isnull().sum()
```

Eliminamos los nulos en las columnas en las que están:

```
ccdf.dropna(subset=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"],
            inplace=True)
```

Además de eso transformamos todos los valores que no son del tipado correcto a tipo float:

```
ccdf['V22'] = ccdf['V22'].astype(float)
```


## IV. Exploración y visualización de los datos
Para la exploración y visualización de los datos realizamos distintos procesos, en primera instancia hacemos un head para ver la estructura de nuestro Dataframe:

![image](https://drive.google.com/uc?export=view&id=162MYEc4cYlY3vltdTicVb2qLRGr66tjA)

Después realizamos una equilibración de Dataset para poder así ajustar la distribución de las clases y poder así representarlas de la misma manera. Esto lo hacemos porque en el Dataset que utilizamos nos hemos encontrado con un desequilibrio muy notable entre los casos de fraude y los casos de no fraude. Este tipo de desequilibrio puede sesgar enormemente los rendimientos de nuestro modelo de detección de fraude y de nuestro modelo de clusterización ya que cuando existe suele favorecer a la clase que más valores tiene. En este fragmento de código podemos apreciar un conteo del número de valores de cada clase:

![image](https://drive.google.com/uc?export=view&id=11r6w-2UovMD9glvVwVsThWDFLzzJT2M4)

Aquí se puede ver el mapa de calor que hemos utilizado para observar las correlaciones existentes de nuestro Dataset:

![image](https://drive.google.com/uc?export=view&id=1YQt_MKyK6rti9XPIiRBSzbiWJyoNubmq)


## V. Preparación de los datos para los algoritmos de Machine Learning.

Ahora pasamos a la preparación de los datos con referencia al desarrollo y entrenamiento de nuestros modelos. Para ello vamos a observar los valores átipicos de cada columna de nuestro Dataset. Esto se ha realizado mediante el siguiente código:

![image](https://drive.google.com/uc?export=view&id=1RuzEi8wgtCYrFYCs0vISaQI5kMP9xPWK)

Y aquí podemos ver el resultado final:

![image](https://drive.google.com/uc?export=view&id=1fd_73OkFeu_cSfTSwWxs0Hee-xzxtsnE)

Ahora pasamos al código con el que eliminamos los valores átipicos de las columnas de nuestro Dataset. En él se puede apreciar que hacemos una observación de varios factores relevantes para detectar correctamente estos valores. Observamos los límites del rango intercuartil (IQR), calculamos el rango (diferencia entre el percentil 75 y el percentil 25), el valor de corte (1,5 veces el IQR) y los límites inferiores y superiores:

![image](https://drive.google.com/uc?export=view&id=1bET0Ph70hvPnffkEwNc65QIvNxcrOIu7)

Aquí pongo los resultados de todos estos factores de una de las columnas:

![image](https://drive.google.com/uc?export=view&id=1r4RbR-o67dRMWIu04b2OAMc8PKSMPogK)

Después volvemos a mostrar las mismas gráficas usadas anteriormente para ver como han desaparecido la mayoría de los valores atípicos:

![image](https://drive.google.com/uc?export=view&id=12ihVD2-DgZqIelbH4nLm4yQ8IZewRU9r)

Ahora haciendo otra pequeña visualización al Dataset podemos observar que los datos están escalados de formas totalmente diferentes las unas de las otras:

![image](https://drive.google.com/uc?export=view&id=1cAFwWPJXPHd5xs_9D7wcKlfN7lxU3B0V)

Debido a ello nos tocará estandarizar los datos:

![image](https://drive.google.com/uc?export=view&id=19KPX0qgIGxZre82wAkwNNWeHsKSxoYyt)

Después de estos procesos aplicamos unas pocas transformaciones más y ya tendremos los datos listos para el desarrollo de nuestros modelos.


## VI.I Entrenamiento del modelo de predicción de fraude y comprobación de su rendimiento.
![image](https://drive.google.com/uc?export=view&id=19KPX0qgIGxZre82wAkwNNWeHsKSxoYyt)

![image](https://drive.google.com/uc?export=view&id=16USXpr8_gcmercnaiyOpGsd8z4XLJ34y)

![image](https://drive.google.com/uc?export=view&id=1svODOsOQpgTH5tiuKkMp0d6BZGTIaAsL)

![image](https://drive.google.com/uc?export=view&id=1VL9D6WcBqnWAFEng9oEXqlNy95xaSvQK)

![image](https://drive.google.com/uc?export=view&id=1iWpsKYozHsOawrhCHvFHowXzuWqYAcPs)

![image](https://drive.google.com/uc?export=view&id=15pHnkp7nBNh8lsrD0tbk8kKWc5UIghOW)

![image](https://drive.google.com/uc?export=view&id=1bZ5zquMvlRIqzSDIJk-BfGLAejlvxTbN)


## VI.II Entrenamiento del modelo de clusterización y comprobación de su rendimiento.
![image](https://drive.google.com/uc?export=view&id=1t-lPQUipgF_RGGYtF-0kRLnGlu9KC3Sg)

![image](https://drive.google.com/uc?export=view&id=1XQ1fDcUmmAhrqexqOzAmz-_3oKLg3UpQ)

![image](https://drive.google.com/uc?export=view&id=1-Hxrc6iCxJhUvrabTyO8z1VtCQFLW9HH)

![image](https://drive.google.com/uc?export=view&id=1kIDYR9hrX8Sy2ZRr_YXFJgsYYVigFO2x)

![image](https://drive.google.com/uc?export=view&id=1XLIn-xVADGe5gRwXmEURV6cafU418_9-)


## VII. PLN


## VIII. Aplicación web


## IX. Conclusiones


## X. Bibliografía

