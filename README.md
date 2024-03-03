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

![image](https://drive.google.com/uc?export=view&id=1LY0ey4j7DPhUAtiGFBwMY7cr9I33Esve)


## III. Limpieza de datos.

Para la primera limpieza de datos sólo necesitamos eliminar los valores nulos existentes en el Dataset, para ello detectamos los valores nulos que hay en nuestro Dataframe:

![image](https://drive.google.com/uc?export=view&id=1e5_ytvHSuh7CMeL-K3hZ_geQdruGUvTJ)

Eliminamos los nulos en las columnas en las que están:

![image](https://drive.google.com/uc?export=view&id=1qJQWEJOd90A87PaLxmxan4UxeWE9htqb)

Además de eso transformamos todos los valores a tipo float:

![image](https://drive.google.com/uc?export=view&id=11IaDE5n2gv-WH9t-CrDy9UB-cf2BceXY)


## IV. Exploración y visualización de los datos
Para la exploración y visualización de los datos realizamos distintos procesos, en primera instancia hacemos un head para ver la estructura de nuestro Dataframe:

![image](https://drive.google.com/uc?export=view&id=162MYEc4cYlY3vltdTicVb2qLRGr66tjA)

Después realizamos una equilibración de Dataset para poder así ajustar la distribución de las clases y poder así representarlas de la misma manera. Esto lo hacemos porque en el Dataset que utilizamos nos hemos encontrado con un desequilibrio muy notable entre los casos de fraude y los casos de no fraude. Este tipo de desequilibrio puede sesgar enormemente los rendimientos de nuestro modelo de detección de fraude y de nuestro modelo de clusterización ya que cuando existe suele favorecer a la clase que más valores tiene. En este fragmento de código podemos apreciar un conteo del número de valores de cada clase:

![image](https://drive.google.com/uc?export=view&id=11r6w-2UovMD9glvVwVsThWDFLzzJT2M4)

Aquí se puede ver el mapa de calor que hemos utilizado para observar las correlaciones existentes de nuestro Dataset:

![image](https://drive.google.com/uc?export=view&id=1YQt_MKyK6rti9XPIiRBSzbiWJyoNubmq)


## V. Preparación de los datos para los algoritmos de Machine Learning.
![image](https://drive.google.com/uc?export=view&id=1RuzEi8wgtCYrFYCs0vISaQI5kMP9xPWK)

![image](https://drive.google.com/uc?export=view&id=1fd_73OkFeu_cSfTSwWxs0Hee-xzxtsnE)

![image](https://drive.google.com/uc?export=view&id=1bET0Ph70hvPnffkEwNc65QIvNxcrOIu7)

![image](https://drive.google.com/uc?export=view&id=1r4RbR-o67dRMWIu04b2OAMc8PKSMPogK)

![image](https://drive.google.com/uc?export=view&id=12ihVD2-DgZqIelbH4nLm4yQ8IZewRU9r)

![image](https://drive.google.com/uc?export=view&id=1cAFwWPJXPHd5xs_9D7wcKlfN7lxU3B0V)

![image](https://drive.google.com/uc?export=view&id=19KPX0qgIGxZre82wAkwNNWeHsKSxoYyt)


## VI.I Entrenamiento del modelo y comprobación del rendimiento del modelo de predicción de fraude.
![image](https://drive.google.com/uc?export=view&id=19KPX0qgIGxZre82wAkwNNWeHsKSxoYyt)

![image](https://drive.google.com/uc?export=view&id=16USXpr8_gcmercnaiyOpGsd8z4XLJ34y)

![image](https://drive.google.com/uc?export=view&id=1svODOsOQpgTH5tiuKkMp0d6BZGTIaAsL)

![image](https://drive.google.com/uc?export=view&id=1VL9D6WcBqnWAFEng9oEXqlNy95xaSvQK)

![image](https://drive.google.com/uc?export=view&id=1iWpsKYozHsOawrhCHvFHowXzuWqYAcPs)

![image](https://drive.google.com/uc?export=view&id=15pHnkp7nBNh8lsrD0tbk8kKWc5UIghOW)

![image](https://drive.google.com/uc?export=view&id=1bZ5zquMvlRIqzSDIJk-BfGLAejlvxTbN)


## VI.II Entrenamiento del modelo y comprobación del rendimiento del modelo de clusterización.
![image](https://drive.google.com/uc?export=view&id=1t-lPQUipgF_RGGYtF-0kRLnGlu9KC3Sg)

![image](https://drive.google.com/uc?export=view&id=1XQ1fDcUmmAhrqexqOzAmz-_3oKLg3UpQ)

![image](https://drive.google.com/uc?export=view&id=1-Hxrc6iCxJhUvrabTyO8z1VtCQFLW9HH)

![image](https://drive.google.com/uc?export=view&id=1kIDYR9hrX8Sy2ZRr_YXFJgsYYVigFO2x)

![image](https://drive.google.com/uc?export=view&id=1XLIn-xVADGe5gRwXmEURV6cafU418_9-)


## VII. PLN


## VIII. Aplicación web


## IX. Conclusiones


## X. Bibliografía

