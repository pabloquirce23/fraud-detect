# FraudDetect
Descripción y logo


## Créditos
[Pablo Santos Quirce](https://github.com/pabloquirce23)

[Pablo Oller Pérez](https://github.com/pabloquirce23)

[Alejandro Castillo](https://github.com/pabloquirce23)


## Video


## Presentación


## I. Justificación y descripción del proyecto
Fraud Detect es una aplicación web diseñada para abordar de manera eficiente y precisa la detección de posibles fraudes bancarios.
Su funcionalidad radica en la capacidad de procesar documentos en formato PDF, extrayendo las tablas contenidas en ellos mediante su lector integrado. A partir de los datos recopilados en estas tablas, la aplicación lleva a cabo un exhaustivo análisis para identificar posibles irregularidades financieras que puedan indicar la presencia de actividades fraudulentas entre una lista de clientes. También cuenta con una sucesión de gráficas con datos que pueden ser de utilidad para el usuario, además de un chatbot implementado desde cero para solucionar cualquier tipo de duda que pueda llegar a tener el usuario.


## II. Obtención de datos.

Los modelos se han entrenado con un conjunto de datos de Kaggle, cuyo enlace se puede consultar en la sección de Bibliografía.

El conjunto de datos contiene información sobre transacciones bancarias y se compone de las siguientes variables:

* **Time:** Intervalo en segundos entre cada transacción y la transacción anterior del conjunto de datos.
* **V1 a V28:** Resultado de aplicar un Análisis de Componentes Principales (una técnica estadística que reduce la dimensionalidad de los datos) a los datos originales, para preservar la privacidad de los clientes del banco.
* **Amount:** Monto de la transacción en la moneda local.
* **Class:** Indicador de si la transacción es fraudulenta o no. 1 = Fraude, 0 = No Fraude.

El siguiente código muestra cómo se lee el conjunto de datos desde un archivo CSV:

```python
ccdf = pd.read_csv('/content/creditcard.csv')
```


## III. Limpieza de datos.

El primer paso en el procesamiento de datos implica la eliminación de valores nulos en el conjunto de datos. Identificamos los valores nulos en nuestro DataFrame con el siguiente código:

```python
ccdf.isnull().sum()
```

Posteriormente, eliminamos los valores nulos en las columnas correspondientes:

```python
ccdf.dropna(subset=["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                    "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"],
            inplace=True)
```

Además, convertimos todos los valores que no son del tipo correcto a float:

```python
ccdf['V22'] = ccdf['V22'].astype(float)
```


## IV. Exploración y visualización de los datos
Para la exploración y visualización de los datos realizamos distintos procesos, en primera instancia hacemos un head para ver la estructura de nuestro Dataframe:

![image](https://drive.google.com/uc?export=view&id=162MYEc4cYlY3vltdTicVb2qLRGr66tjA)

Para explorar y visualizar los datos, primero examinamos la estructura de nuestro DataFrame. Luego, equilibramos el conjunto de datos para ajustar la distribución de las clases y representarlas de manera equitativa. Esto es necesario debido al desequilibrio significativo entre los casos de fraude y no fraude en el conjunto de datos que estamos utilizando. Este desequilibrio puede sesgar considerablemente el rendimiento de nuestro modelo de detección de fraude y nuestro modelo de agrupación, ya que generalmente favorece a la clase con más valores. El siguiente fragmento de código muestra un recuento del número de valores de cada clase:

```python
legit = ccdf[ccdf['Class']==0]
fraud = ccdf[ccdf['Class']==1]

legit.shape()
fraud.shape()

legit = legit.sample(fraud.shape[0])
legit.shape()

ccdf['Class'].value_counts()
```

Finalmente, utilizamos un mapa de calor para observar las correlaciones existentes en nuestro conjunto de datos.

![image](https://drive.google.com/uc?export=view&id=1YQt_MKyK6rti9XPIiRBSzbiWJyoNubmq)


## V. Preparación de los datos para los algoritmos de Machine Learning.

En esta etapa, preparamos los datos para el desarrollo y entrenamiento de nuestros modelos. Primero, identificamos los valores atípicos en cada columna de nuestro conjunto de datos utilizando el siguiente código:

```python
# Lista de colores
colores = ['#FB8861', '#56F9BB', '#C5B3F9', '#F94144', '#F3722C', '#F8961E',
           '#F9C74F', '#90BE6D', '#43AA8B', '#577590', '#6D597A', '#B56576',
           '#E56B6F', '#EAAC8B', '#D6A2E8', '#B0A8B9', '#2E294E', '#EF8354',
           '#7D70BA', '#5B84B1', '#FB8861', '#56F9BB', '#C5B3F9', '#F94144',
           '#F3722C', '#F8961E', '#F9C74F', '#90BE6D', '#43AA8B', '#577590']

# Subplots
fig, axs = plt.subplots(14, 2, figsize=(15, 70))

# Bucle para crear las gráficas tipo boxplot para poder observar los outliers de todas las columnas
for i in range(1, 29):
    fraud_dist = ccdf['V'+str(i)].loc[ccdf['Class'] == 1]
    ax = axs[(i-1)//2, (i-1)%2]

    sns.boxplot(x=fraud_dist, color=colores[i-1], ax=ax)
    ax.set_title('V'+str(i)+' Distribution \n (Transacciones Fraudulentas)', fontsize=14)

plt.tight_layout()
plt.show()
```

El resultado final se puede ver en la siguiente imagen:

![image](https://drive.google.com/uc?export=view&id=1fd_73OkFeu_cSfTSwWxs0Hee-xzxtsnE)

Posteriormente, eliminamos los valores atípicos en las columnas de nuestro conjunto de datos. Para ello, observamos los límites del rango intercuartil (IQR), calculamos el rango (diferencia entre el percentil 75 y el percentil 25), el valor de corte (1,5 veces el IQR) y los límites inferiores y superiores:

```python
columnas = ['V'+str(i) for i in range(1, 29)]

# Bucle que se encarga de eliminar los outliers de todas las columnas.
for columna in columnas:
    v_fraud = ccdf[columna].loc[ccdf['Class'] == 1].values
    q25, q75 = np.percentile(v_fraud, 25), np.percentile(v_fraud, 75)
    print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
    v_iqr = q75 - q25
    print('iqr: {}'.format(v_iqr))

    v_cut_off = v_iqr * 1.5
    v_lower, v_upper = q25 - v_cut_off, q75 + v_cut_off
    print('Cut Off: {}'.format(v_cut_off))
    print(columna+' Lower: {}'.format(v_lower))
    print(columna+' Upper: {}'.format(v_upper))

    outliers = [x for x in v_fraud if x < v_lower or x > v_upper]
    print('Atributo '+columna+' Outliers para Casos de Fraude: {}'.format(len(outliers)))
    print(columna+' outliers:{}'.format(outliers))

    ccdf = ccdf.drop(ccdf[(ccdf[columna] > v_upper) | (ccdf[columna] < v_lower)].index)
    print('Numero de ocurrencias despúes de eliminar outliers: {}'.format(len(ccdf)))
    print('-' * 333)
```

Aquí se presentan los resultados de estos factores para una de las columnas:

```
Quartile 25: -6.03606299434868 | Quartile 75: -0.419200076257679
iqr: 5.616862918091001
Cut Off: 8.4252943771365
V1 Lower: -14.461357371485182
V1 Upper: 8.006094300878821
Atributo V1 Outliers para Casos de Fraude: 52
V1 outliers:[-14.4744374924863, -15.3988450085358, -14.7246270119253, -15.2713618637585, -15.8191787207718, -16.3679230107968, -16.9174682656955, -17.4677100117887, -18.0185611876771, -15.9036352020113, -16.5986647432584, -17.2751911945397, -18.474867903441, -19.1798264145873, -19.8563223334433, -20.5327510764355, -21.2091195927913, -21.8854339051741, -22.5616992591298, -23.237920244511, -23.9141008948243, -24.5902447690465, -25.2663550194138, -25.9424344479142, -27.1436784229495, -27.84818067198, -28.5242675938406, -29.2003285905744, -29.8763655139763, -30.552380043581, -15.0209806030789, -14.9703456545046, -15.1404496225073, -16.5265065691231, -18.2475132286673, -19.1397328634111, -20.9069081014654, -26.4577446501446, -26.4577446501446, -26.4577446501446, -26.4577446501446, -15.1920640113629, -16.308650062507, -17.5189091261484, -17.9762660516057, -17.5375916846763, -19.6418567335974, -22.3418888868038, -23.9847466495794, -25.825982149082, -28.2550528932108, -28.7092292541793]
Numero de ocurrencias despúes de eliminar outliers: 932
```

Una vez eliminados los valores atípicos, volvemos a visualizar las gráficas iniciales para confirmar que la mayoría de estos valores han sido eliminados.

![image](https://drive.google.com/uc?export=view&id=12ihVD2-DgZqIelbH4nLm4yQ8IZewRU9r)

Posteriormente, realizamos una visualización adicional del conjunto de datos y observamos que los datos están escalados de manera diferente en cada columna.

![image](https://drive.google.com/uc?export=view&id=1cAFwWPJXPHd5xs_9D7wcKlfN7lxU3B0V)

Para abordar este problema, es necesario estandarizar los datos. La estandarización es un proceso que transforma los datos para que tengan una media de cero y una desviación estándar de uno. Esto se realiza utilizando el siguiente código:

```python
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Este proceso asegura que todas las columnas del conjunto de datos estén en la misma escala, lo cual es esencial para muchos algoritmos de aprendizaje automático. Con los datos ahora estandarizados, estamos listos para proceder con el desarrollo y entrenamiento de nuestros modelos de aprendizaje automático.


## VI.I Entrenamiento del modelo de predicción de fraude y comprobación de su rendimiento.

En esta sección, nos enfocamos en el desarrollo y entrenamiento de nuestro modelo de predicción para la detección de fraude. Utilizamos la biblioteca Keras para construir y entrenar una red convolucional unidimensional (1D):

```python
epochs = 15
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(MaxPool1D(2)) # Aquí
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPool1D(2)) # Aquí
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train,  epochs=epochs, validation_data=(X_test, y_test), verbose=1)
```

A continuación, se muestra el rendimiento y la validación de nuestro modelo a medida que avanzan las épocas de entrenamiento:

```
Epoch 1/15
5437/5437 [==============================] - 40s 7ms/step - loss: 0.0280 - accuracy: 0.9942 - val_loss: 0.0047 - val_accuracy: 0.9995
Epoch 2/15
5437/5437 [==============================] - 39s 7ms/step - loss: 0.0066 - accuracy: 0.9991 - val_loss: 0.0046 - val_accuracy: 0.9994
Epoch 3/15
5437/5437 [==============================] - 38s 7ms/step - loss: 0.0053 - accuracy: 0.9993 - val_loss: 0.0042 - val_accuracy: 0.9995
Epoch 4/15
5437/5437 [==============================] - 39s 7ms/step - loss: 0.0048 - accuracy: 0.9993 - val_loss: 0.0039 - val_accuracy: 0.9995
Epoch 5/15
5437/5437 [==============================] - 39s 7ms/step - loss: 0.0046 - accuracy: 0.9994 - val_loss: 0.0032 - val_accuracy: 0.9996
Epoch 6/15
5437/5437 [==============================] - 38s 7ms/step - loss: 0.0041 - accuracy: 0.9995 - val_loss: 0.0034 - val_accuracy: 0.9995
Epoch 7/15
5437/5437 [==============================] - 38s 7ms/step - loss: 0.0040 - accuracy: 0.9994 - val_loss: 0.0036 - val_accuracy: 0.9996
Epoch 8/15
5437/5437 [==============================] - 39s 7ms/step - loss: 0.0040 - accuracy: 0.9995 - val_loss: 0.0038 - val_accuracy: 0.9996
Epoch 9/15
5437/5437 [==============================] - 38s 7ms/step - loss: 0.0037 - accuracy: 0.9995 - val_loss: 0.0032 - val_accuracy: 0.9996
Epoch 10/15
5437/5437 [==============================] - 39s 7ms/step - loss: 0.0034 - accuracy: 0.9995 - val_loss: 0.0029 - val_accuracy: 0.9996
Epoch 11/15
5437/5437 [==============================] - 41s 8ms/step - loss: 0.0035 - accuracy: 0.9995 - val_loss: 0.0030 - val_accuracy: 0.9995
Epoch 12/15
5437/5437 [==============================] - 37s 7ms/step - loss: 0.0032 - accuracy: 0.9995 - val_loss: 0.0029 - val_accuracy: 0.9995
Epoch 13/15
5437/5437 [==============================] - 38s 7ms/step - loss: 0.0036 - accuracy: 0.9995 - val_loss: 0.0028 - val_accuracy: 0.9995
Epoch 14/15
5437/5437 [==============================] - 38s 7ms/step - loss: 0.0031 - accuracy: 0.9995 - val_loss: 0.0029 - val_accuracy: 0.9995
Epoch 15/15
5437/5437 [==============================] - 39s 7ms/step - loss: 0.0032 - accuracy: 0.9995 - val_loss: 0.0030 - val_accuracy: 0.9995
```

Ahora desarrollamos una función para visualizar la precisión y la pérdida de nuestro modelo durante el entrenamiento y la validación a lo largo de las épocas:

```python
def plot_learning_curve(history, epoch):
  # Entrenamiento & valores de validation accuracy
  epoch_range = range(1, epoch+1)
  plt.plot(epoch_range, history.history['accuracy'])
  plt.plot(epoch_range, history.history['val_accuracy'])
  plt.title('Precisión del modelo')
  plt.ylabel('Precisión')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  # Entrenamiento & valores de validation loss
  plt.plot(epoch_range, history.history['loss'])
  plt.plot(epoch_range, history.history['val_loss'])
  plt.title('Loss del modelo')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()
```

La gráfica resultante permite observar los cambios en la precisión y la pérdida del modelo a medida que se entrena a lo largo de las épocas. Esto también nos permite detectar si hay sobreajuste o subajuste. Por ejemplo, alrededor de la época 7, se puede apreciar un subajuste. Tras varias pruebas entrenando el modelo con distintas combinaciones de capas, hemos atribuido este subajuste a la capa de tipo MaxPool1D, que es comúnmente utilizada en entrenamientos de modelos con características similares.

```python
plot_learning_curve(history, epochs)
```

![image](https://drive.google.com/uc?export=view&id=1VL9D6WcBqnWAFEng9oEXqlNy95xaSvQK)

A continuación, presentamos la estructura final de capas de nuestra red neuronal convolucional 1D:

```python
epochs = 15
model = Sequential()
model.add(Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=X_train[0].shape))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv1D(filters=64, kernel_size=2, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(units=1, activation='sigmoid'))
```

A continuación, proporcionamos un resumen de las características de nuestra red neuronal:

```python
model.summary()
```

```
Model: "sequential_2"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 conv1d_4 (Conv1D)           (None, 29, 32)            96        
                                                                 
 batch_normalization_4 (Bat  (None, 29, 32)            128       
 chNormalization)                                                
                                                                 
 dropout_6 (Dropout)         (None, 29, 32)            0         
                                                                 
 conv1d_5 (Conv1D)           (None, 28, 64)            4160      
                                                                 
 batch_normalization_5 (Bat  (None, 28, 64)            256       
 chNormalization)                                                
                                                                 
 dropout_7 (Dropout)         (None, 28, 64)            0         
                                                                 
 flatten_2 (Flatten)         (None, 1792)              0         
                                                                 
 dense_4 (Dense)             (None, 64)                114752    
                                                                 
 dropout_8 (Dropout)         (None, 64)                0         
                                                                 
 dense_5 (Dense)             (None, 1)                 65        
                                                                 
=================================================================
Total params: 119457 (466.63 KB)
Trainable params: 119265 (465.88 KB)
Non-trainable params: 192 (768.00 Byte)
_________________________________________________________________
```

Tras realizar los ajustes necesarios en nuestro modelo, volvemos a visualizar la precisión y la pérdida durante el entrenamiento y la validación a lo largo de las épocas:

```python
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train,  epochs=epochs, validation_data=(X_test, y_test), verbose=1)
```

```
Epoch 1/15
5437/5437 [==============================] - 57s 10ms/step - loss: 0.0556 - accuracy: 0.9804 - val_loss: 0.0034 - val_accuracy: 0.9995
Epoch 2/15
5437/5437 [==============================] - 57s 11ms/step - loss: 0.0049 - accuracy: 0.9993 - val_loss: 0.0029 - val_accuracy: 0.9995
Epoch 3/15
5437/5437 [==============================] - 58s 11ms/step - loss: 0.0041 - accuracy: 0.9995 - val_loss: 0.0026 - val_accuracy: 0.9995
Epoch 4/15
5437/5437 [==============================] - 55s 10ms/step - loss: 0.0039 - accuracy: 0.9995 - val_loss: 0.0024 - val_accuracy: 0.9996
Epoch 5/15
5437/5437 [==============================] - 58s 11ms/step - loss: 0.0037 - accuracy: 0.9995 - val_loss: 0.0024 - val_accuracy: 0.9996
Epoch 6/15
5437/5437 [==============================] - 54s 10ms/step - loss: 0.0036 - accuracy: 0.9995 - val_loss: 0.0027 - val_accuracy: 0.9996
Epoch 7/15
5437/5437 [==============================] - 54s 10ms/step - loss: 0.0036 - accuracy: 0.9995 - val_loss: 0.0027 - val_accuracy: 0.9996
Epoch 8/15
5437/5437 [==============================] - 57s 10ms/step - loss: 0.0034 - accuracy: 0.9995 - val_loss: 0.0027 - val_accuracy: 0.9994
Epoch 9/15
5437/5437 [==============================] - 57s 10ms/step - loss: 0.0033 - accuracy: 0.9995 - val_loss: 0.0026 - val_accuracy: 0.9996
Epoch 10/15
5437/5437 [==============================] - 55s 10ms/step - loss: 0.0030 - accuracy: 0.9995 - val_loss: 0.0027 - val_accuracy: 0.9996
Epoch 11/15
5437/5437 [==============================] - 55s 10ms/step - loss: 0.0033 - accuracy: 0.9995 - val_loss: 0.0027 - val_accuracy: 0.9996
Epoch 12/15
5437/5437 [==============================] - 56s 10ms/step - loss: 0.0031 - accuracy: 0.9995 - val_loss: 0.0023 - val_accuracy: 0.9996
Epoch 13/15
5437/5437 [==============================] - 56s 10ms/step - loss: 0.0031 - accuracy: 0.9995 - val_loss: 0.0024 - val_accuracy: 0.9996
Epoch 14/15
5437/5437 [==============================] - 56s 10ms/step - loss: 0.0031 - accuracy: 0.9995 - val_loss: 0.0026 - val_accuracy: 0.9995
Epoch 15/15
5437/5437 [==============================] - 54s 10ms/step - loss: 0.0030 - accuracy: 0.9995 - val_loss: 0.0025 - val_accuracy: 0.9996
```

Al observar la representación gráfica, podemos confirmar que el ajuste que anteriormente perjudicaba la precisión de nuestro modelo ya no está presente.

```python
plot_learning_curve(history, epochs)
```

![image](https://drive.google.com/uc?export=view&id=16sajE8s_rHvhw0lomre-TjooQp88xGAM)

Esto indica que nuestro modelo ha mejorado y ahora es capaz de realizar predicciones con mayor precisión. La eliminación del ajuste es un indicativo de que nuestro modelo está bien ajustado y es capaz de generalizar bien a partir de los datos de entrenamiento. Esto es crucial para garantizar que nuestro modelo sea efectivo al detectar fraudes en transacciones no vistas previamente.


## VI.II Entrenamiento del modelo de clusterización y comprobación de su rendimiento.

El siguiente paso es el modelado de clusterización, que agrupa los registros proporcionados en uno de los siguientes grupos:

* 0: Desarrollo Personal (Personal Growth)
* 1: Ocio (Leisure)
* 2: Necesidades Básicas (Basic Necessities)
* 3: Préstamos (Loans)
* 4: Inversiones (Investments)

Para lograr agrupaciones coherentes, hemos introducido un nuevo atributo, ‘Median’, calculado como la media de los valores de las columnas V1 a V28:

```python
ccdf['Median'] = ccdf[["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                       "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                       "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]].mean(axis=1)
```

Existen múltiples métodos para determinar el número óptimo de clusters. En nuestro caso, hemos optado por el método Elbow, que nos proporciona una medida de la eficacia de la agrupación realizada por el algoritmo K-Means.

A continuación, presentamos el código que nos permite identificar el número óptimo de clusters:

```python
# Función para buscar el número óptimo de clusters
def optimizacion_cluster(data, max_clstr):
  means = []
  inertias = []

  for i in range(1, max_clstr):
    kmeans= KMeans(n_clusters=i)
    kmeans.fit(data)

    means.append(i)
    inertias.append(kmeans.inertia_)

  # Generación de la gráfica
  fig = plt.subplots(figsize=(10, 5))
  plt.plot(means, inertias, 'o-')
  plt.xlabel('Número de Clusters')
  plt.ylabel('Inercia')
  plt.grid(True)
  plt.show()
```

La gráfica resultante muestra cómo la inercia (la suma de las distancias al cuadrado al centro del cluster más cercano) disminuye a medida que aumentamos el número de clusters. Observamos un equilibrio a partir de la división en 2 clusters.

```python
optimizacion_cluster(ccdf[["Median", "Amount"]], 20)
```

![image](https://drive.google.com/uc?export=view&id=1-Hxrc6iCxJhUvrabTyO8z1VtCQFLW9HH)

```python
for i in range(1, 11):
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(ccdf[['Median', 'Amount']])
  ccdf[f'KMeans_{i}'] = kmeans.labels_
```

A continuación, aplicamos el algoritmo KMeans 10 veces (número seleccionado basándonos en la gráfica anterior, ya que la inercia se estabiliza a partir de los 10 clusters) y añadimos una columna a nuestro DataFrame con la información del grupo al que pertenece cada fila en cada aplicación del algoritmo.

```python
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20,10))

for i, ax in enumerate(fig.axes, start=1):
  ax.scatter(x=ccdf['Median'], y=ccdf['Amount'], c=ccdf[f'KMeans_{i}'])
  ax.set_title(f'Número Clusters: {i}')
```

Finalmente, comparamos gráficamente los resultados de KMeans para diferentes números de clusters y decidimos quedarnos con una agrupación de 5 clusters.

![image](https://drive.google.com/uc?export=view&id=1kIDYR9hrX8Sy2ZRr_YXFJgsYYVigFO2x)

Después de aplicar el código anterior, nuestro conjunto de datos contiene varias columnas que ya no son necesarias. Por lo tanto, procedemos a eliminarlas para limpiar el conjunto de datos:

```python
drp_clmns = ['KMeans_1', 'KMeans_2', 'KMeans_3', 'KMeans_4',
             'KMeans_5', 'KMeans_6', 'KMeans_7', 'KMeans_8',
             'KMeans_9', 'KMeans_10']

ccdf.drop(columns=drp_clmns, inplace=True)
```

A continuación, implementamos el algoritmo final con el número de clusters que seleccionamos previamente.

```python
kmeans = KMeans(n_clusters=5)
kmeans.fit(ccdf[['Median', 'Amount']])
ccdf[f'KMeans_{5}'] = kmeans.labels_
```

Este es el aspecto final del conjunto de datos después de la limpieza y la implementación del algoritmo de agrupación.

![image](https://drive.google.com/uc?export=view&id=1ITCax33yYYf_CHFsX-3Znsd6ziXY71Gl)


## VII. PLN

Para abordar el desafío del Procesamiento del Lenguaje Natural, hemos optado por desarrollar un chatbot desde cero. Consideramos que este es un complemento valioso para nuestra aplicación, ya que proporciona a los usuarios un asistente personal accesible con solo dos clics para resolver cualquier duda que puedan tener sobre la información proporcionada o sobre conceptos relacionados con el sector bancario.

![image](https://drive.google.com/uc?export=view&id=15H1sPLH-fxr4-LlkPbadj9HykIwm37gv)

A continuación, se detalla el funcionamiento del chatbot.

Gracias a la carpeta Secrets de StreamlitCloud, podemos proteger y utilizar nuestra clave API de OpenAI de manera segura:

```python
openai.api_key = st.secrets["OPENAI_API_KEY"]
```

El modelo utilizado para desarrollar el chatbot es el siguiente:

```python
if "openai_model" not in st.session_state:
    # 00475-AEDF-52510-2
    st.session_state["openai_model"] = "gpt-3.5-turbo"
```

A continuación, se presenta la estructura de nuestro chatbot.

Inicializamos el historial del chat si aún no se ha hecho:

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

Este bucle muestra todos los mensajes en el historial del chat:

```python
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
```

Este código recoge la entrada del usuario, la muestra como mensaje en la interfaz de usuario y la añade al historial del chat. Luego, genera y visualiza la respuesta del chatbot utilizando la función `ChatCompletion.create` de la API de OpenAI. La respuesta se genera en tiempo real gracias a la opción `stream=True`. Finalmente, añade la respuesta del chatbot al historial del chat y la muestra en la interfaz de usuario.

```python
if prompt := st.chat_input("Escriba aquí su consulta"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
                    message_placeholder.markdown(full_response + "| ")
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})
```


## VIII. Aplicación web

La aplicación web se ha hecho mediante el framework de Streamlit que está diseñado para facilitar al desarrollador y al analista la creación de interfaces interactivas con el lenguaje Python. Los fuentes del proyecto están en este ![enlace](https://github.com/pabloquirce23/fraud-detect/tree/main/src).

### Lógica de la aplicación

#### Inicio

Aquí va la documentación de Pablo Oller (botón de carga de tablas PDF)

#### Predicción

En esta sección, la aplicación muestra los resultados de la predicción de fraude y la clusterización de las transacciones. Para ello, se visualiza el DataFrame generado por la función de lectura de tablas.

![image](https://drive.google.com/uc?export=view&id=14eGh_eQZUqf1TJ0d340PT-eQY9Pw1T9H)

A continuación, se detalla el proceso:

Primero, se verifica si existe un DataFrame en el estado de la sesión de Streamlit y si no está vacío. En caso afirmativo, se accede al DataFrame.

```python
if 'df' in st.session_state and not st.session_state['df'].empty:
  df = st.session_state['df']  # Acceso directo al DataFrame
```

Luego, se seleccionan las columnas ‘Median’ y ‘Amount’ del DataFrame para el modelo de clustering.

```python
df_clustering = df[['Median', 'Amount']]
```

Posteriormente, se convierte el DataFrame a un tensor para su uso con TensorFlow. Se seleccionan solo las columnas originales.

```python
df_tensor = tf.convert_to_tensor(df[columnas].values, dtype=tf.float32)  # Solo selecciona las columnas originales aquí
```

El DataFrame de clustering también se convierte a un tensor.

```python
df_clustering_tensor = tf.convert_to_tensor(df_clustering.values, dtype=tf.float32)
```

Si el DataFrame no está vacío, se aplica el modelo de predicción al tensor y se guarda el resultado en la columna ‘Class’ del DataFrame. Además, si el DataFrame no está vacío, también se aplica el modelo de clustering al tensor de clustering y se guarda en la columna ‘Cluster’ del DataFrame.

```python
if not df.empty:
  df['Class'] = modelo.predict(df_tensor)
  df['Cluster'] = modelo_clustering.predict(df_clustering_tensor)
```

Se crea un nuevo DataFrame para almacenar los resultados.

```python
results_df = pd.DataFrame(columns=["Detección Fraude", "Cluster"])
```

Finalmente, se recorre el DataFrame y se añaden los resultados al DataFrame de resultados.

```python
for i in range(len(df)):
  new_row = pd.DataFrame({"Detección Fraude": ["NO FRAUDE ✅" if df['Class'][i] == 0 else "FRAUDE ❌"], 
                          "Cluster": [cluster_labels[df['Cluster'][i]]]})
  results_df = pd.concat([results_df, new_row], ignore_index=True)
```

##### Visualización de los resultados

Para facilitar la exploración y análisis de los datos, se han generado diversas gráficas con enfoques distintos.

La primera gráfica establece una relación entre las predicciones y los clusters.

![image](https://drive.google.com/uc?export=view&id=1Lid5pahmiNy7jPYF091HFoxTCP_KfERw)

Aquí mostramos el como funciona esta gráfica:

Primero, se ordenan los clusters de menor a mayor y se mapean a las etiquetas preferidas.

```python
clusters = sorted(df['Cluster'].unique())
cluster_labels_2 = {0: "PG", 1: "LS", 2: "BN", 3: "LN", 4: "IN"}
```

Se crea una gráfica circular para cada cluster, donde se calcula el porcentaje de fraude y no fraude. Los colores ‘darkblue’ y ‘lightblue’ representan ‘Fraude’ y ‘No Fraude’, respectivamente.

```python
fig, axs = plt.subplots(1, len(clusters), figsize=(10, 15))
colors = ['darkblue', 'lightblue']
```

Para cada cluster, se filtra el DataFrame correspondiente, se cuentan los casos de fraude y se calculan los porcentajes. Luego, se crea la gráfica circular y se ajusta la posición del título y la leyenda de manera alternativa.

```python
for i, cluster in enumerate(clusters):
    df_cluster = df[df['Cluster'] == cluster]
    fraud_counts = df_cluster['Class'].value_counts()
    not_fraud_percentage = fraud_counts.get(0, 0) / fraud_counts.sum() * 100
    fraud_percentage = 100 - not_fraud_percentage
    labels = [f'Fraud {fraud_percentage:.1f}%', f'Not Fraud {not_fraud_percentage:.1f}%']
    axs[i].pie([fraud_percentage, not_fraud_percentage], startangle=90, colors = colors)
    if i % 2 == 0:
        axs[i].set_title(cluster_labels_2[cluster], y=1.1)
        axs[i].legend(labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)
    else:
        axs[i].set_title(cluster_labels_2[cluster], y=-0.1)
        axs[i].legend(labels, loc='lower center', bbox_to_anchor=(0.5, 1.05), fancybox=True, shadow=True, ncol=5)
```


Como parte de los componentes gráficos que se presentan al realizar la predicción de los datos, esta gráfica es una de las disponibles en la página **Predicción** para la evaluación de la predicción. La segmentación por colores indica los distintos productos contratados por los clientes la entidad que suministra los datos.

![image](https://drive.google.com/uc?export=view&id=1u0W_4uz_xYzgk5aYxAjFNwN87yCw1KIP)<br>
*grafica_distribucion_scatter*

```python
# Crea un gráfico de dispersión para visualizar los clusters
plt.figure(figsize=(10, 6))

for cluster in df['Cluster'].unique():

    # Filtra los datos por cluster
    cluster_data = df[df['Cluster'] == cluster]

    # Plotea los datos con un color diferente para cada cluster
    plt.scatter(cluster_data['Median'], cluster_data['Amount'], label=f'{cluster_labels_2[cluster]}')

plt.title('Distribución de Transacciones por Clusters')
plt.xlabel('Mediana de V1-V28')
plt.ylabel('Amount')
plt.legend()
st.pyplot(plt)
```

#### Eto'o Bot

La descripción del rol del chatbot y la explicación de su funcionalidad se han expuesto en el apartado VII (Procesamiento del Lenguaje Natural).

### Lógica de la interfaz

#### Menú Principal


En este área se localiza el selector de cambio de pantallas que permite al usuario moverse entre las distintas pantallas disponibles.

![image](https://drive.google.com/uc?export=view&id=1pXgar6GKGOpPAFPsdjAsUH5vUZJdL91j)<br>
*menu_principal_selector*

En app.py dentro de la función main() se crean las variables st.session_state de los elementos 'page' para el control de la página seleccionada y 'prediction_generated' para el estado que se utiliza para emitir el mensaje *Predicción completada*. El botón **Genera Predicción** permite que al cambiar de página los datos aparezcan allí, ya que ninguna página hace uso de carga asíncrona.

```python
# Contenido de la página de Inicio
if st.session_state['page'] == 'Inicio':
    show_home_page()

# Contenido de la página de Predicción
elif st.session_state['page'] == 'Predicción':
    if st.session_state.get('df') is not None and not st.session_state['df'].empty:
        show_prediction_page()  # Ahora accede a `st.session_state['df']` internamente
    else:
        st.error('Por favor, carga los datos en la página de Inicio primero.')

# Contenido de otras páginas
elif st.session_state['page'] == 'Análisis Exploratorio y Modelo':
    show_analysis_page()

# Contenido de la página de Eto'o Bot
elif st.session_state['page'] == 'Eto\'o Bot':
        show_etoobot_page()

# Botón para cambiar a la página de Predicción
if st.session_state['page'] not in ['Predicción', 'Análisis Exploratorio y Modelo', 'Eto\'o Bot']:
    if st.button('Genera Predicción'):
        if st.session_state.get('df') is not None and not st.session_state['df'].empty:
            st.session_state['prediction_generated'] = True
            st.session_state['page'] = 'Predicción'  # Esto debería actualizar el selectbox automáticamente
        else:
            st.error('Por favor, carga los datos en la página de Inicio primero.')

# Mensaje de estado del procesamiento y predicción
if st.session_state['page'] == 'Predicción' and st.session_state['prediction_generated']:
    st.success('Predicción completada.')
```


La modularidad aplicada al proyecto facilita la lectura, el mantenimiento por otros programadores, la escalabilidad y reutilización del código. Aquí hay un esquema del árbol de carpetas y ficheros que componen la aplicación.

![image](https://drive.google.com/uc?export=view&id=1GBR3XheM0BsdXsGZDU8DlHrd13Ph94yH)<br>
*project_tree*

#### Pantalla de inicio

El punto de entrada es la pantalla de inicio desde la que se aporta una introducción sobre la utilidad que proporciona al público. Seleccionados los extractos en formato pdf que se van a analizar se inicia el proceso de predicción pulsando el botón **"Genera Predicción"** tras lo que se avisará con un mensaje *"Predicción completada"*.

![image](https://drive.google.com/uc?export=view&id=1H5XA4F0JCoSLDkunvyfOkK4zuEWgXeJD)<br>
*genera_prediccion_aviso*

#### Pantalla de predicción

El aspecto de la página de Predicción muestra la información generada por los componentes de Streamlit para mostrar la información de interés para el usuario.

![image](https://drive.google.com/uc?export=view&id=1KroUYNdk_FmDjdZ0atYPUcf2GcMze5k3)<br>
*prediccion_basic_preview*

La función show_prediction_page del fichero ![prediction_page.py](https://github.com/pabloquirce23/fraud-detect/blob/main/src/prediction_page.py) comprueba que ha recibido mediante st.session_state el dataframe importado desde la pantalla de inicio.

```python
def show_prediction_page():
    # Titulo de la aplicación
    st.markdown(custom_title('Fraud-Detect'), unsafe_allow_html=True)

    st.subheader("Predicciones de Fraude")

    # Verifica que el DataFrame exista y no esté vacío
    if 'df' in st.session_state and not st.session_state['df'].empty:
        df = st.session_state['df']  # Acceso directo al DataFrame
```


#### Tema de personalización

La posibilidad de aplicar un tema personalizado de Streamlit requiere del fichero "./streamlit/config.toml" que debe ubicarse en la raíz del proyecto. Esto aplica estilos sin tener que convertir componentes nativos de Streamlit como por ejemplo *st.dataframe* a tablas en html-css. Los componentes nativos de Streamlit ofrecen muchas posibilidades, como hacer zoom, cambiar el tamaño, busquedas, ordenaciones, etc.


![image](https://drive.google.com/uc?export=view&id=1vQXWv0UFTwlJcjnCFz8h4-Vrvs0inS9M)<br>
*custom_theme_config_toml*

Estas especificaciones son las que nos ha permitido modificar los colores predeterminados de Streamlit y mejorar su apariencia.
```
[theme]
primaryColor="#1d3557"
backgroundColor="#457b9d"
secondaryBackgroundColor="#003049"
textColor="#f1faee"
```

#### Pantalla de Análisis Exploratorio y Modelo

En línea con los principios de la explicabilidad respecto a los sistemas de IA que generan tales predicciones en base a modelos, la aplicación **Fraud-Detect** ofrece una aproximación de cómo se realiza el *Análisis Exploratorio, Estudio y Desarrollo de los Modelos* integrados en esta aplicación. Esta visualización se consigue haciendo una importación y procesado de cada una de los componentes que contiene un notebook de Jupyter: celdas markdown, celdas de código, resultados en formato texto, resultados gráficos o imágenes, tablas o dataframes, etc.

![image](https://drive.google.com/uc?export=view&id=1Gm3jWfyb7Lm-b-vxHndATrLWayGEJDOS)<br>
*nbformat_jupyter_1*

La función **show_analysis_page** del fichero ![pages.py](https://github.com/pabloquirce23/fraud-detect/blob/main/src/pages.py) invoca al método cargar_cuaderno_jupyter que lee el fichero .ipynb indicado en la ruta, y este invoca a su vez a mostrar_cuaderno_jupyter que realiza el procesado del notebook.
```python
def show_analysis_page():
    # Titulo de la aplicación
    st.markdown(custom_title('Fraud-Detect'), unsafe_allow_html=True)

    st.subheader("Análisis Exploratorio y Modelo")
    # Aquí puedes añadir más contenido para esta página

    # Cargar el cuaderno Jupyter
    nb_path = 'src/static/notebooks/modelo_tfm.ipynb'
    with st.spinner('Cargando análisis exploratorio y Modelo...'):
        # Cargar y mostrar el contenido pesado aquí
        nb = cargar_cuaderno_jupyter(nb_path)

        # Especificar el inicio y el final
        num_celda_inicio = 1  # Ajusta este valor según sea necesario
        num_celda_final = 33  # Ajusta este valor según sea necesario

        # Mostrar el rango especificado de celdas del cuaderno en Streamlit
        mostrar_cuaderno_jupyter(nb, num_celda_inicio)
```


La función mostrar_cuaderno_jupyter procesa las celda y según el tipo de contenido trata la información para adecuarla, como es el caso de las imagenes que requieren un deserilizado mediante la librería base64.
```python
def cargar_cuaderno_jupyter(path):
    with open(path, 'r', encoding='utf-8') as f:
        return nbformat.read(f, as_version=4)

def mostrar_cuaderno_jupyter(nb, num_celda_inicio=0, num_celda_final=None):
    for i, cell in enumerate(nb.cells):
        # Si se especificó un número de celda final y se alcanza, detener el bucle
        if num_celda_final is not None and i > num_celda_final:
            break
        # Continuar si el índice de la celda actual aún no ha alcanzado el número de celda de inicio
        if i < num_celda_inicio:
            continue
        
        if cell.cell_type == 'markdown':
            st.markdown(cell.source)
        elif cell.cell_type == 'code':
            st.code(cell.source, language='python')
            for output in cell.outputs:
                if output.output_type == 'execute_result' or output.output_type == 'display_data':
                    if 'text/plain' in output.data:
                        st.text(output.data['text/plain'])
                    if 'image/png' in output.data:
                        base64_img = output.data['image/png']
                        img_bytes = base64.b64decode(base64_img)
                        st.image(img_bytes, use_column_width=True)
                    if 'text/html' in output.data:
                        # Llamar a limpiar_html solo para contenido HTML
                        cleaned_html = limpiar_html(output.data['text/html'])
                        st.markdown(cleaned_html, unsafe_allow_html=True)
                        # st.markdown(output.data['text/html'], unsafe_allow_html=True)
                elif output.output_type == 'stream':
                    st.text(output.text)
                elif output.output_type == 'error':
                    st.error('\n'.join(output.traceback))
```

#### Pantalla de Eto'o Bot

![image](https://drive.google.com/uc?export=view&id=11paefrcw7Kmol00uzPoTJjV-kfW87Nlo)<br>
*etoo_bot_page*

La función **show_etoobot_page** del fichero ![etoo_bot.py](https://github.com/pabloquirce23/fraud-detect/blob/main/src/etoo_bot.py) es llamada cuando el usuario selecciona desde la función main() de **app.py** el acceso a esta herramienta.

```python
# Contenido de la página de Eto'o Bot
elif st.session_state['page'] == 'Eto\'o Bot':
        show_etoobot_page()
```

La descripción del rol del chatbot y la explicación de su funcionamiento se encuentran en el apartado VII.

#### Módulo de Componentes

En ![components.py](https://github.com/pabloquirce23/fraud-detect/blob/main/src/components.py) se almacena todo el contenido HTML que es utilizado en la aplicación. Así se facilita la localización del código relacionado con el lenguaje de marcas.
```python
from static.styles.css_styles import *

def custom_footer():
    html_content = (
    FOOTER_STYLE +
    "<div class=\"custom-footer\">" +
        "<p>Creadores:</p>" +
        "<a href=\"https://www.linkedin.com/in/pablo-oller-perez-7995721b2\" target=\"_blank\">Pablo Oller Pérez</a><br>" +
        "<a href=\"https://github.com/pabloquirce23\" target=\"_blank\">Pablo Santos Quirce</a><br>" +
        "<a href=\"https://github.com/acscr44\" target=\"_blank\">Alejandro Castillo Carmona</a>" +
    "</div>"
    )
    return html_content

def custom_width():
    return WIDTH_STILE

def description():
    html_content = f"""
    <div>
        <p><strong>Fraud Detect</strong> es una aplicación web diseñada para abordar de manera eficiente y precisa la detección de 
        posibles fraudes bancarios.<br>
        Su funcionalidad radica en la capacidad de procesar documentos en formato PDF, extrayendo las tablas contenidas en ellos 
        mediante su lector integrado. 
        A partir de los datos recopilados en estas tablas, la aplicación lleva a cabo un exhaustivo análisis para identificar 
        posibles irregularidades financieras que puedan indicar la presencia de actividades fraudulentas entre una lista de clientes.<br>
        Además muestra una sucesión de gráficas con datos que pueden ser de utilidad para el usuario.
        </p>
        <br><br>
    </div>
    """.strip().replace('\n', '')
    return html_content
```

#### Módulo de Estilos

En ![css_styles.py](https://github.com/pabloquirce23/fraud-detect/blob/main/src/css_styles.py) reune toda la colección de personalización CSS.
```python
FOOTER_STYLE = """
    <style>
        .custom-footer {
            bottom: 0;
            padding: 1rem;
            margin-top: 35rem;
            font-size: 16px;
            text-align: center;    
            width: 100%;
            border-top: 1px solid #aaa;     
        }

        .custom-footer p {
            margin: 0;
            color: #666;
        }
    </style>
    """
```




## IX. Conclusiones


## X. Bibliografía

