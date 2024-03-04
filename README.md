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

```
legit = ccdf[ccdf['Class']==0]
fraud = ccdf[ccdf['Class']==1]

legit.shape()
fraud.shape()

legit = legit.sample(fraud.shape[0])
legit.shape()

ccdf['Class'].value_counts()
```

Aquí se puede ver el mapa de calor que hemos utilizado para observar las correlaciones existentes de nuestro Dataset:

![image](https://drive.google.com/uc?export=view&id=1YQt_MKyK6rti9XPIiRBSzbiWJyoNubmq)


## V. Preparación de los datos para los algoritmos de Machine Learning.

Ahora pasamos a la preparación de los datos con referencia al desarrollo y entrenamiento de nuestros modelos. Para ello vamos a observar los valores átipicos de cada columna de nuestro Dataset. Esto se ha realizado mediante el siguiente código:

```
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

Y aquí podemos ver el resultado final:

![image](https://drive.google.com/uc?export=view&id=1fd_73OkFeu_cSfTSwWxs0Hee-xzxtsnE)

Ahora pasamos al código con el que eliminamos los valores átipicos de las columnas de nuestro Dataset. En él se puede apreciar que hacemos una observación de varios factores relevantes para detectar correctamente estos valores. Observamos los límites del rango intercuartil (IQR), calculamos el rango (diferencia entre el percentil 75 y el percentil 25), el valor de corte (1,5 veces el IQR) y los límites inferiores y superiores:

```
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

Aquí pongo los resultados de todos estos factores de una de las columnas:

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

Después volvemos a mostrar las mismas gráficas usadas anteriormente para ver como han desaparecido la mayoría de los valores atípicos:

![image](https://drive.google.com/uc?export=view&id=12ihVD2-DgZqIelbH4nLm4yQ8IZewRU9r)

Ahora haciendo otra pequeña visualización al Dataset podemos observar que los datos están escalados de formas totalmente diferentes las unas de las otras:

![image](https://drive.google.com/uc?export=view&id=1cAFwWPJXPHd5xs_9D7wcKlfN7lxU3B0V)

Debido a ello nos tocará estandarizar los datos:

```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

Después de estos procesos aplicamos unas pocas transformaciones más y ya tendremos los datos listos para el desarrollo de nuestros modelos.


## VI.I Entrenamiento del modelo de predicción de fraude y comprobación de su rendimiento.

```
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

```
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

```
plot_learning_curve(history, epochs)
```

![image](https://drive.google.com/uc?export=view&id=1VL9D6WcBqnWAFEng9oEXqlNy95xaSvQK)

```
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

```
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

```
plot_learning_curve(history, epochs)
```

![image](https://drive.google.com/uc?export=view&id=16sajE8s_rHvhw0lomre-TjooQp88xGAM)

## VI.II Entrenamiento del modelo de clusterización y comprobación de su rendimiento.

```
ccdf['Median'] = ccdf[["V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
                       "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19",
                       "V20", "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28"]].mean(axis=1)
```

```
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

```
optimizacion_cluster(ccdf[["Median", "Amount"]], 20)
```

![image](https://drive.google.com/uc?export=view&id=1-Hxrc6iCxJhUvrabTyO8z1VtCQFLW9HH)

```
for i in range(1, 11):
  kmeans = KMeans(n_clusters=i)
  kmeans.fit(ccdf[['Median', 'Amount']])
  ccdf[f'KMeans_{i}'] = kmeans.labels_
```

```
fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(20,10))

for i, ax in enumerate(fig.axes, start=1):
  ax.scatter(x=ccdf['Median'], y=ccdf['Amount'], c=ccdf[f'KMeans_{i}'])
  ax.set_title(f'Número Clusters: {i}')
```

![image](https://drive.google.com/uc?export=view&id=1kIDYR9hrX8Sy2ZRr_YXFJgsYYVigFO2x)

```
drp_clmns = ['KMeans_1', 'KMeans_2', 'KMeans_3', 'KMeans_4',
             'KMeans_5', 'KMeans_6', 'KMeans_7', 'KMeans_8',
             'KMeans_9', 'KMeans_10']

ccdf.drop(columns=drp_clmns, inplace=True)
```

```
kmeans = KMeans(n_clusters=5)
kmeans.fit(ccdf[['Median', 'Amount']])
ccdf[f'KMeans_{5}'] = kmeans.labels_
```

![image](https://drive.google.com/uc?export=view&id=1ITCax33yYYf_CHFsX-3Znsd6ziXY71Gl)


## VII. PLN


## VIII. Aplicación web


## IX. Conclusiones


## X. Bibliografía

