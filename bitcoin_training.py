import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as ply
import plotly.express as px
import os
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *
import plotly.graph_objects as go
from IPython.display import clear_output
from plotly.subplots import make_subplots
import plotly.express as px

def get_dataframe(show=False):
  # Descarga los datos desde GitHub.
  # Toma la referencia de tiempo, y el valor de apertura de la divisa.
  # Los datos se organizan en x entradas valor de la divisa y Y etiquetas o siguiente valor de la divisa.
  # Se eliminan los primeros valores ya que estaban en ceros
  
  if not os.path.exists('BTCUSD_1hr.csv'):
    os.system('wget https://raw.githubusercontent.com/SmartAppsTech/Datasets/main/BTCUSD_1hr.csv')
  df=pd.read_csv('/content/BTCUSD_1hr.csv')
  inf=pd.DataFrame()                                          #Crea el dataframe
  inf['Date'] = np.array(pd.to_datetime(df["Date"][::-1]))    #Toma la el registro de tiempo 
  inf['y_value']=np.array(df.loc[:,"Open"][::-1])             #Toma el valor de apertura de la divisa
  step=(inf.loc[:,'y_value']).shift()                         #Se genera una nueva variable con los valores desplazados en una posición
  inf['x_0']=step                                             #Se crea la variable de entrada del conjunto de datos  
  inf.fillna(0, inplace=True)                                 #Se eliminan los valores vacios
  inf=inf.drop([0,1,2])                                       #Se eliminan las primeras filas (Estaban en ceros)  
  if show: print(inf.head())                                  #Muestra el encabezado de los datos  
  return inf

def get_data(frame_in, width=1, split=0.8):
  # Función para generar los datos de entrenamiento y de prueba
  # La función crea datos organizados de la siguiente manera (Batch, times, features)
  # El número de puntos de tiempo (times) depende del ancho de la ventana que se desea tomar para las predicciones
  # Advertencia, el orden es sumamente importante (Batch, times, features), times---->x-4, x-3, x-2, x_1, x_0 

  frame=frame_in.copy()
  for i in range(width-1):
    frame['x'+str(-(i+1))]=np.array([0,*frame.loc[:,frame.columns[-1]]][:-1])  # Crear las nuevas posiciones de tiempo desplanzado el vector de entrada original
  out=frame.iloc[width-1:,:]                                                   # Se eliminan los valores correspondientes a los desplazamientos

  #dividir datos
  data=np.array(out)[:,1:]                                                     # se las xs y las ys para generar los datos de entrenamiento y prueba
  train_p, test_p =data[:int(len(out)*split)], data[int(len(out)*split):]      # Se generan las particiones dependiendo del valor de división establecido (split) 

  #Normalizar
  scaler=MinMaxScaler()                                                        # operador para escalar los valores
  train=scaler.fit_transform(train_p)                                          # Escalado para el maximo y minimo de los datos de entrenamiento
  test =scaler.transform(test_p)                                               # Se ajustan los datos de prueba al escalo de los datos de entrenamiento  

  #Separar los datos en diferentes variables 
  y_train = train[:,0]   #Etiquetas                                 
  x_train = train[:,1:]  #Entrenamiento
  # test data
  y_test = test[:,0]     #Etiquetas prueba 
  x_test = test[:,1:]    #datos prueba 

  # Organizar la forma de los datos para las redes.
  # forma x_train---> (batch, times, features). Son necesarias 3 dimensiones
  # forma y_train---> (batch, labels).          Son necesarias 2 dimensiones

  x_train=x_train.reshape((np.shape(x_train)[0],1,width))
  y_train=y_train.reshape((np.shape(y_train)[0],1))
  x_test =x_test.reshape((np.shape(x_test)[0],1,width))
  y_test =y_test.reshape((np.shape(y_test)[0],1))

  #Los datos de entrada deben estar organizados temporalmente crecientes times---->[x-4, x-3, x-2, x_1, x_0] 
  return (np.rot90(x_train, axes=(1,2)), y_train), (np.rot90(x_test, axes=(1,2)), y_test)

def prediction_points(model, x_test, points=1):
  # Predice el número de puntos de tiempo establecidos por points.
  # Los cada punto nuevo predecido se agregar a los datos de entrada y predice hasta el numero de puntos points

  eval=x_test                                                                             #Datos de prueba orginal
  for i in range(points):                                                                 #Bucle para la predicción  
    y_hat=model.predict(eval)                                                             #Predecir valores   
    last=np.array([*eval[-1,:,0], *y_hat[-1,-1]][1:]).reshape((1,*np.shape(eval)[-2:]))   #Agregar a la última secuencia de tiempos (ventanas)
    eval=np.concatenate((eval, last))                                                     #Agregar ventana a los datos de prueba
  return y_hat

def show_results(x_test, y_test, hat, points, show_all=True):
  # función que compara entre los datos de entrada, valores reales y predicciones.
  # La función puede mostrar todos los puntos predecidos o los últimos 100.
  
  # Organización del dataframe para la visualización de los datos.
  # El bucle se usa para agregar el corrimiento por puntos de tiempo tomados para predecir.
  out=pd.DataFrame()
  sh=np.shape(x_test)
  for i in range(sh[-2]):      
    out['Input_'+str(-i)]=[*x_test[:,sh[-2]-i-1,0], *[np.nan for j in range(points)]]
  if points==1:
    out['Value']=[np.nan, *y_test[:,0]]
  else:
    out['Value']=[np.nan, *y_test[:,0], *[np.nan for j in range(points-1)]]
  out['Predic']=[np.nan, *hat[:,0,0]]
  out['Times']=np.arange(sh[0]+points)

  # Cortar datos para o mostrarlos todos
  if not show_all:
    out=out.iloc[-100:,:]

  #Gráfica de variables organizadas en el dataframe
  fig = go.Figure()
  fig.add_trace(go.Scatter(x=out['Times'], y=out['Input_0'], mode='lines+markers', name='Input'))
  fig.add_trace(go.Scatter(x=out['Times'], y=out['Value'], mode='markers', name='Value'))
  fig.add_trace(go.Scatter(x=out['Times'], y=out['Predic'], mode='markers', name='Prediction'))
  fig.show()


####-------------------------------------------Modelos
def linear_m(in_shape):
  inputs = tf.keras.Input(shape=in_shape)
  x=tf.keras.layers.Dense(units=1)(inputs)
  linear = tf.keras.Model(inputs, x)
  linear.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
  return linear

def MLP(in_shape):
  inputs = tf.keras.Input(shape=in_shape)
  x=tf.keras.layers.Dense(units=64, activation='relu')(inputs)
  x=tf.keras.layers.Dense(units=64, activation='relu')(x)
  x=tf.keras.layers.Dense(units=1)(x)
  md = tf.keras.Model(inputs, x)
  md.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
  return md

def Convolutional_m(in_shape):
  inputs = tf.keras.Input(shape=in_shape)
  x=tf.keras.layers.Conv1D(filters=32, kernel_size=(3,), activation='relu')(inputs)
  x=tf.keras.layers.Dense(units=32, activation='relu')(x)
  x=tf.keras.layers.Dense(units=1)(x)
  md = tf.keras.Model(inputs, x)
  md.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
  return md

def LSTM_m(in_shape):
  inputs = tf.keras.Input(shape=in_shape)
  x=tf.keras.layers.LSTM(32, return_sequences=True)(inputs)
  x=tf.keras.layers.Dense(units=1)(x)
  md = tf.keras.Model(inputs, x)
  md.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
  return md

def GRU_m(in_shape):
  inputs = tf.keras.Input(shape=in_shape)
  x=tf.keras.layers.GRU(32, return_sequences=True)(inputs)
  x=tf.keras.layers.Dense(units=1)(x)
  md = tf.keras.Model(inputs, x)
  md.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
  return md

def AR_LSTM_m(in_shape):
  inputs = tf.keras.Input(shape=(20,10))
  x=tf.keras.layers.RNN(tf.keras.layers.LSTMCell(32), return_state=True)(inputs)
  x=tf.keras.layers.Dense(1)(x)
  md = tf.keras.Model(inputs, x)
  md.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
  return md

def Fusion_CNN_LSTM(in_shape):
  inputs = tf.keras.Input(shape=in_shape)
  x=tf.keras.layers.Conv1D(filters=32, kernel_size=5, activation='relu')(inputs)
  x=tf.keras.layers.LSTM(64, return_sequences=True)(x)
  x=tf.keras.layers.LSTM(64, return_sequences=True)(x)
  x=tf.keras.layers.Dense(30, activation="relu")(x)
  x=tf.keras.layers.Dense(10, activation="relu")(x)
  x=tf.keras.layers.Dense(units=1)(x)
  md = tf.keras.Model(inputs, x)
  md.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
  return md

models={'Linear': linear_m,
        'MLP': MLP,
        'Convolutional': Convolutional_m,
        'LSTM': LSTM_m,
        'GRU': GRU_m,
        'AR_LSTM': AR_LSTM_m,
        'Fusion': Fusion_CNN_LSTM}

def Training_and_show(name, data_frame, points=1, width=1, show_all=False, Max_epochs=500):
  # Función para entrenar modelos.
  # Los modelos se llaman por el nombre a traves de la variable (name).
  # Los nombres deben corresponder a diccionario models.
  # La función recibe:
  #       data_frame: Dataframe con la información de las serie de tiempo.
  #       points    : El número de puntos a predecir.
  #       width:    : El ancho de la ventana (puntos de tiempo tomados para generar las predicciones). 
  #       show_all  : Un boleano que indica si mostrar todos las predicciones o los últimos 100 puntos
  #       Max_epochs: El número máximo de épocas utilizadas para el entrenamiento de los modelos  

  train, test=get_data(data_frame, width)                             # Obtiene los datos de entrenamiento y prueba
  model=models[name](np.shape(train[0])[-2:])                         # Llama al modelo especificando la entrada del modelo ----> (times, features)
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',   # operado para detener el entrenamiento si no hay mejora del desempeño
                                                    patience=30,
                                                    mode='min')
  lx=model.fit(train[0], train[1],                                    # Entrenamiento del modelo
            epochs=Max_epochs,
            batch_size=1000,
            shuffle=False,
            callbacks=[early_stopping])
  
  clear_output(wait=True)                                             # Borra el registro del entrenamiento
  fig2=px.line(lx.history, y=list(lx.history.keys()),                 # Gráfica de las métricas de entrenamiento del modelo
               title='Training model',
               labels=dict(value="Score",index="Epoch", variable="Metrics"))
  fig2.show()
  #y_hat=prediction_points(model, test[0], points=points)              #Predicción del número de puntos de tiempo establecido
  #show_results(test[0], test[1], y_hat, points, show_all=show_all)    #Muestra los resultados obtenidos
  return model, (data_frame, points, width)   

def simulation(mx, ref):
  _,test=get_data(ref[0], ref[2])
  np.shape(test[0]), np.shape(test[1])

  fx=pd.DataFrame()
  anc=100
  for j in range(anc)[::-1]:
    for i in ['value', 'input', 'predic']:
      df0=pd.DataFrame()
      if i=='input':
        df0['Currency value']=test[0][-((2*anc)+j):-((anc)+j),-1,0]
        df0['Time (hour)']=-np.arange(anc)[::-1]
        df0['Time simulation (hour)']=(anc-np.ones(anc)*j).astype('int')

      if i=='value':
        df0['Currency value']=[np.nan, *test[1][-((2*anc)+j):-((anc)+j),0]]
        df0['Time (hour)']=-np.arange(-1,anc)[::-1]
        df0['Time simulation (hour)']=(anc-np.ones(anc+1)*j).astype('int')

      if i=='predic':
        y_hat=prediction_points(mx, test[0][-((2*anc)+j):-((anc)+j)], points=ref[1])
        df0['Currency value']=[np.nan, *y_hat[:,-1,0]]
        df0['Time (hour)']=-np.arange(-ref[1],anc)[::-1]
        df0['Time simulation (hour)']=(anc-np.ones(anc+ref[1])*j).astype('int')
      
      df0['variable']=i    
      fx=pd.concat([fx, df0], ignore_index=True)

  fig = px.line(fx, x="Time (hour)",
                y="Currency value",
                animation_frame="Time simulation (hour)",
                color="variable",
                hover_name="variable",
                template="plotly_dark",
                title="Autotrading with artificial intelligence (simulation)",
                range_y=(np.min(fx['Currency value']), np.max(fx['Currency value'])),
                markers=True)
  fig["layout"].pop("updatemenus") # optional, drop animation buttons
  fig.show()





