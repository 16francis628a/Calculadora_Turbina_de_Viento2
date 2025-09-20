
import streamlit as st
import numpy as np
import joblib


def main():
        
    mlp_cargado = joblib.load("mejor_mlp_modelo.joblib")
    scaler_cargado = joblib.load("scaler_mlp.joblib")

    st.title('Predicción de la potencia esperada de una turbina de viento eólica')

    #RangeIndex: 50530 entries, 0 to 50529
    #Data columns (total 6 columns):
    # #   Column                   Non-Null Count  Dtype         
    #---  ------                   --------------  -----         
    # 0   Date                     50530 non-null  datetime64[ns]
    # 1   LV ActivePower           50530 non-null  float64          Target      
    # 2   Wind Speed               50530 non-null  float64       
    # 3   Theoretical_Power_Curve  50530 non-null  float64       
    # 4   Wind Direction           50530 non-null  float64       
    # 5   mes                      50530 non-null  int32         


    Wind_Speed = st.number_input('Velocidad del viento (m/s):')
    st.write(f'{Wind_Speed} kW')

    Theoretical_Power_Curve = st.number_input('Curva Teorica de potencia (kW):')
    st.write(f'{Theoretical_Power_Curve} kW')

    Wind_Direction = st.number_input('Dirección del viento (°):')
    st.write(f'{Wind_Direction} °')

    # Ejemplo de predicción

    X_nuevo = np.array([[Wind_Speed, Theoretical_Power_Curve, Wind_Direction]])  # datos de entrada
    X_nuevo_scaled = scaler_cargado.transform(X_nuevo)
    prediccion = mlp_cargado.predict(X_nuevo_scaled)

    prediccion = float(f'{prediccion[0]:.3f}')

    if prediccion > 0:
        st.success(f'La potencia esperada es {prediccion} kW') # [0]:.3f
    else:
        st.success(f'')
    
if __name__ == '__main__':
    main()