import pandas as pd
import numpy as np
import os 

# Base path to the folder containing the CSV files
# base_path = '/content/------------------/MobiVital-dataset/'

# List of the 8 CSV files to prototype.
csv_files = 
[
    '231012_userK_tripod_01_4.csv',
    '231026_userG_tripod_01_7.csv',
    '231026_userI_tripod_01_2.csv',
    '231026_userI_tripod_01_5.csv',
    '231026_userJ_tripod_03_5.csv',
    '231116_userJ_tripod_01_5.csv',
    '231130_userJ_tripod_02_3.csv',
    '240205_userI_tripod_01_1.csv'
]

print("Procesando archivos...")

for file_name in csv_files:
    file_path = os.path.join(base_path, file_name) # Construir la ruta completa

    try:
        df = pd.read_csv(file_path, header=None)

        # --- Separar los datos ---
        # Columnas 13-252: Datos de Radar UWB I y Q (índices 12-251 en pandas)
        radar_iq_data = df.iloc[:, 12:252]

        # Columnas 253-254: Referencias (Ground Truth) (índices 252-253 en pandas)
        references_data = df.iloc[:, 252:254]

        # --- Formar números complejos a partir de Radar I/Q ---
        # Las primeras 120 columnas de radar_iq_data (índices 0-119) son I
        radar_i = radar_iq_data.iloc[:, 0:120].values # .values convierte a NumPy array
        # Las siguientes 120 columnas de radar_iq_data (índices 120-239) son Q
        radar_q = radar_iq_data.iloc[:, 120:240].values # .values convierte a NumPy array

        # Formar números complejos: I + j*Q
        radar_complex_single_file = radar_i + 1j * radar_q

        # --- Almacenar en las listas ---
        lista_radar_complex.append(radar_complex_single_file)
        lista_references.append(references_data.values) # .values convierte a NumPy array

        print(f" - Procesado {file_name}. Shape radar complex: {radar_complex_single_file.shape}, Shape references: {references_data.shape}")

    except FileNotFoundError:
        print(f"Error: Archivo no encontrado en {file_path}")
    except Exception as e:
        print(f"Error procesando {file_name}: {e}")


# Concatenar a lo largo del eje 0 (tiempo/scanlines)
if lista_radar_complex: # Verificar que las listas no estén vacías
    all_radar_complex = np.concatenate(lista_radar_complex, axis=0)
    all_references = np.concatenate(lista_references, axis=0)

    print("\n¡Procesamiento de archivos completo!")
    print("\nDimensiones Totales Radar Complejo:", all_radar_complex.shape)
    # El shape será (total_scanlines_de_todos_los_archivos, 120 range bins)
    print("Dimensiones Totales Referencias:", all_references.shape)
    # El shape será (total_scanlines_de_todos_los_archivos, 2 referencias)

    # Ahora all_radar_complex y all_references contienen los datos combinados
    # Puedes mostrar las primeras filas para verificar (ahora son NumPy arrays)
    print("\nPrimeros Scanlines del Radar Complejo Combinado:")
    print(all_radar_complex[:5, :5])

    print("\nPrimeras Referencias Combinadas:")
    print(all_references[:5, :])

else:
    print("No files could be processed.")
