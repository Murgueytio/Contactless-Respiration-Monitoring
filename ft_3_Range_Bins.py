# Basado en la observación del gráfico, seleccionamos un rango de range bins de interés
# Ajusta estos índices según donde veas el pico principal en tu gráfico (ej: 25 a 30)
start_bin = 25
end_bin = 31 # El slicing en Python es exclusivo en el final, así que 31 incluye el índice 30

# Seleccionar los datos de radar complejos para este rango de range bins
# .iloc en pandas se usa para seleccionar por índice. Como all_radar_complex_clutter_removed
# es un NumPy array, usamos indexación de NumPy.
target_range_bins_data_cr = all_radar_complex_clutter_removed[:, start_bin:end_bin]

print(f"\nDatos seleccionados de los Range Bins {start_bin} a {end_bin-1}:")
print("Dimensiones de los datos seleccionados:", target_range_bins_data_cr.shape)
# El shape será (total_scanlines, numero_de_bins_seleccionados)

import numpy as np
import matplotlib.pyplot as plt
from numpy import unwrap  # Para "desenvolver" la fase

start_bin = 25
end_bin = 31 # Recuerda que el slicing en Python es exclusivo en el final (31 incluye el índice 30)

# 1. Seleccionar los datos de radar complejos para el rango de bins elegido
target_range_bins_data_cr = all_radar_complex_clutter_removed[:, start_bin:end_bin]

print(f"Dimensiones de los datos seleccionados de los Range Bins {start_bin} a {end_bin-1}: {target_range_bins_data_cr.shape}")

# --- Paso 1: Extracción de la Señal de Movimiento (mediante la fase) ---

# Sumar las señales complejas de los bins seleccionados para obtener una única serie temporal.
# Esto es una forma común de integrar la información a través de la extensión espacial del objetivo.
vital_sign_complex_signal = np.sum(target_range_bins_data_cr, axis=1)

# np.angle devuelve la fase en radianes. np.unwrap maneja los saltos de fase de 2*pi
# que ocurren naturalmente, convirtiendo una señal de fase "envuelta" en una continua.
vital_sign_phase_signal = unwrap(np.angle(vital_sign_complex_signal))

print(f"Dimensiones de la señal de fase extraída: {vital_sign_phase_signal.shape}")

# Opcional: Visualizar una pequeña porción de la señal de fase para ver el movimiento
plt.figure(figsize=(10, 4))
plt.plot(vital_sign_phase_signal[::50]) # Plotear cada 50a muestra para que no sea muy denso
plt.title('Señal de Fase Extraída (Movimiento Fisiológico)')
plt.xlabel('Muestra (tiempo)')
plt.ylabel('Fase (radianes)')
plt.grid(True)
plt.show()


# --- Paso 2: Análisis en Frecuencia (FFT) ---

# Frecuencia de muestreo (Fs) del eje de "slow-time" del radar
# (MobiVital: 20.48 frames/s)
Fs = 20.48 # Hz

# Número de puntos en la señal de fase
N = len(vital_sign_phase_signal)

# Calcular la Transformada Rápida de Fourier (FFT)
fft_result = np.fft.fft(vital_sign_phase_signal)

# Calcular las frecuencias correspondientes a los puntos de la FFT
# ffthfreq devuelve las frecuencias para N puntos dado un periodo de muestreo (1/Fs)
frequencies = np.fft.fftfreq(N, 1/Fs)

# Tomar solo la parte positiva del espectro (las frecuencias negativas son un espejo)
positive_frequencies = frequencies[frequencies >= 0]
positive_fft_magnitude = np.abs(fft_result[frequencies >= 0])

# Normalizar la magnitud para una mejor visualización (opcional)
# positive_fft_magnitude_normalized = positive_fft_magnitude / np.max(positive_fft_magnitude)

# --- Paso 3: Visualización del Espectro de Frecuencia ---

plt.figure(figsize=(15, 10))

# Subplot para el rango de la respiración
plt.subplot(2, 1, 1) # 2 filas, 1 columna, primer plot
plt.plot(positive_frequencies, positive_fft_magnitude)
plt.title('Espectro de Frecuencia de la Señal de Signos Vitales')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (Arbitraria)')
plt.xlim(0.05, 0.6) # Rango típico de frecuencia para la respiración
plt.grid(True)
plt.axvspan(0.1, 0.5, color='red', alpha=0.2, label='Rango Respiración Esperada (0.1-0.5 Hz)')
plt.legend()
plt.tight_layout() # Ajusta automáticamente los subplots

# Subplot para el rango del latido cardíaco
plt.subplot(2, 1, 2) # 2 filas, 1 columna, segundo plot
plt.plot(positive_frequencies, positive_fft_magnitude)
plt.title('Espectro de Frecuencia de la Señal de Signos Vitales')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (Arbitraria)')
plt.xlim(0.7, 3.0) # Rango típico de frecuencia para el latido
plt.grid(True)
plt.axvspan(0.8, 2.5, color='green', alpha=0.2, label='Rango Latido Esperado (0.8-2.5 Hz)')
plt.legend()
plt.tight_layout() # Ajusta automáticamente los subplots

plt.show()


# --- Paso 4: Estimación de Frecuencias (Busqueda de picos) ---

# Estimar la frecuencia de respiración
respiration_freq_mask = (positive_frequencies >= 0.05) & (positive_frequencies <= 0.6)
if np.any(respiration_freq_mask):
    respiration_frequencies_in_range = positive_frequencies[respiration_freq_mask]
    respiration_magnitude_in_range = positive_fft_magnitude[respiration_freq_mask]
    estimated_respiration_freq = respiration_frequencies_in_range[np.argmax(respiration_magnitude_in_range)]
    print(f"\nFrecuencia de Respiración Estimada: {estimated_respiration_freq:.2f} Hz")
    print(f" (o {estimated_respiration_freq * 60:.1f} respiraciones por minuto)")
else:
    print("\nNo se detectó un pico claro de respiración en el rango esperado.")

# Estimar la frecuencia de latido
heartbeat_freq_mask = (positive_frequencies >= 0.7) & (positive_frequencies <= 3.0)
if np.any(heartbeat_freq_mask):
    heartbeat_frequencies_in_range = positive_frequencies[heartbeat_freq_mask]
    heartbeat_magnitude_in_range = positive_fft_magnitude[heartbeat_freq_mask]
    estimated_heartbeat_freq = heartbeat_frequencies_in_range[np.argmax(heartbeat_magnitude_in_range)]
    print(f"Frecuencia de Latido Estimada: {estimated_heartbeat_freq:.2f} Hz")
    print(f" (o {estimated_heartbeat_freq * 60:.1f} latidos por minuto)")
else:
    print("No se detectó un pico claro de latido en el rango esperado.")
