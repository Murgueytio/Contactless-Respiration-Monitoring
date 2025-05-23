import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.stats import pearsonr # Para la autocorrelación
from scipy.interpolate import interp1d
from scipy.ndimage import median_filter # Para post-procesamiento

# --- Asume que las siguientes variables están disponibles de pasos anteriores: ---
# vital_sign_phase_signal (la señal de fase extraída sin filtrar)
# Fs = 20.48 (Frecuencia de muestreo)
# ---------------------------------------------------------------------------------

# Regenerar filtered_heartbeat_signal y los parámetros de filtrado para asegurar disponibilidad
nyquist_freq = 0.5 * Fs # Variable correctamente definida
lowcut_heart, highcut_heart = 0.8, 2.5
norm_lowcut_heart = lowcut_heart / nyquist_freq
norm_highcut_heart = highcut_heart / nyquist_freq
b_heart, a_heart = butter(4, [norm_lowcut_heart, norm_highcut_heart], btype='band')
filtered_heartbeat_signal = filtfilt(b_heart, a_heart, vital_sign_phase_signal)


# --- Parámetros para el análisis de Autocorrelación ---
# Tamaño de la ventana en segundos para el análisis de autocorrelación
window_size_sec = 5 # Más larga que la STFT anterior para mejor periodicidad
window_overlap_sec = 0.75 * window_size_sec # 75% de superposición

window_size_samples = int(window_size_sec * Fs)
overlap_samples = int(window_overlap_sec * Fs)

# Calcular los tiempos centrales de cada ventana
time_points = []
for i in range(0, len(filtered_heartbeat_signal) - window_size_samples + 1, window_size_samples - overlap_samples):
    time_points.append((i + window_size_samples / 2) / Fs)

# Rango de interés para el latido cardíaco en términos de Lags (muestras)
# Frecuencia: 0.8 Hz a 2.5 Hz
# Periodo: 1/2.5 = 0.4 s a 1/0.8 = 1.25 s
min_period_samples = int(0.4 * Fs)
max_period_samples = int(1.25 * Fs)

# Asegurarse de que el rango de búsqueda de lags es válido para la ventana
if min_period_samples < 1: min_period_samples = 1
if max_period_samples >= window_size_samples: max_period_samples = window_size_samples - 1 # No se puede desplazar más que la ventana

# --- Función para calcular la Autocorrelación ---
def calculate_autocorrelation(signal_segment, max_lag):
    """
    Calcula la función de autocorrelación de una señal.
    Utiliza np.correlate para mayor eficiencia.
    Retorna los valores de autocorrelación normalizados por el valor en lag 0.
    """
    n = len(signal_segment)
    # Autocorrelación sesgada (biased), normalizada por n
    autocorr = np.correlate(signal_segment, signal_segment, mode='full') / n
    # Solo nos interesa la segunda mitad (lags positivos)
    autocorr = autocorr[n-1:]
    # Normalizar por el valor en lag 0 (energía de la señal)
    autocorr = autocorr / autocorr[0]
    return autocorr[:max_lag + 1] # Limitar al max_lag deseado

# --- Procesar cada ventana y extraer la frecuencia de latido ---
heart_rate_autocorr = []
lags_in_range = np.arange(min_period_samples, max_period_samples + 1)

for i in range(0, len(filtered_heartbeat_signal) - window_size_samples + 1, window_size_samples - overlap_samples):
    segment = filtered_heartbeat_signal[i : i + window_size_samples]

    if len(segment) == 0:
        heart_rate_autocorr.append(np.nan)
        continue

    # Calcular autocorrelación para el segmento
    # Asegúrate de que max_lag sea menor que la longitud del segmento
    current_max_lag = min(max_period_samples, len(segment) - 1)
    if current_max_lag <= min_period_samples: # Si la ventana es muy corta para el rango, saltar
        heart_rate_autocorr.append(np.nan)
        continue

    autocorr_segment = calculate_autocorrelation(segment, current_max_lag)

    # Buscar el pico más alto en el rango de interés (excluyendo lag 0)
    # Mapear lags_in_range a los índices de autocorr_segment
    search_indices = np.arange(min_period_samples, current_max_lag + 1) # Asegurar que los índices están dentro del rango de autocorr_segment
    if len(search_indices) == 0 or np.all(np.isnan(autocorr_segment[search_indices])) : # Si no hay lags válidos
        heart_rate_autocorr.append(np.nan)
        continue

    # Encuentra el índice del valor máximo en el rango relevante
    max_autocorr_idx_relative = np.argmax(autocorr_segment[search_indices])
    # El lag correspondiente es el valor en search_indices
    estimated_period_samples = search_indices[max_autocorr_idx_relative]

    # Convertir el periodo a frecuencia
    if estimated_period_samples > 0:
        estimated_freq = Fs / estimated_period_samples
        heart_rate_autocorr.append(estimated_freq)
    else:
        heart_rate_autocorr.append(np.nan) # No se pudo estimar


# --- Post-procesamiento: Filtrado de Mediana e Interpolación ---
heart_rate_autocorr = np.array(heart_rate_autocorr)

# Paso 1: Filtrado de mediana
# Ajusta el tamaño de la ventana del filtro de mediana si la curva final es muy ruidosa
median_filtered_hr = median_filter(heart_rate_autocorr, size=3) # Un tamaño más pequeño (e.g., 3) es un buen comienzo

# Paso 2: Interpolación para rellenar NaNs
valid_indices = np.where(~np.isnan(median_filtered_hr))[0]
if len(valid_indices) > 1:
    interp_func = interp1d(valid_indices, median_filtered_hr[valid_indices], kind='linear', fill_value="extrapolate")
    final_heart_rate = interp_func(np.arange(len(median_filtered_hr)))
else:
    final_heart_rate = np.full_like(median_filtered_hr, np.nan)


# --- Visualización ---
plt.figure(figsize=(15, 6))
plt.plot(time_points, final_heart_rate, label='Frecuencia de Latido (Autocorrelación)')
plt.title('Frecuencia de Latido Dominante a lo largo del Tiempo (Autocorrelación)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.grid(True)
plt.ylim(0.7, 3.0) # Mismo rango de visualización
plt.axhspan(lowcut_heart, highcut_heart, color='green', alpha=0.1, label='Rango Esperado')
plt.legend()
plt.tight_layout()
plt.show()

# --- Comparación con Referencia ---
mean_autocorr_heart_freq = np.nanmean(final_heart_rate)

# Asumimos estimated_heartbeat_ref_freq está definido
estimated_heartbeat_ref_freq = 1.35 # Usar el valor real de tu referencia

print("\n--- Comparación de Frecuencias de Latido (Autocorrelación) ---")
if not np.isnan(mean_autocorr_heart_freq):
    print(f"  Estimación por Radar (Promedio Autocorrelación): {mean_autocorr_heart_freq:.2f} Hz ({mean_autocorr_heart_freq * 60:.1f} LPM)")
    print(f"  Estimación por Referencia: {estimated_heartbeat_ref_freq:.2f} Hz ({estimated_heartbeat_ref_freq * 60:.1f} LPM)")
    print(f"  Diferencia Absoluta: {np.abs(mean_autocorr_heart_freq - estimated_heartbeat_ref_freq):.2f} Hz")
    if estimated_heartbeat_ref_freq != 0:
        print(f"  Error Porcentual: {np.abs((mean_autocorr_heart_freq - estimated_heartbeat_ref_freq) / estimated_heartbeat_ref_freq) * 100:.2f}%")
    else:
        print("  Error Porcentual: No se puede calcular (frecuencia de referencia es 0).")
else:
    print("No se pudo calcular la frecuencia promedio del latido con autocorrelación (demasiados NaNs).")
