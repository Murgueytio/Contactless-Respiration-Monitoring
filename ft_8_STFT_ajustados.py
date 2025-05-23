import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, filtfilt, savgol_filter
import matplotlib.cm as cm

# --- Asume que las siguientes variables están disponibles de pasos anteriores: ---
# vital_sign_phase_signal (la señal de fase extraída sin filtrar)
# Fs = 20.48 (Frecuencia de muestreo)
# ---------------------------------------------------------------------------------

# Regenerar filtered_heartbeat_signal y los parámetros STFT para asegurar disponibilidad
nyquist_freq = 0.5 * Fs
lowcut_heart, highcut_heart = 0.8, 2.5
norm_lowcut_heart = lowcut_heart / nyquist_freq
norm_highcut_heart = highcut_heart / nyquist_freq
b_heart, a_heart = butter(4, [norm_lowcut_heart, norm_highcut_heart], btype='band')
filtered_heartbeat_signal = filtfilt(b_heart, a_heart, vital_sign_phase_signal)


# --- AJUSTES CLAVE para la STFT del Latido ---
# Aumentar el tamaño de la ventana de la STFT para mejorar la resolución en frecuencia
# Esto significará menos puntos en el eje de tiempo pero más puntos en el eje de frecuencia
# Una ventana de 10 segundos es 10 * 20.48 = ~205 muestras.
nperseg_heart = int(10 * Fs) # Aumentado a ~10 segundos de ventana
noverlap_heart = int(nperseg_heart * 0.75) # Mantenemos 75% de superposición

print(f"STFT Latido (Ajustado): nperseg={nperseg_heart} muestras, noverlap={noverlap_heart} muestras")

f_heart, t_heart, Zxx_heart = stft(filtered_heartbeat_signal, Fs, nperseg=nperseg_heart, noverlap=noverlap_heart)
spectrogram_heart = np.abs(Zxx_heart)
spectrogram_heart_dB = 20 * np.log10(spectrogram_heart + 1e-10)


# --- Parámetros de búsqueda de picos para el latido ---
search_low_heart = 0.8
search_high_heart = 2.5

# Parámetros para el filtro Savitzky-Golay (Ajustados si es necesario)
# Ahora deberíamos tener más puntos de frecuencia, así que podemos usar window_length más grandes
window_length_savgol = 9 # Debe ser impar. Ajustar según la densidad de puntos de frecuencia.
polyorder_savgol = 3     # Orden del polinomio para el ajuste local

# Máscara de frecuencias para el rango de interés del latido
freq_mask_heart_search = (f_heart >= search_low_heart) & (f_heart <= search_high_heart)
heart_frequencies_in_range = f_heart[freq_mask_heart_search]


# --- NUEVA ESTRATEGIA DE DETECCIÓN DE PICO PARA EL LATIDO ---
# Vamos a encontrar el pico más alto en el rango de interés después del suavizado.
# Esto es más robusto cuando find_peaks falla por criterios de prominencia.

refined_dominant_heart_freqs = []
for i in range(spectrogram_heart_dB.shape[1]): # Iterar sobre las columnas (ventanas de tiempo)
    spectrum_window_raw = spectrogram_heart_dB[freq_mask_heart_search, i]

    if len(spectrum_window_raw) == 0 or np.all(np.isnan(spectrum_window_raw)) or np.all(np.isinf(spectrum_window_raw)):
        refined_dominant_heart_freqs.append(np.nan)
        continue

    # Aplicar suavizado Savitzky-Golay
    current_window_length = len(spectrum_window_raw)
    if current_window_length < window_length_savgol:
        # Si la ventana es demasiado pequeña para el filtro Savitzky-Golay, no lo aplicamos
        # O podemos reducir dinámicamente window_length a current_window_length - 1 si es impar
        if current_window_length > 1: # Asegurarse de que haya al menos 2 puntos para un filtro
            # Ajustar dinámicamente window_length para que sea impar y menor que current_window_length
            adjusted_window_length = current_window_length - 1 if current_window_length % 2 == 0 else current_window_length
            if adjusted_window_length < 3: # Asegurar un mínimo de 3 para Savitzky-Golay efectivo
                spectrum_window_smoothed = spectrum_window_raw
            else:
                spectrum_window_smoothed = savgol_filter(spectrum_window_raw, adjusted_window_length, polyorder_savgol)
        else:
            spectrum_window_smoothed = spectrum_window_raw # No se puede suavizar con 1 punto
    else:
        spectrum_window_smoothed = savgol_filter(spectrum_window_raw, window_length_savgol, polyorder_savgol)

    # Encontrar el índice del pico máximo en el espectro suavizado
    peak_idx_in_range = np.argmax(spectrum_window_smoothed)
    peak_freq = heart_frequencies_in_range[peak_idx_in_range]

    refined_dominant_heart_freqs.append(peak_freq)

# --- Post-procesamiento para suavizar la serie de tiempo de frecuencias ---
# Usaremos un filtro de mediana para eliminar picos erráticos (valores atípicos)
# y luego interpolación lineal para rellenar los NaNs
from scipy.ndimage import median_filter
from scipy.interpolate import interp1d

# Convertir a array de numpy para facilitar el manejo de NaNs
refined_dominant_heart_freqs = np.array(refined_dominant_heart_freqs)

# Paso 1: Filtrado de mediana para eliminar los valores atípicos
# El tamaño de la ventana (e.g., 5 o 7) dependerá de cuán "picada" esté la señal
median_filtered_freqs = median_filter(refined_dominant_heart_freqs, size=5) # Ajusta el tamaño según sea necesario

# Paso 2: Interpolación para rellenar NaNs
# Encuentra los índices donde los valores no son NaN
not_nan_indices = np.where(~np.isnan(median_filtered_freqs))[0]

if len(not_nan_indices) > 1: # Necesitamos al menos 2 puntos para interpolar
    # Crea una función de interpolación solo para los puntos válidos
    interp_func = interp1d(not_nan_indices, median_filtered_freqs[not_nan_indices], kind='linear', fill_value="extrapolate")
    # Aplica la interpolación a todos los puntos de tiempo
    interpolated_freqs = interp_func(np.arange(len(median_filtered_freqs)))
else:
    # Si no hay suficientes puntos válidos, la interpolación no es posible. Dejar NaNs.
    interpolated_freqs = np.full_like(median_filtered_freqs, np.nan)


# --- Visualizar la Frecuencia de Latido Refinada (con Nuevo Método y Post-procesamiento) ---
plt.figure(figsize=(15, 6))
plt.plot(t_heart, interpolated_freqs, label='Frecuencia de Latido Refinada (Suavizado y Mediana)')
plt.title(f'Frecuencia de Latido Dominante Refinada a lo largo del Tiempo (Nueva Estrategia)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.grid(True)
plt.ylim(0.7, 3.0)
plt.axhspan(0.8, 2.5, color='green', alpha=0.1, label='Rango Esperado')
plt.legend()
plt.tight_layout()
plt.show()

# --- Comparar la frecuencia media del latido refinada con la referencia ---
mean_refined_heartbeat_freq_final = np.nanmean(interpolated_freqs)

# Asumimos estimated_heartbeat_ref_freq está definido, o lo definimos aquí para el ejemplo
estimated_heartbeat_ref_freq = 1.35 # Asegúrate de usar el valor real que obtuviste.

print("\n--- Comparación de Frecuencias de Latido (Refinada con Nueva Estrategia y Post-procesamiento) ---")
if not np.isnan(mean_refined_heartbeat_freq_final):
    print(f"  Estimación por Radar (Promedio Refinado): {mean_refined_heartbeat_freq_final:.2f} Hz ({mean_refined_heartbeat_freq_final * 60:.1f} LPM)")
    print(f"  Estimación por Referencia: {estimated_heartbeat_ref_freq:.2f} Hz ({estimated_heartbeat_ref_freq * 60:.1f} LPM)")
    print(f"  Diferencia Absoluta (Refinada): {np.abs(mean_refined_heartbeat_freq_final - estimated_heartbeat_ref_freq):.2f} Hz")
    if estimated_heartbeat_ref_freq != 0:
        print(f"  Error Porcentual (Refinada): {np.abs((mean_refined_heartbeat_freq_final - estimated_heartbeat_ref_freq) / estimated_heartbeat_ref_freq) * 100:.2f}%")
    else:
        print("  Error Porcentual (Refinada): No se puede calcular (frecuencia de referencia es 0).")
else:
    print("No se pudo calcular la frecuencia promedio del latido refinada (demasiados NaNs después del procesamiento).")
