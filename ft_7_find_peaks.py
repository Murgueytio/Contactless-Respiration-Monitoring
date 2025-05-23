import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, filtfilt, find_peaks, savgol_filter # Importar savgol_filter
import matplotlib.cm as cm

# Regenerar filtered_heartbeat_signal y los parámetros STFT para asegurar disponibilidad
nyquist_freq = 0.5 * Fs
lowcut_heart, highcut_heart = 0.8, 2.5
norm_lowcut_heart = lowcut_heart / nyquist_freq
norm_highcut_heart = highcut_heart / nyquist_freq
b_heart, a_heart = butter(4, [norm_lowcut_heart, norm_highcut_heart], btype='band')
filtered_heartbeat_signal = filtfilt(b_heart, a_heart, vital_sign_phase_signal)

nperseg_heart = int(2 * Fs) # ~2 segundos de ventana
noverlap_heart = int(nperseg_heart * 0.75) # 75% de superposición

f_heart, t_heart, Zxx_heart = stft(filtered_heartbeat_signal, Fs, nperseg=nperseg_heart, noverlap=noverlap_heart)
spectrogram_heart = np.abs(Zxx_heart)
spectrogram_heart_dB = 20 * np.log10(spectrogram_heart + 1e-10)


# --- Parámetros de búsqueda de picos para el latido ---
search_low_heart = 0.8
search_high_heart = 2.5
min_peak_height_db = -80 # Podemos mantenerlo así por ahora, el suavizado ayudará
min_peak_prominence_db = 1 # Lo mantenemos bajo, el suavizado debería ayudar aquí

# Parámetros para el filtro Savitzky-Golay
window_length = 5 # Debe ser impar. Tamaño de la ventana del suavizado (en número de puntos de frecuencia)
polyorder = 2     # Orden del polinomio para el ajuste local

# Máscara de frecuencias para el rango de interés del latido
freq_mask_heart_search = (f_heart >= search_low_heart) & (f_heart <= search_high_heart)
heart_frequencies_in_range = f_heart[freq_mask_heart_search]


# --- ANÁLISIS DE UNA VENTANA ESPECÍFICA (CON SUAVIZADO Y NUEVOS PARÁMETROS) ---

# Seleccionar un índice de tiempo para la ventana a inspeccionar
time_to_inspect = 100
time_index = np.argmin(np.abs(t_heart - time_to_inspect))
print(f"Inspeccionando la ventana en el tiempo: {t_heart[time_index]:.2f} segundos")

# Extraer el espectro (en dB) para la ventana seleccionada, solo en el rango de búsqueda de latido
spectrum_raw_to_inspect = spectrogram_heart_dB[freq_mask_heart_search, time_index]

# APLICAR SUAVIZADO Savitzky-Golay
# Asegurarse de que window_length no sea mayor que la longitud del espectro
if len(spectrum_raw_to_inspect) < window_length:
    print(f"Advertencia: Longitud de la ventana ({len(spectrum_raw_to_inspect)}) es menor que window_length de Savitzky-Golay ({window_length}). No se aplicará suavizado.")
    spectrum_to_inspect = spectrum_raw_to_inspect
else:
    spectrum_to_inspect = savgol_filter(spectrum_raw_to_inspect, window_length, polyorder)


# Encontrar picos en esta ventana con el espectro suavizado y los parámetros ACTUALIZADOS
peaks_indices, properties = find_peaks(
    spectrum_to_inspect,
    height=min_peak_height_db,
    prominence=min_peak_prominence_db
)

# Obtener las frecuencias y prominencias de los picos detectados
detected_peak_freqs = heart_frequencies_in_range[peaks_indices]
detected_peak_heights = spectrum_to_inspect[peaks_indices]
detected_peak_prominences = properties['prominences']

print(f"\nPicos detectados en la ventana de tiempo {t_heart[time_index]:.2f} s (con suavizado):")
if len(detected_peak_freqs) > 0:
    for i in range(len(detected_peak_freqs)):
        print(f"  - Frecuencia: {detected_peak_freqs[i]:.2f} Hz, Magnitud: {detected_peak_heights[i]:.2f} dB, Prominencia: {detected_peak_prominences[i]:.2f} dB")
else:
    print("  No se detectaron picos con los parámetros actuales en esta ventana (incluso con suavizado).")


# --- Visualización del Espectro de la Ventana Específica ---
plt.figure(figsize=(12, 6))
plt.plot(heart_frequencies_in_range, spectrum_raw_to_inspect, label='Espectro Original (dB)', alpha=0.5, linestyle='--')
plt.plot(heart_frequencies_in_range, spectrum_to_inspect, label='Espectro Suavizado (dB)', color='blue')
plt.plot(detected_peak_freqs, detected_peak_heights, 'x', color='red', markersize=8, label='Picos Detectados')
plt.title(f'Espectro de Frecuencia de la Ventana de Latido en {t_heart[time_index]:.2f} s (Prominencia={min_peak_prominence_db} dB, Suavizado)')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (dB)')
plt.xlim(search_low_heart - 0.1, search_high_heart + 0.1)
plt.axvspan(search_low_heart, search_high_heart, color='green', alpha=0.1, label='Rango de Búsqueda')
plt.axhline(y=min_peak_height_db, color='gray', linestyle='--', label=f'Altura Mínima ({min_peak_height_db} dB)')

# Graficar la prominencia de los picos
for i, peak_idx in enumerate(peaks_indices):
    plt.vlines(x=heart_frequencies_in_range[peak_idx], ymin=detected_peak_heights[i] - detected_peak_prominences[i],
               ymax=detected_peak_heights[i], color='purple', linestyle=':', linewidth=1,
               label=f'Prominencia ({detected_peak_prominences[i]:.1f} dB)' if i == 0 else "")

plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# --- Función para encontrar el pico de latido más robusto (CON SUAVIZADO) ---
# Esta función es la misma que antes, pero internamente aplicará el suavizado
def find_robust_heartbeat_peak_smoothed(spectrum_window_db_raw, frequencies_in_range, window_length, polyorder):
    """
    Encuentra el pico de latido más robusto en una ventana de espectro dada.
    Aplica suavizado antes de la detección de picos.
    Prioriza picos con alta prominencia y posibles armónicos.
    """
    if len(spectrum_window_db_raw) == 0 or np.all(np.isnan(spectrum_window_db_raw)) or np.all(np.isinf(spectrum_window_db_raw)):
        return np.nan

    # Aplicar suavizado Savitzky-Golay
    if len(spectrum_window_db_raw) < window_length:
        spectrum_window_db_smoothed = spectrum_window_db_raw
    else:
        spectrum_window_db_smoothed = savgol_filter(spectrum_window_db_raw, window_length, polyorder)

    peaks_indices, properties = find_peaks(
        spectrum_window_db_smoothed, # Usar el espectro suavizado aquí
        height=min_peak_height_db,
        prominence=min_peak_prominence_db
    )

    if len(peaks_indices) == 0:
        return np.nan

    candidate_freqs = frequencies_in_range[peaks_indices]
    candidate_prominences = properties['prominences']

    best_peak_freq = np.nan
    max_score = -1

    for i, freq in enumerate(candidate_freqs):
        prominence = candidate_prominences[i]
        score = prominence

        harmonic_search_range_low = freq * 1.8
        harmonic_search_range_high = freq * 2.2

        harmonic_indices = np.where(
            (frequencies_in_range >= harmonic_search_range_low) &
            (frequencies_in_range <= harmonic_search_range_high)
        )[0]

        for h_idx in harmonic_indices:
            if h_idx in peaks_indices:
                idx_in_peaks = np.where(peaks_indices == h_idx)[0][0]
                score += properties['prominences'][idx_in_peaks] * 0.5

        if score > max_score and search_low_heart <= freq <= search_high_heart:
            max_score = score
            best_peak_freq = freq

    return best_peak_freq


# --- Aplicar la función de detección robusta con suavizado a cada ventana del espectrograma ---
refined_dominant_heart_freqs_smoothed = []
for i in range(spectrogram_heart_dB.shape[1]):
    spectrum_window_raw = spectrogram_heart_dB[freq_mask_heart_search, i]
    peak_freq = find_robust_heartbeat_peak_smoothed(spectrum_window_raw, heart_frequencies_in_range, window_length, polyorder)
    refined_dominant_heart_freqs_smoothed.append(peak_freq)


# --- Visualizar la Frecuencia de Latido Refinada (con Suavizado) ---
plt.figure(figsize=(15, 6))
plt.plot(t_heart, refined_dominant_heart_freqs_smoothed)
plt.title(f'Frecuencia de Latido Dominante Refinada a lo largo del Tiempo (Prominencia={min_peak_prominence_db} dB, Suavizado)')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.grid(True)
plt.ylim(0.7, 3.0)
plt.axhspan(0.8, 2.5, color='green', alpha=0.1, label='Rango Esperado')
plt.legend()
plt.tight_layout()
plt.show()

# --- Comparar la frecuencia media del latido refinada con la referencia ---
mean_refined_heartbeat_freq_smoothed = np.nanmean(refined_dominant_heart_freqs_smoothed)

# Asumimos estimated_heartbeat_ref_freq está definido, o lo definimos aquí para el ejemplo
estimated_heartbeat_ref_freq = 1.35 # Asegúrate de usar el valor real que obtuviste.

print("\n--- Comparación de Frecuencias de Latido (Refinada con Suavizado) ---")
if not np.isnan(mean_refined_heartbeat_freq_smoothed):
    print(f"  Estimación por Radar (Promedio Refinado): {mean_refined_heartbeat_freq_smoothed:.2f} Hz ({mean_refined_heartbeat_freq_smoothed * 60:.1f} LPM)")
    print(f"  Estimación por Referencia: {estimated_heartbeat_ref_freq:.2f} Hz ({estimated_heartbeat_ref_freq * 60:.1f} LPM)")
    print(f"  Diferencia Absoluta (Refinada): {np.abs(mean_refined_heartbeat_freq_smoothed - estimated_heartbeat_ref_freq):.2f} Hz")
    if estimated_heartbeat_ref_freq != 0:
        print(f"  Error Porcentual (Refinada): {np.abs((mean_refined_heartbeat_freq_smoothed - estimated_heartbeat_ref_freq) / estimated_heartbeat_ref_freq) * 100:.2f}%")
    else:
        print("  Error Porcentual (Refinada): No se puede calcular (frecuencia de referencia es 0).")
else:
    print("No se pudo calcular la frecuencia promedio del latido refinada (demasiados picos no detectados).")

