import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft, butter, filtfilt # Importar stft
import matplotlib.cm as cm # Para los mapas de color

nyquist_freq = 0.5 * Fs

lowcut_resp, highcut_resp = 0.1, 0.5
norm_lowcut_resp, norm_highcut_resp = lowcut_resp / nyquist_freq, highcut_resp / nyquist_freq
b_resp, a_resp = butter(4, [norm_lowcut_resp, norm_highcut_resp], btype='band')
filtered_respiration_signal = filtfilt(b_resp, a_resp, vital_sign_phase_signal)

lowcut_heart, highcut_heart = 0.8, 2.5
norm_lowcut_heart = lowcut_heart / nyquist_freq
norm_highcut_heart = highcut_heart / nyquist_freq # ¡CORREGIDO AQUÍ!
b_heart, a_heart = butter(4, [norm_lowcut_heart, norm_highcut_heart], btype='band')
filtered_heartbeat_signal = filtfilt(b_heart, a_heart, vital_sign_phase_signal)


# --- Parámetros para la STFT ---
# Duración total de la señal: N / Fs = 12000 / 20.48 = ~585 segundos (~9.7 minutos)

# Para Respiración: Queremos buena resolución en frecuencia en el rango bajo.
# Una ventana de 4 segundos (4 * Fs = ~82 muestras) es un buen compromiso para la resolución.
nperseg_resp = int(4 * Fs) # ~4 segundos de ventana, buen equilibrio
noverlap_resp = int(nperseg_resp * 0.75) # 75% de superposición

# Para Latido Cardíaco: Necesitamos un poco más de resolución en tiempo y aún así decente en frecuencia.
# Una ventana más pequeña de 2 segundos (2 * Fs = ~41 muestras) puede ser mejor para capturar cambios rápidos.
nperseg_heart = int(2 * Fs) # ~2 segundos de ventana
noverlap_heart = int(nperseg_heart * 0.75) # 75% de superposición

print(f"STFT Respiración: nperseg={nperseg_resp} muestras, noverlap={noverlap_resp} muestras")
print(f"STFT Latido: nperseg={nperseg_heart} muestras, noverlap={noverlap_heart} muestras")


# --- Aplicar STFT a la Señal de Respiración Filtrada ---
f_resp, t_resp, Zxx_resp = stft(filtered_respiration_signal, Fs, nperseg=nperseg_resp, noverlap=noverlap_resp)
# Tomar la magnitud del espectrograma y convertir a dB para mejor visualización
spectrogram_resp = np.abs(Zxx_resp)
# Convertir a dB (logarítmico) para visualizar mejor las variaciones de magnitud
# Evitar log(0) añadiendo un pequeño valor
spectrogram_resp_dB = 20 * np.log10(spectrogram_resp + 1e-10)

# --- Aplicar STFT a la Señal de Latido Filtrada ---
f_heart, t_heart, Zxx_heart = stft(filtered_heartbeat_signal, Fs, nperseg=nperseg_heart, noverlap=noverlap_heart)
spectrogram_heart = np.abs(Zxx_heart)
spectrogram_heart_dB = 20 * np.log10(spectrogram_heart + 1e-10)

# --- Visualización de los Espectrogramas ---
plt.figure(figsize=(16, 12))

# Espectrograma de Respiración
plt.subplot(2, 1, 1)
# En f_resp, selecciona solo las frecuencias dentro del rango de interés
freq_mask_resp_plot = (f_resp >= 0.05) & (f_resp <= 0.6) # Rango de visualización más amplio que el filtro
plt.pcolormesh(t_resp, f_resp[freq_mask_resp_plot], spectrogram_resp_dB[freq_mask_resp_plot, :], shading='gouraud', cmap='viridis')
plt.title('Espectrograma de la Señal de Respiración Filtrada')
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.ylim(0.05, 0.6) # Limitar el eje Y al rango de respiración
plt.colorbar(label='Magnitud (dB)')
plt.grid(True)


# Espectrograma de Latido
plt.subplot(2, 1, 2)
# En f_heart, selecciona solo las frecuencias dentro del rango de interés
freq_mask_heart_plot = (f_heart >= 0.7) & (f_heart <= 3.0) # Rango de visualización más amplio que el filtro
plt.pcolormesh(t_heart, f_heart[freq_mask_heart_plot], spectrogram_heart_dB[freq_mask_heart_plot, :], shading='gouraud', cmap='magma')
plt.title('Espectrograma de la Señal de Latido Filtrada')
plt.ylabel('Frecuencia (Hz)')
plt.xlabel('Tiempo (s)')
plt.ylim(0.7, 3.0) # Limitar el eje Y al rango de latido
plt.colorbar(label='Magnitud (dB)')
plt.grid(True)

plt.tight_layout()
plt.show()

# --- Extracción de la Frecuencia Dominante a lo largo del Tiempo (Opcional, para STFT) ---
# Definir los rangos de búsqueda de picos para la estimación temporal
search_low_resp, search_high_resp = 0.1, 0.5
search_low_heart, search_high_heart = 0.8, 2.5

# Frecuencias dominantes de respiración a lo largo del tiempo
dominant_resp_freqs = []
# Máscara para la búsqueda de picos dentro del rango de respiración
peak_search_mask_resp = (f_resp >= search_low_resp) & (f_resp <= search_high_resp)
freqs_for_peak_search_resp = f_resp[peak_search_mask_resp]

for i in range(spectrogram_resp.shape[1]): # Iterar sobre las columnas (ventanas de tiempo)
    spectrum_window = spectrogram_resp_dB[peak_search_mask_resp, i]
    if len(spectrum_window) > 0 and not np.all(np.isnan(spectrum_window)) and not np.all(np.isinf(spectrum_window)): # Asegurar que hay datos válidos
        peak_idx = np.argmax(spectrum_window)
        dominant_resp_freqs.append(freqs_for_peak_search_resp[peak_idx])
    else:
        dominant_resp_freqs.append(np.nan) # Si no hay datos válidos en el rango, añadir NaN

# Frecuencias dominantes de latido a lo largo del tiempo
dominant_heart_freqs = []
# Máscara para la búsqueda de picos dentro del rango de latido
peak_search_mask_heart = (f_heart >= search_low_heart) & (f_heart <= search_high_heart)
freqs_for_peak_search_heart = f_heart[peak_search_mask_heart]

for i in range(spectrogram_heart.shape[1]):
    spectrum_window = spectrogram_heart_dB[peak_search_mask_heart, i]
    if len(spectrum_window) > 0 and not np.all(np.isnan(spectrum_window)) and not np.all(np.isinf(spectrum_window)):
        peak_idx = np.argmax(spectrum_window)
        dominant_heart_freqs.append(freqs_for_peak_search_heart[peak_idx])
    else:
        dominant_heart_freqs.append(np.nan)


plt.figure(figsize=(15, 6))
plt.subplot(2, 1, 1)
plt.plot(t_resp, dominant_resp_freqs)
plt.title('Frecuencia de Respiración Dominante a lo largo del Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.grid(True)
plt.ylim(0.05, 0.6) # Mantener el rango de visualización
plt.axhspan(0.1, 0.5, color='red', alpha=0.1, label='Rango Esperado')
plt.legend()


plt.subplot(2, 1, 2)
plt.plot(t_heart, dominant_heart_freqs)
plt.title('Frecuencia de Latido Dominante a lo largo del Tiempo')
plt.xlabel('Tiempo (s)')
plt.ylabel('Frecuencia (Hz)')
plt.grid(True)
plt.ylim(0.7, 3.0) # Mantener el rango de visualización
plt.axhspan(0.8, 2.5, color='green', alpha=0.1, label='Rango Esperado')
plt.legend()

plt.tight_layout()
plt.show()
