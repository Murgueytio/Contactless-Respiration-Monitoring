from scipy.signal import butter, filtfilt
import numpy as np
import matplotlib.pyplot as plt
from numpy import unwrap # Corrected import as per your feedback

# --- Paso 3: Filtrado Paso Banda de la Señal de Fase ---
# Frecuencia de Nyquist (máxima frecuencia representable)
nyquist_freq = 0.5 * Fs

# --- Filtro para Respiración ---
# Rango de frecuencia para respiración: 0.1 Hz a 0.5 Hz
lowcut_resp = 0.1
highcut_resp = 0.5
# Normalizar las frecuencias de corte a la frecuencia de Nyquist
# Esto es un requisito para las funciones de diseño de filtros de SciPy
norm_lowcut_resp = lowcut_resp / nyquist_freq
norm_highcut_resp = highcut_resp / nyquist_freq

# Diseñar el filtro Butterworth de paso banda (orden 4, un buen balance)
# b: coeficientes del numerador del filtro, a: coeficientes del denominador
b_resp, a_resp = butter(4, [norm_lowcut_resp, norm_highcut_resp], btype='band')

# Aplicar el filtro a la señal de fase.
# filtfilt aplica el filtro dos veces (hacia adelante y hacia atrás) para evitar desplazamiento de fase.
filtered_respiration_signal = filtfilt(b_resp, a_resp, vital_sign_phase_signal)

print(f"Dimensiones de la señal de respiración filtrada: {filtered_respiration_signal.shape}")


# --- Filtro para Latido Cardíaco ---
# Rango de frecuencia para latido: 0.8 Hz a 2.5 Hz
lowcut_heart = 0.8
highcut_heart = 2.5
norm_lowcut_heart = lowcut_heart / nyquist_freq
norm_highcut_heart = highcut_heart / nyquist_freq

b_heart, a_heart = butter(4, [norm_lowcut_heart, norm_highcut_heart], btype='band')
filtered_heartbeat_signal = filtfilt(b_heart, a_heart, vital_sign_phase_signal)

print(f"Dimensiones de la señal de latido filtrada: {filtered_heartbeat_signal.shape}")

# --- Paso 4: Re-Análisis en Frecuencia (FFT) de las Señales Filtradas ---

# Calcular la FFT para la señal de respiración filtrada
N_resp = len(filtered_respiration_signal)
fft_respiration = np.fft.fft(filtered_respiration_signal)
frequencies_resp = np.fft.fftfreq(N_resp, 1/Fs)
positive_frequencies_resp = frequencies_resp[frequencies_resp >= 0]
positive_fft_magnitude_resp = np.abs(fft_respiration[frequencies_resp >= 0])

# Calcular la FFT para la señal de latido filtrada
N_heart = len(filtered_heartbeat_signal)
fft_heartbeat = np.fft.fft(filtered_heartbeat_signal)
frequencies_heart = np.fft.fftfreq(N_heart, 1/Fs)
positive_frequencies_heart = frequencies_heart[frequencies_heart >= 0]
positive_fft_magnitude_heart = np.abs(fft_heartbeat[frequencies_heart >= 0])


# --- Paso 5: Visualización de los Espectros Filtrados ---

plt.figure(figsize=(15, 10))

# Subplot para la respiración (filtrada)
plt.subplot(2, 1, 1)
plt.plot(positive_frequencies_resp, positive_fft_magnitude_resp)
plt.title('Espectro de Frecuencia de la Señal de Respiración Filtrada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (Arbitraria)')
plt.xlim(0.05, 0.6) # Rango de visualización para respiración
plt.grid(True)
plt.axvspan(0.1, 0.5, color='red', alpha=0.2, label='Rango Respiración Esperada (0.1-0.5 Hz)')
plt.legend()
plt.tight_layout()

# Subplot para el latido (filtrado)
plt.subplot(2, 1, 2)
plt.plot(positive_frequencies_heart, positive_fft_magnitude_heart)
plt.title('Espectro de Frecuencia de la Señal de Latido Filtrada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (Arbitraria)')
plt.xlim(0.7, 3.0) # Rango de visualización para latido
plt.grid(True)
plt.axvspan(0.8, 2.5, color='green', alpha=0.2, label='Rango Latido Esperado (0.8-2.5 Hz)')
plt.legend()
plt.tight_layout()

plt.show()

# --- Paso 6: Re-Estimación de Frecuencias a partir de los Espectros Filtrados ---

# Estimar la frecuencia de respiración desde la señal filtrada
respiration_freq_mask_filtered = (positive_frequencies_resp >= 0.1) & (positive_frequencies_resp <= 0.5)
if np.any(respiration_freq_mask_filtered):
    respiration_frequencies_in_range_filtered = positive_frequencies_resp[respiration_freq_mask_filtered]
    respiration_magnitude_in_range_filtered = positive_fft_magnitude_resp[respiration_freq_mask_filtered]
    estimated_respiration_freq_filtered = respiration_frequencies_in_range_filtered[np.argmax(respiration_magnitude_in_range_filtered)]
    print(f"\nFrecuencia de Respiración Estimada (FILTRADA): {estimated_respiration_freq_filtered:.2f} Hz")
    print(f" (o {estimated_respiration_freq_filtered * 60:.1f} respiraciones por minuto)")
else:
    print("\nNo se detectó un pico claro de respiración en el rango filtrado.")

# Estimar la frecuencia de latido desde la señal filtrada
heartbeat_freq_mask_filtered = (positive_frequencies_heart >= 0.8) & (positive_frequencies_heart <= 2.5)
if np.any(heartbeat_freq_mask_filtered):
    heartbeat_frequencies_in_range_filtered = positive_frequencies_heart[heartbeat_freq_mask_filtered]
    heartbeat_magnitude_in_range_filtered = positive_fft_magnitude_heart[heartbeat_freq_mask_filtered]
    estimated_heartbeat_freq_filtered = heartbeat_frequencies_in_range_filtered[np.argmax(heartbeat_magnitude_in_range_filtered)]
    print(f"Frecuencia de Latido Estimada (FILTRADA): {estimated_heartbeat_freq_filtered:.2f} Hz")
    print(f" (o {estimated_heartbeat_freq_filtered * 60:.1f} latidos por minuto)")
else:
    print("No se detectó un pico claro de latido en el rango filtrado.")
