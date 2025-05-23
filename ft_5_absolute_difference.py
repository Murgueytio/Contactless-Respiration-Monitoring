import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

# --- 1. Extraer las Señales de Referencia ---
# La columna 0 de all_references es la Respiración_Ref
respiration_ref_signal = all_references[:, 0]
# La columna 1 de all_references es la Heartbeat_Ref
heartbeat_ref_signal = all_references[:, 1]

print(f"Dimensiones de la señal de referencia de respiración: {respiration_ref_signal.shape}")
print(f"Dimensiones de la señal de referencia de latido: {heartbeat_ref_signal.shape}")

# --- 2. Filtrar las Señales de Referencia ---

# Aplicar el mismo filtro paso banda de respiración a la señal de referencia de respiración
filtered_respiration_ref_signal = filtfilt(b_resp, a_resp, respiration_ref_signal)

# Aplicar el mismo filtro paso banda de latido a la señal de referencia de latido
filtered_heartbeat_ref_signal = filtfilt(b_heart, a_heart, heartbeat_ref_signal)

print(f"Dimensiones de la señal de referencia de respiración filtrada: {filtered_respiration_ref_signal.shape}")
print(f"Dimensiones de la señal de referencia de latido filtrada: {filtered_heartbeat_ref_signal.shape}")

# --- 3. Análisis en Frecuencia (FFT) de las Señales de Referencia Filtradas ---

# FFT para la señal de referencia de respiración
N_ref_resp = len(filtered_respiration_ref_signal)
fft_ref_resp = np.fft.fft(filtered_respiration_ref_signal)
frequencies_ref_resp = np.fft.fftfreq(N_ref_resp, 1/Fs)
positive_frequencies_ref_resp = frequencies_ref_resp[frequencies_ref_resp >= 0]
positive_fft_magnitude_ref_resp = np.abs(fft_ref_resp[frequencies_ref_resp >= 0])

# FFT para la señal de referencia de latido
N_ref_heart = len(filtered_heartbeat_ref_signal)
fft_ref_heart = np.fft.fft(filtered_heartbeat_ref_signal)
frequencies_ref_heart = np.fft.fftfreq(N_ref_heart, 1/Fs)
positive_frequencies_ref_heart = frequencies_ref_heart[frequencies_ref_heart >= 0]
positive_fft_magnitude_ref_heart = np.abs(fft_ref_heart[frequencies_ref_heart >= 0])

# --- 4. Visualización de los Espectros de Referencia Filtrados ---

plt.figure(figsize=(15, 10))

# Subplot para la referencia de respiración
plt.subplot(2, 1, 1)
plt.plot(positive_frequencies_ref_resp, positive_fft_magnitude_ref_resp)
plt.title('Espectro de Frecuencia de la Señal de Referencia de Respiración Filtrada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (Arbitraria)')
plt.xlim(0.05, 0.6) # Rango de visualización para respiración
plt.grid(True)
plt.axvspan(0.1, 0.5, color='red', alpha=0.2, label='Rango Respiración Esperada (0.1-0.5 Hz)')
plt.legend()
plt.tight_layout()

# Subplot para la referencia de latido
plt.subplot(2, 1, 2)
plt.plot(positive_frequencies_ref_heart, positive_fft_magnitude_ref_heart)
plt.title('Espectro de Frecuencia de la Señal de Referencia de Latido Filtrada')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Magnitud (Arbitraria)')
plt.xlim(0.7, 3.0) # Rango de visualización para latido
plt.grid(True)
plt.axvspan(0.8, 2.5, color='green', alpha=0.2, label='Rango Latido Esperado (0.8-2.5 Hz)')
plt.legend()
plt.tight_layout()

plt.show()

# --- 5. Estimar Frecuencias de Referencia y Comparar con las del Radar ---

# Estimar la frecuencia de respiración desde la referencia filtrada
respiration_ref_freq_mask_filtered = (positive_frequencies_ref_resp >= 0.1) & (positive_frequencies_ref_resp <= 0.5)
estimated_respiration_ref_freq = 0.0 # Valor por defecto si no se encuentra un pico
if np.any(respiration_ref_freq_mask_filtered):
    respiration_ref_frequencies_in_range_filtered = positive_frequencies_ref_resp[respiration_ref_freq_mask_filtered]
    respiration_ref_magnitude_in_range_filtered = positive_fft_magnitude_ref_resp[respiration_ref_freq_mask_filtered]
    if len(respiration_ref_magnitude_in_range_filtered) > 0:
        estimated_respiration_ref_freq = respiration_ref_frequencies_in_range_filtered[np.argmax(respiration_ref_magnitude_in_range_filtered)]

# Estimar la frecuencia de latido desde la referencia filtrada
heartbeat_ref_freq_mask_filtered = (positive_frequencies_ref_heart >= 0.8) & (positive_frequencies_ref_heart <= 2.5)
estimated_heartbeat_ref_freq = 0.0 # Valor por defecto si no se encuentra un pico
if np.any(heartbeat_ref_freq_mask_filtered):
    heartbeat_ref_frequencies_in_range_filtered = positive_frequencies_ref_heart[heartbeat_ref_freq_mask_filtered]
    heartbeat_ref_magnitude_in_range_filtered = positive_fft_magnitude_ref_heart[heartbeat_ref_freq_mask_filtered]
    if len(heartbeat_ref_magnitude_in_range_filtered) > 0:
        estimated_heartbeat_ref_freq = heartbeat_ref_frequencies_in_range_filtered[np.argmax(heartbeat_ref_magnitude_in_range_filtered)]


print("\n--- Comparación de Frecuencias ---")
print(f"**Respiración:**")
print(f"  Estimación por Radar: {estimated_respiration_freq_filtered:.2f} Hz ({estimated_respiration_freq_filtered * 60:.1f} RPM)")
print(f"  Estimación por Referencia: {estimated_respiration_ref_freq:.2f} Hz ({estimated_respiration_ref_freq * 60:.1f} RPM)")
print(f"  Diferencia Absoluta: {np.abs(estimated_respiration_freq_filtered - estimated_respiration_ref_freq):.2f} Hz")
if estimated_respiration_ref_freq != 0:
    print(f"  Error Porcentual: {np.abs((estimated_respiration_freq_filtered - estimated_respiration_ref_freq) / estimated_respiration_ref_freq) * 100:.2f}%")
else:
    print("  Error Porcentual: No se puede calcular (frecuencia de referencia es 0).")


print(f"\n**Latido Cardíaco:**")
print(f"  Estimación por Radar: {estimated_heartbeat_freq_filtered:.2f} Hz ({estimated_heartbeat_freq_filtered * 60:.1f} LPM)")
print(f"  Estimación por Referencia: {estimated_heartbeat_ref_freq:.2f} Hz ({estimated_heartbeat_ref_freq * 60:.1f} LPM)")
print(f"  Diferencia Absoluta: {np.abs(estimated_heartbeat_freq_filtered - estimated_heartbeat_ref_freq):.2f} Hz")
if estimated_heartbeat_ref_freq != 0:
    print(f"  Error Porcentual: {np.abs((estimated_heartbeat_freq_filtered - estimated_heartbeat_ref_freq) / estimated_heartbeat_ref_freq) * 100:.2f}%")
else:
    print("  Error Porcentual: No se puede calcular (frecuencia de referencia es 0).")
