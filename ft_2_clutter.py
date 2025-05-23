import numpy as np

print("Aplicando eliminación de clutter estático...")

# Calcular el promedio temporal para cada range bin
# El promedio se calcula a lo largo del eje 0 (tiempo/scanlines)
mean_profile = np.mean(all_radar_complex, axis=0)

# Restar el promedio de cada scanline
# NumPy broadcasteará mean_profile a cada fila de all_radar_complex
all_radar_complex_clutter_removed = all_radar_complex - mean_profile

print("Eliminación de clutter estático completa.")
print("Dimensiones de los datos con clutter removido:", all_radar_complex_clutter_removed.shape)

# --- Vuelve a calcular la Varianza con los datos sin clutter ---
variance_profile_cr = np.var(np.abs(all_radar_complex_clutter_removed), axis=0)

# --- Vuelve a Visualizar el Perfil de Varianza ---
plt.figure(figsize=(12, 6))
plt.plot(variance_profile_cr)
plt.title('Perfil de Varianza a lo largo de los Range Bins (Clutter Removido)')
xlabel_text = f'Índice del Range Bin (Distancia Relativa)'
plt.xlabel(xlabel_text) # Actualizado el label para reflejar que es post-clutter
plt.ylabel('Varianza de la Magnitud de la Señal (Clutter Removido)')
plt.grid(True)
plt.show()

# Vuelve a encontrar el índice del range bin con la máxima varianza
max_variance_bin_index_cr = np.argmax(variance_profile_cr)
print(f"\n(Clutter Removido) El Range Bin con la máxima varianza es el índice: {max_variance_bin_index_cr}")
print(f"(Clutter Removido) Valor de la máxima varianza: {variance_profile_cr[max_variance_bin_index_cr]}")

# Vuelve a seleccionar los datos del Range Bin elegido basado en el nuevo resultado
target_range_bin_data_cr = all_radar_complex_clutter_removed[:, max_variance_bin_index_cr]

print(f"(Clutter Removido) Dimensiones de los datos del Range Bin {max_variance_bin_index_cr}: {target_range_bin_data_cr.shape}")
