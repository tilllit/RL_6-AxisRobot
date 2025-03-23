import os
import imageio

# Liste der Ordner, aus denen die Bilder geladen werden sollen
folders = ['Success', 'TCP']

# Liste für alle Bilddateien (vollständige Pfade)
files = []

for folder in folders:
    # Überprüfen, ob der Ordner existiert
    if os.path.exists(folder):
        # Alle Bilddateien (PNG, JPG, JPEG) im aktuellen Ordner erfassen
        folder_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        files.extend(folder_files)
    else:
        print(f"Ordner {folder} existiert nicht!")

# Sortieren aller Dateien nach Änderungsdatum (übergreifend in allen Ordnern)
files.sort(key=lambda f: os.path.getmtime(f))

# Bilder einlesen
images = []
for file in files:
    images.append(imageio.imread(file))

# Erstellen des GIFs; die Dauer (in Sekunden) pro Frame kann angepasst werden
output_path = 'FullGIF_withoutTraj.gif'
imageio.mimsave(output_path, images, duration=0.5)

print(f"GIF wurde erfolgreich erstellt: {output_path}")
