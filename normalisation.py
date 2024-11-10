import os
import cv2
import numpy as np
from tqdm import tqdm
import shutil

def get_all_images(input_folder):
    """
    Récupère tous les chemins d'images dans l'arborescence complète
    """
    image_paths = []
    for root, _, files in os.walk(input_folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

def calculate_target_distribution(input_folder):
    """
    Calcule la distribution cible pour toutes les images dans tous les sous-dossiers
    """
    means_sum = np.zeros(3)
    stds_sum = np.zeros(3)
    n_images = 0
    
    print("Calcul de la distribution cible...")
    
    # Récupérer tous les chemins d'images dans l'arborescence
    image_paths = get_all_images(input_folder)
    
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            
            # Calculer moyenne et écart-type pour chaque canal de l'image
            means = np.mean(image, axis=(0,1))
            stds = np.std(image, axis=(0,1))
            
            means_sum += means
            stds_sum += stds
            n_images += 1
    
    # Calculer les moyennes des moyennes et des écarts-types
    target_means = means_sum / n_images
    target_stds = stds_sum / n_images
    
    print("\nDistribution cible calculée:")
    print(f"Moyennes RGB cibles: {target_means}")
    print(f"Écarts-types RGB cibles: {target_stds}")
    
    return target_means, target_stds

def create_output_structure(input_folder, output_folder):
    """
    Recrée la structure des dossiers dans le dossier de sortie
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # Parcourir l'arborescence et recréer la structure
    for root, dirs, _ in os.walk(input_folder):
        for dir in dirs:
            # Calculer le chemin relatif par rapport au dossier d'entrée
            relative_path = os.path.relpath(os.path.join(root, dir), input_folder)
            # Créer le même chemin dans le dossier de sortie
            os.makedirs(os.path.join(output_folder, relative_path), exist_ok=True)

def standardize_image(image, target_means, target_stds):
    """
    Standardise une image pour que sa distribution corresponde à la distribution cible
    """
    image = image.astype(np.float32)
    
    for i in range(3):
        channel = image[:,:,i]
        current_mean = np.mean(channel)
        current_std = np.std(channel)
        channel = (channel - current_mean) / (current_std + 1e-10)
        channel = channel * target_stds[i] + target_means[i]
        image[:,:,i] = channel
    
    return image

def process_dataset(input_folder, output_folder, target_means, target_stds):
    """
    Traite toutes les images en préservant la structure des dossiers
    """
    # Créer la structure des dossiers de sortie
    create_output_structure(input_folder, output_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    print("\nStandardisation des images...")
    
    # Récupérer tous les chemins d'images
    image_paths = get_all_images(input_folder)
    
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        
        if image is not None:
            # Calculer le chemin de sortie en préservant la structure
            relative_path = os.path.relpath(image_path, input_folder)
            output_path = os.path.join(output_folder, relative_path)
            
            # Créer le dossier parent si nécessaire
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Traiter l'image
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            standardized = standardize_image(image, target_means, target_stds)
            standardized = np.clip(standardized, 0, 255)
            standardized = standardized.astype(np.uint8)
            standardized = cv2.cvtColor(standardized, cv2.COLOR_RGB2BGR)
            
            # Sauvegarder l'image
            cv2.imwrite(output_path, standardized)

def verify_standardization(folder, target_means, target_stds):
    """
    Vérifie la standardisation sur toute l'arborescence
    """
    print("\nVérification de la standardisation...")
    
    total_diff_means = np.zeros(3)
    total_diff_stds = np.zeros(3)
    n_images = 0
    
    image_paths = get_all_images(folder)
    
    for image_path in tqdm(image_paths):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.astype(np.float32)
            
            means = np.mean(image, axis=(0,1))
            stds = np.std(image, axis=(0,1))
            
            total_diff_means += np.abs(means - target_means)
            total_diff_stds += np.abs(stds - target_stds)
            n_images += 1
    
    avg_diff_means = total_diff_means / n_images
    avg_diff_stds = total_diff_stds / n_images
    
    print("\nDifférences moyennes par rapport à la cible:")
    print(f"Différences moyennes: {avg_diff_means}")
    print(f"Différences écarts-types: {avg_diff_stds}")

if __name__ == "__main__":
    input_folder = "../Merged_Dataset"
    output_folder = "Normalized_Dataset"
    
    # Calculer la distribution cible
    target_means, target_stds = calculate_target_distribution(input_folder)
    
    # Standardiser toutes les images
    process_dataset(input_folder, output_folder, target_means, target_stds)
    
    # Vérifier la standardisation
    verify_standardization(output_folder, target_means, target_stds)