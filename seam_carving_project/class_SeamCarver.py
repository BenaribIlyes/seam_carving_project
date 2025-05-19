# Import necessary libraries

# To use the Numba JIT compiler
from numba import njit

# To use joblib for parallel processing
from joblib import Parallel, delayed

# For the seam carving algorithm
import cv2 # type: ignore
import numpy as np
import matplotlib.pyplot as plt

#For the entropy energy map
from skimage.filters import rank
from skimage.morphology import disk

# For the HoG energy map
from skimage.feature import hog
from skimage.color   import rgb2gray


# Extrcation of the remove_seam function to use Numba
@njit
def remove_seam_numba(image: np.ndarray, seam: np.ndarray, orientation: int = 0) -> np.ndarray:
    rows, cols, _ = image.shape

    if orientation == 0:  # vertical
        new_image = np.zeros((rows, cols - 1, 3), dtype=image.dtype)
        for i in range(rows):
            j = seam[i]
            for c in range(3):
                new_image[i, :j, c] = image[i, :j, c]
                new_image[i, j:, c] = image[i, j + 1:, c]

    elif orientation == 1:  # horizontal
        new_image = np.zeros((rows - 1, cols, 3), dtype=image.dtype)
        for j in range(cols):
            i = seam[j]
            for c in range(3):
                new_image[:i, j, c] = image[:i, j, c]
                new_image[i:, j, c] = image[i + 1:, j, c]

    else:
        raise ValueError("orientation must be 0 (vertical) or 1 (horizontal)")

    return new_image

# Compute the HoG energy map
# This function is called in parallel for each pixel
# It computes the HoG energy for a given pixel (y, x) using the local window around it
# It returns the pixel coordinates and the normalized HoG energy value
def compute_pixel_hog(y: int, x: int, angle: np.ndarray, magnitude: np.ndarray, half: int, bin_edges: np.ndarray):
    """
    Calcule l'énergie HoG d'un pixel (fonction appelée en parallèle)
    """
    rows, cols = angle.shape

    # 1. Fenêtre locale autour du pixel (clampée aux bords)
    y0, y1 = max(0, y - half), min(rows, y + half + 1)
    x0, x1 = max(0, x - half), min(cols, x + half + 1)

    # 2. Extraction locale
    patch_angle = angle[y0:y1, x0:x1].ravel()
    patch_mag = magnitude[y0:y1, x0:x1].ravel()

    # 3. Histogramme pondéré
    hist, _ = np.histogram(patch_angle, bins=bin_edges, weights=patch_mag)
    max_hist = hist.max() if hist.max() > 0 else 1.0

    # 4. Retourne la coordonnée + valeur normalisée
    return y, x, magnitude[y, x] / max_hist

# Compute the HoG energy map
# This function computes the HoG energy map for the entire image
# It uses the compute_pixel_hog function in parallel for each pixel
# It returns the HoG energy map
# It uses the joblib library to parallelize the computation
def compute_hog_custom_parallel(gray: np.ndarray, win_size: int = 15, nbins: int = 9) -> np.ndarray:
    """
    Version parallélisée du calcul HoG avec joblib
    """
    # 1. Gradient image
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    # 2. Paramètres de fenêtre et histogramme
    bin_edges = np.linspace(0, 180, nbins + 1)
    rows, cols = gray.shape
    half = win_size // 2

    # 3. Coordonnées de tous les pixels
    coords = [(y, x) for y in range(rows) for x in range(cols)]

    # 4. Appels parallèles à compute_pixel_hog()
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_pixel_hog)(y, x, angle, magnitude, half, bin_edges)
        for y, x in coords
    )

    # 5. Reconstruction de l'image finale
    eHoG = np.zeros((rows, cols), dtype=np.float64)
    for y, x, val in results:
        eHoG[y, x] = val

    return eHoG


class SeamCarver:
    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.float32)
        self.history = []  # To store removed seams

    def compute_energy(self, method: str = 'l1') -> np.ndarray:
        """
        Compute the energy map of the current image.
        Supported methods: 'sobel', 'l1', 'l2', 'entropy', 'hog', ...
        """
        gray = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if method == 'l1':
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            energy = np.abs(gx) + np.abs(gy) # Norm-L1  

        elif method == 'l2':
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            energy = np.hypot(gx, gy) # Euclidian norm, Norm-L2
     
        elif method == 'entropy':
            #1. Compute the histogram of the image
            hist = cv2.calcHist([gray], [0], None, [256], [0,256]).ravel()
            #2. Compute the entropy, with the histogram normalized
            #   to get the probability of each pixel value
            #   and then compute the entropy
            #   H = -sum(p(x) * log2(p(x)))  
            p = hist / hist.sum()
            p = p[p>0] # remove zero values to avoid log(0)
            H = -(p * np.log2(p)).sum()
            # on remplit la carte d'énergie avec cette valeur constante
            energy = np.full_like(gray, H, dtype=np.float32)

        elif method == 'HoG': # Compute the HoG energy map eHoG(x,y) = (|Ix|+|Iy|) / max(HoG_local(x,y))
            #  Compute the grad X/Y, magnitude & degree orientation [0,180)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
            ang = (np.degrees(np.arctan2(gy, gx)) + 360) % 180

            # For each pixels we extract the window to compute the histogram ponderate by the magnitude.
            for y in range(rows):
                y0, y1 = max(0, y-half), min(rows, y+half+1)
                for x in range(cols):
                    x0, x1 = max(0, x-half), min(cols, x+half+1)
                    patch_ang = ang[y0:y1, x0:x1].ravel()
                    patch_mag = mag[y0:y1, x0:x1].ravel()
            
                    # ponderate histogram: we use magnitude as weight
                    hist, _ = np.histogram(
                        patch_ang,
                        bins=bin_edges,
                        weights=patch_mag,
                        density=False
                    )# Practicle rule: for an image of less than 500 pixels on the smallest side use 8 bins
                    max_hist = hist.max() if hist.max()>0 else 1.0
            
                    # Energie eHoG
                    eHoG[y, x] = (mag[y, x]) / max_hist
            
            # 1. Calcul du gradient en x et y avec Sobel
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # dérivée horizontale
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # dérivée verticale

            # 2. Calcul de la norme et de l'orientation du gradient
            magnitude = np.hypot(gx, gy)  # norme euclidienne du gradient
            angle = (np.degrees(np.arctan2(gy, gx)) + 180) % 180  # angle ∈ [0, 180)

            # 3. Paramètres pour l'histogramme
            # To prepare the bin (angle intervals) and the win_size (size of the local window to see the orientation of the histigram)
            # Practicle rule: for an image of less than 500 pixels on the smallest side use 8 bins
            # Practicle rule: win_size = alpha  * min(rows,cols) with apha between 0.01 and 0.05
            nbins = 12 # Default Value about 22.5° 
            win_size = 15
            bin_edges = np.linspace(0, 180, nbins + 1)  # intervalles de quantification angulaire
            rows, cols = gray.shape
            half = win_size // 2  # moitié de la fenêtre

            # 4. Image de sortie initialisée à 0
            eHoG = np.zeros((rows, cols), dtype=np.float64)

            # 5. Parcours de chaque pixel de l'image
            for y in range(rows):
                y0, y1 = max(0, y - half), min(rows, y + half + 1)  # bornes verticales de la fenêtre

            for x in range(cols):
                x0, x1 = max(0, x - half), min(cols, x + half + 1)  # bornes horizontales de la fenêtre

                # 6. Extraction des angles et magnitudes dans la fenêtre locale
                patch_angle = angle[y0:y1, x0:x1].ravel()
                patch_mag = magnitude[y0:y1, x0:x1].ravel()

                # 7. Calcul de l'histogramme pondéré par les magnitudes
                hist, _ = np.histogram(patch_angle, bins=bin_edges, weights=patch_mag)

                # 8. Normalisation par la valeur max de l'histogramme
                max_hist = hist.max() if hist.max() > 0 else 1.0

                # 9. Calcul de l'énergie HoG locale
                eHoG[y, x] = magnitude[y, x] / max_hist

            energy = eHoG
        
        elif method == 'fast_HOG':
            return compute_hog_custom_parallel(gray)  # version rapide

        elif method == 'skimage_entropy':
            gray_u8 = gray  # already uint8
            ent = rank.entropy(gray_u8, disk(5))  # 11×11 window
            energy = ent.astype(np.float64)

        elif method == 'skimage_HoG':
            gray_f = rgb2gray(self.image.astype(np.uint8))
            hog_feats, _ = hog(
                gray_f,
                orientations=12,
                pixels_per_cell=(16,16),
                cells_per_block=(1,1),
                block_norm='L2-Hys',
                visualize=True,
                feature_vector=False
            )
            local_max = np.max(hog_feats[...,0,0,:], axis=-1)
            denom = cv2.resize(local_max, (gray.shape[1], gray.shape[0]),
                            interpolation=cv2.INTER_NEAREST)
            gx = cv2.Sobel(gray_f, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray_f, cv2.CV_64F, 0, 1)
            mag = np.hypot(gx, gy)
            energy=  mag / (denom + 1e-6)
        
        elif method == 'saliency':
            # Création du détecteur de saillance
            saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()

            # Calcul de la carte de saillance
            success, saliencyMap = saliency_detector.computeSaliency(self.image)

            if not success:
                    raise RuntimeError("Échec du calcul de la carte de saillance.")

            # Normalisation
            saliencyMap = (saliencyMap * 255).astype("uint8")

            # Conversion en float64 pour être cohérent avec les autres méthodes
            energy = saliencyMap.astype(np.float64)

        elif method == 'combined_sobel_saliency':
             # 1. Sobel (norme L2)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            sobel_energy = gx**2 + gy**2

            # 2. Saliency
            saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliencyMap = saliency_detector.computeSaliency(self.image)
            if not success:
                raise RuntimeError("Erreur lors du calcul de la carte de saillance.")
            saliencyMap = (saliencyMap * 255).astype("float64")

             # 3. Fusion linéaire
            energy = 0.5 * sobel_energy + 0.5 * saliencyMap


        else:
            raise ValueError(f"compute_energy: méthode inconnue '{method}'")
        
        return energy

    def find_seam(self, image: np.ndarray, energy: np.ndarray, orientation: int = 1) -> np.ndarray:
        """
        Find the minimal-energy seam in the given orientation.
        Returns an array of indices (row -> col) for vertical, or (col -> row) for horizontal.
        """
         #  Here we compute the energy for all the image every tiùe but doing it localy will be better (for only around the seams left)
        if orientation == 1: # horizontal
            energy = np.transpose(energy)
            image = np.transpose(image)
    
        rows, cols = energy.shape
        seam = np.zeros((rows,), dtype=int) # Initialize the seam array
        dp = np.zeros((rows, cols), dtype=np.float32) # Initialize the dynamic programming table
        dp[0] = energy[0] # The first row is just the energy of the first row

        for i in range(1, rows):
                for j in range(cols):
                    if j == 0: ## If we are at the first column
                        dp[i][j] = energy[i][j] + min(dp[i-1][j], dp[i-1][j+1])
                    elif j == cols - 1: ## If we are at the last column
                        dp[i][j] = energy[i][j] + min(dp[i-1][j-1], dp[i-1][j])
                    else: # If we are in the middle columns
                        dp[i][j] = energy[i][j] + min(dp[i-1][j-1], dp[i-1][j], dp[i-1][j+1])

        seam[-1] = np.argmin(dp[-1]) # The last element of the seam is the column with the minimum value in the last row of dp

        for i in range(rows - 2, -1, -1):
            j = seam[i + 1] 
            if j == 0:
                seam[i] = j if dp[i][j] < dp[i][j + 1] else j + 1
            elif j == cols - 1:
                seam[i] = j if dp[i][j] < dp[i][j - 1] else j - 1
            else:
                seam[i] = j if dp[i][j] < min(dp[i][j - 1], dp[i][j + 1]) else (j - 1 if dp[i][j - 1] < dp[i][j + 1] else j + 1)
            
        return seam

#   def remove_seam(self,image: np.ndarray, seam: np.ndarray) -> np.ndarray:
#        """
#        Remove the seam from the image.
#        """
#        rows, cols, _ = image.shape # Get the dimensions of the image
#        new_image = np.zeros((rows, cols - 1, 3), dtype=image.dtype) # Create a new image with one less column
#
#       for i in range(rows):
#            j = seam[i]
#            new_image[i, :j] = image[i, :j] # Copy the pixels to the left of the seam
#            new_image[i, j:] = image[i, j + 1:] # Copy the pixels to the right of the seam
#
#        return new_image 

    def remove_seam(self, image: np.ndarray, seam: np.ndarray, orientation: int) -> np.ndarray:
        return remove_seam_numba(image, seam, orientation)       

    def seam_carve(self, num_seams: int, method: str = 'l1', orientation: str = 'vertical') -> np.ndarray:
        """
        Découpe num_seams coutures verticales dans self.image.
        Retourne l'image retarotée.
        """
        for _ in range(num_seams):
            # 1) Compute the energy map
            energy = self.compute_energy(method)
            # 2) Find the seam
            orientation_flag = 0 if orientation == 'vertical' else 1
            seam = self.find_seam(self.image, energy, orientation=orientation_flag)
            # 3) Store the seam in self.history
            self.history.append(seam)
            # 4) Remove the seam from the image
            self.image = self.remove_seam(self.image, seam, orientation=orientation_flag)
            # 5) Show the seam on the image
            #self.show_history(self.image, seam)


        return self.image

    def show_history(self, seam: np.ndarray) -> None:
        """
        Show the history of removed seams.
        """ 
        for i, seam in enumerate(self.history):
            plt.imshow(self.image.astype(np.uint8))
            plt.plot(seam, range(len(seam)), 'r-')
            plt.title(f"Seam {i+1}")
            plt.show()

    def find_best_seam(self, seam: np.ndarray, energy: np.ndarray) -> None:
        """
        Find the best seam to remove between the horizontal and vertical seams.
        """
        # Compute each seam
        seam_vertical = self.find_seam(self.image, seam, orientation=0)
        seam_horizontal = self.find_seam(self.image, seam, orientation=1)
        # Choose the best seam to remove
        seam_to_remove = seam_vertical if seam_vertical.sum() < seam_horizontal.sum() else seam_horizontal
        kind = 'vertical' if seam_vertical.sum() < seam_horizontal.sum() else 'horizontal'
        
        return seam_to_remove, kind
    
    def seam_carve_optimal(self, num_seams: int, method: str = 'l1') -> np.ndarray:
        """
        Découpe num_seams coutures verticales dans self.image.
        Retourne l'image retarotée.
        """
        for _ in range(num_seams):
            # 1) Compute the energy map
            energy = self.compute_energy(method)
            # 2) Find the best seam to remove beween the horizontal and vertical seams
            seam,kind = self.find_best_seam(self, seam)
            # 3) Store the seam in self.history
            self.history.append(seam)
            # 4) Remove the seam from the image
            self.image = self.remove_seam(self.image, seam, kind)
            # 5) Show the seam on the image
            self.show_history(self.image, seam)

        return self.image
    
    def add_seam(self, image: np.ndarray, seam: np.ndarray, orientation: int) -> np.ndarray:
        """
        Add the seam to the image.
        """
        rows, cols, _ = image.shape

        if orientation == 0:
            new_image = np.zeros((rows, cols + 1, 3), dtype=image.dtype)
            for i in range(rows):                
                j = seam[i]
                new_image[i, :j] = image[i, :j]
                new_image[i, j] = image[i, j]
                new_image[i, j + 1:] = image[i, j:]
        elif orientation == 1:
            new_image = np.zeros((rows+1, cols, 3), dtype=image.dtype)
            for j in range(cols):
                i = seam[j]
                new_image[:i, j] = image[:i, j]
                new_image[i, j] = image[i, j]
                new_image[i + 1:, j] = image[i:, j]
        else:
            print("Error: orientation must be 0 (vertical) or 1 (horizontal)")
        return new_image

    def upsize(self, nums_seams: int, method: str = 'l1', orientation: str = 'vertical') -> np.ndarray:
        """
        Upsize the image by adding num_seams seams.
        """
        for _ in range(nums_seams):
            # 1) Compute the energy map
            energy = self.compute_energy(method)
            # 2) Find the seam
            orientation_flag = 0 if orientation == 'vertical' else 1
            seam = self.find_seam(self.image, energy, orientation=orientation_flag)
            # 3) Store the seam in self.history
            self.history.append(seam)
            # 4) Add the seam to the image
            self.image = self.add_seam(self.image, seam, orientation=orientation_flag)
            # 5) Show the seam on the image
            #self.show_history(self.image, seam)

        return self.image
    
def compare_saliency_preservation(original: np.ndarray, reduced_images: dict) -> dict:
    """
    Compare la préservation de la saillance entre l'image originale et plusieurs versions réduites.

    Args:
        original (np.ndarray): Image d'origine.
        reduced_images (dict): Dictionnaire {nom_méthode: image_réduite}.

    Returns:
        dict: Dictionnaire {nom_méthode: score}.
    """
    # Utilise compute_energy de SeamCarver pour obtenir la carte de saillance
    sc_ref = SeamCarver(original.copy())
    saliency_orig = sc_ref.compute_energy('saliency')  # ⚠️ le nom de méthode doit être 'saliency' (tout en minuscule)

    results = {}

    for label, reduced in reduced_images.items():
        # Redimensionner la carte de saillance à la taille réduite
        saliency_resized = cv2.resize(saliency_orig, (reduced.shape[1], reduced.shape[0]))

        # Convertir l'image réduite en niveaux de gris
        gray_reduced = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)

        # Produit pondéré : approximation du chevauchement entre salience et contraste
        score = np.sum((gray_reduced / 255.0) * (saliency_resized / 255.0))

        results[label] = score

    # Affichage des scores
    print("\nComparaison de la préservation de la saillance :")
    for label, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{label:25s} → score de saillance préservée : {score:.2f}")

    return results
