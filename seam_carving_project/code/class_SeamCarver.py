# Import necessary libraries

# To use the Numba JIT compiler
from numba import njit

# To use joblib for parallel processing
from joblib import Parallel, delayed

# For the seam carving algorithm
import cv2 
import numpy as np
import matplotlib.pyplot as plt


#For the entropy energy map
from skimage.filters import rank
from skimage.morphology import disk

# For the HoG energy map
from skimage.feature import hog
from skimage.color   import rgb2gray
import time


# Extrcation of the remove_seam function to use Numba
@njit
def remove_seam_numba(image: np.ndarray, seam: np.ndarray, orientation: str = 'vertical') -> np.ndarray:
    """Remove a seam from the image using Numba for performance.
    Args:
        image (np.ndarray): The input image from which the seam will be removed.
        seam (np.ndarray): The seam to be removed, represented as an array of indices.
        orientation (str): The orientation of the seam ('vertical' or 'horizontal').
    Returns:
        np.ndarray: The new image with the seam removed.
    """

    if image.ndim == 2:
        rows, cols = image.shape
        if orientation == 'vertical':
            new_image = np.zeros((rows, cols - 1), dtype=image.dtype)
            for i in range(rows):
                j = seam[i]
                new_image[i, :j] = image[i, :j]
                new_image[i, j:] = image[i, j + 1:]
        elif orientation == 'horizontal':
            new_image = np.zeros((rows - 1, cols), dtype=image.dtype)
            for j in range(cols):
                i = seam[j]
                new_image[:i, j] = image[:i, j]
                new_image[i:, j] = image[i + 1:, j]
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
    else:  # RGB image
        rows, cols, _ = image.shape
        if orientation == 'vertical':
            new_image = np.zeros((rows, cols - 1, 3), dtype=image.dtype)
            for i in range(rows):
                j = seam[i]
                for c in range(3):
                    new_image[i, :j, c] = image[i, :j, c]
                    new_image[i, j:, c] = image[i, j + 1:, c]
        elif orientation == 'horizontal':
            new_image = np.zeros((rows - 1, cols, 3), dtype=image.dtype)
            for j in range(cols):
                i = seam[j]
                for c in range(3):
                    new_image[:i, j, c] = image[:i, j, c]
                    new_image[i:, j, c] = image[i + 1:, j, c]
        else:
            raise ValueError("orientation must be 'vertical' or 'horizontal'")
    
    return new_image

# Compute the HoG energy map
# This function is called in parallel for each pixel
# It computes the HoG energy for a given pixel (y, x) using the local window around it
# It returns the pixel coordinates and the normalized HoG energy value
def compute_pixel_hog(y: int, x: int, angle: np.ndarray, magnitude: np.ndarray, half: int, bin_edges: np.ndarray):
    """
    Compute the HoG energy for a single pixel (y, x) using the local window around it.
    """
    rows, cols = angle.shape

    # 1. Definition of the local window
    y0, y1 = max(0, y - half), min(rows, y + half + 1)
    x0, x1 = max(0, x - half), min(cols, x + half + 1)

    # 2. Extraction of the angles and magnitudes in the local window
    patch_angle = angle[y0:y1, x0:x1].ravel()
    patch_mag = magnitude[y0:y1, x0:x1].ravel()

    # 3. Calculation of the weighted histogram
    hist, _ = np.histogram(patch_angle, bins=bin_edges, weights=patch_mag)
    max_hist = hist.max() if hist.max() > 0 else 1.0

    # 4. Calculation of the HoG energy
    return y, x, magnitude[y, x] / max_hist

# Compute the HoG energy map
# This function computes the HoG energy map for the entire image
# It uses the compute_pixel_hog function in parallel for each pixel
# It returns the HoG energy map
# It uses the joblib library to parallelize the computation, we realise later that the skimage library has a built-in function to compute the HoG energy map, so we decide to use it instead
def compute_hog_custom_parallel(gray: np.ndarray, win_size: int = 15, nbins: int = 9) -> np.ndarray:
    """
    Version parallélisée du calcul HoG avec joblib
    """
    # 1. Gradient image
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    angle = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    # 2. Parameters fot the histogram
    bin_edges = np.linspace(0, 180, nbins + 1)
    rows, cols = gray.shape
    half = win_size // 2

    # 3. Coordinates for parallel processing
    coords = [(y, x) for y in range(rows) for x in range(cols)]

    # 4. Parallel computation of HoG energy for each pixel
    results = Parallel(n_jobs=-1, prefer="threads")(
        delayed(compute_pixel_hog)(y, x, angle, magnitude, half, bin_edges)
        for y, x in coords
    )

    # 5. Construct the HoG energy map
    eHoG = np.zeros((rows, cols), dtype=np.float64)
    for y, x, val in results:
        eHoG[y, x] = val

    return eHoG

def compare_saliency_preservation(original: np.ndarray, reduced_images: dict) -> dict:
    """
    Compare the preservation of saliency between the original image and several reduced versions.

    Args:
        original (np.ndarray): Original image.
        reduced_images (dict): Dictionary {method_name: reduced_image}.

    Returns:
       dict: Dictionary {method_name: score}.
    """
    # Use SeamCarver's compute_energy to get the saliency map
    sc_ref = SeamCarver(original)
    saliency_orig = sc_ref.compute_energy('saliency',original)  # method name must be 'saliency' (all lowercase)

    results = {}

    for label, reduced in reduced_images.items():
        # Resize the saliency map to the reduced image size
        saliency_resized = cv2.resize(saliency_orig, (reduced.shape[1], reduced.shape[0]))
        # Convert the reduced image to grayscale
        gray_reduced = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)

        # Weighted product: approximation of the overlap between saliency and contrast
        score = np.sum((gray_reduced / 255.0) * (saliency_resized / 255.0))

        results[label] = score

        #   Display scores
        for label, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
            print(f"{label:25s} → preserved saliency score: {score:.2f}")

    return results


class SeamCarver:
    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.float32)
        self.history = []  # To store removed seams
        self.orientation_history = []  # To store the orientation of removed seams
        self.original_copy = image.copy()  # Keep a copy of the original image

    def compute_energy(self, method: str = 'l1', image: np.ndarray = None) -> np.ndarray:
        """
        Compute the energy map of the current image.
        Supported methods: 'sobel', 'l1', 'l2', 'entropy', 'hog', ...
        """

        gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
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

            # 4. Initialization of the HoG energy map
            eHoG = np.zeros((rows, cols), dtype=np.float64)

            # 5. Loop over each pixel in the image
            for y in range(rows):
                y0, y1 = max(0, y - half), min(rows, y + half + 1)  # vertical bounds of the window

            for x in range(cols):
                x0, x1 = max(0, x - half), min(cols, x + half + 1)  # horizontal bounds of the window

                # 6. Extract angles and magnitudes in the local window
                patch_angle = angle[y0:y1, x0:x1].ravel()
                patch_mag = magnitude[y0:y1, x0:x1].ravel()

                # 7. Compute the histogram weighted by magnitudes
                hist, _ = np.histogram(patch_angle, bins=bin_edges, weights=patch_mag)

                # 8. Normalize by the max value of the histogram
                max_hist = hist.max() if hist.max() > 0 else 1.0

                # 9. Compute the local HoG energy
                eHoG[y, x] = magnitude[y, x] / max_hist

            energy = eHoG
        
        elif method == 'fast_HOG':
            return compute_hog_custom_parallel(gray)  # fast version

        elif method == 'skimage_entropy':
            gray_u8 = gray  # already uint8
            ent = rank.entropy(gray_u8, disk(5))  # 11×11 window
            energy = ent.astype(np.float64)

        elif method == 'skimage_HOG':
            gray_f = rgb2gray(image.astype(np.uint8))
            h, w = gray_f.shape

            # Define the cell size based on the image dimensions
            # Practicle rule: for an image of less than 500 pixels on the smallest side use 8 bins
            cell_h = min(16, max(1, h // 2))
            cell_w = min(16, max(1, w // 2))

            hog_feats, _ = hog(
            gray_f,
            orientations=12,
            pixels_per_cell=(cell_h, cell_w),
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
            # Create the saliency detector
            saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()

            # Compute the saliency map
            success, saliencyMap = saliency_detector.computeSaliency(self.image)

            if not success:
                raise RuntimeError("Failed to compute the saliency map.")

            # Normalization
            saliencyMap = (saliencyMap * 255).astype("uint8")

            # Convert to float64 to be consistent with other methods
            energy = saliencyMap.astype(np.float64)

        elif method == 'combined_sobel_saliency':
             # 1. Sobel (L2 norm)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            sobel_energy = gx**2 + gy**2

            # 2. Saliency
            saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
            success, saliencyMap = saliency_detector.computeSaliency(self.image)
            if not success:
                raise RuntimeError("Error while computing the saliency map.")
            saliencyMap = (saliencyMap * 255).astype("float64")

             # 3. Linear fusion
            energy = 0.5 * sobel_energy + 0.5 * saliencyMap


        else:
            raise ValueError(f"compute_energy: méthode inconnue '{method}'")
        
        return energy

    def find_seam(self, image: np.ndarray, energy: np.ndarray, orientation: str, return_cost) -> np.ndarray:
        """
        Find the minimal-energy seam in the given orientation.
        Returns an array of indices (row -> col) for vertical, or (col -> row) for horizontal.
        """
         #  Here we compute the energy for all the image every time but doing it localy will be better (for only around the seams left)
        if orientation == 'horizontal': # horizontal
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
            
        cost = dp[-1, seam[-1]]
        if return_cost:
            return seam, cost
        else:
            return seam
        return seam
    
#  This function is used to remove the seam from the image
#  It is commented out because we use the numba version instead

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

    def remove_seam(self, image: np.ndarray, seam: np.ndarray, orientation: str = 'vertical') -> np.ndarray:
        return remove_seam_numba(image, seam, orientation)       

    def seam_carve(self, num_seams: int, method: str = 'l1', orientation: str = 'vertical') -> np.ndarray:
        """
        Removes num_seams vertical seams from self.image.
        Returns the retargeted image.
        """
        for _ in range(num_seams):
            # 1) Compute the energy map
            energy = self.compute_energy(method)
            # 2) Find the seam
            orientation_flag = 0 if orientation == 'vertical' else 1
            seam = self.find_seam(self.image, energy, orientation,return_cost=False)
            # 3) Store the seam in self.history
            self.history.append(seam)
            # 4) Remove the seam from the image
            self.image = self.remove_seam(self.image, seam, orientation)

        # Show the seam on the image
        self.show_history()
        print(f"Seam carving with {num_seams} seams removed in {orientation} orientation.")
        return self.image

    def show_history(self) -> None:
        """
        Display all seams that have been removed from the image.
        Red lines = vertical seams, blue lines = horizontal seams.
        """
        img = self.image.copy().astype(np.uint8)
        plt.imshow(img)

        # Show all removed seams with different colors for each orientation
        for idx, seam in enumerate(self.history):
            # Get the orientation for this seam
            if len(self.orientation_history) > idx:
                orientation = self.orientation_history[idx]
            else:
                orientation = 'vertical'  # Default if not available

            if orientation == 'vertical':
            # Draw vertical seam in red
                for i in range(len(seam)):
                    plt.scatter(seam[i], i, color='red', s=1)
            elif orientation == 'horizontal':
            # Draw horizontal seam in blue
                for i in range(len(seam)):
                    plt.scatter(i, seam[i], color='blue', s=1)
        plt.title("Removed seams (red: vertical, blue: horizontal)")
        plt.show()

    def find_best_seam(self, image: np.ndarray, energy: np.ndarray) -> None:
        """
        Find the best seam to remove between the horizontal and vertical seams.
        """
        # Compute each seam
        seam_v, cost_v = self.find_seam(image, energy, 'vertical',   return_cost=True)
        seam_h, cost_h = self.find_seam(image, energy, 'horizontal', return_cost=True)
        # Choose the best seam to remove
        if cost_v < cost_h:
            return seam_v, 'vertical'
        else:
            return seam_h, 'horizontal'
        
    
    def optimal_seam_carve(self, num_seams: int, method: str = 'l1') -> np.ndarray:
        """
        Removes num_seams vertical or horizontal seams from self.image based on the energy.
        Returns the retargeted image.
        This method alternates between vertical and horizontal seams to minimize the total energy.
        """
        start_time = time.time()
        for _ in range(num_seams):
            # 1) Compute the energy map
            energy = self.compute_energy(method, self.image)
            # 2) Find the best seam to remove beween the horizontal and vertical seams
            seam,orientation = self.find_best_seam(self.image, energy)
            # 3) Store the seam in self.history and the orientation in self.orientation.history
            self.history.append(seam)
            self.orientation_history.append(orientation)
            # 4) Remove the seam from the image
            self.image = self.remove_seam(self.image, seam, orientation)
        # copy the image to return
        output_image = self.image

        #reset the image to the original copy
        self.image = self.original_copy.copy()
        
        elapsed_time = time.time() - start_time
        print(f"Optimal seam carving with {num_seams} seams removed in {elapsed_time:.2f} seconds.")
        
        return output_image
    
    def add_seam(self, image: np.ndarray, seam: np.ndarray, orientation: str = 'vertical') -> np.ndarray:
        """
        Add the seam to the image.
        """
        rows, cols, _ = image.shape

        if orientation =='vertical':  # vertical
            new_image = np.zeros((rows, cols + 1, 3), dtype=image.dtype)
            for i in range(rows):
                j = seam[i]
                for c in range(3):
                    if j == 0:
                        # Left border
                        new_image[i, 0, c] = image[i, 0, c]
                        new_image[i, 1:, c] = image[i, :, c]
                    elif j >= cols - 1:
                        # Right border or overflow
                        new_image[i, :cols, c] = image[i, :, c]
                        new_image[i, cols, c] = image[i, cols - 1, c]
                    else:
                        # Normal case
                        new_image[i, :j+1, c] = image[i, :j+1, c]
                        new_image[i, j+1, c] = image[i, j, c]
                        new_image[i, j+2:, c] = image[i, j+1:, c]

        elif orientation == 'horizontal':  # horizontal
            new_image = np.zeros((rows + 1, cols, 3), dtype=image.dtype)
            for j in range(cols):
                i = seam[j]
                for c in range(3):
                    if i == 0:
                        # Top border
                        new_image[0, j, c] = image[0, j, c]
                        new_image[1:, j, c] = image[:, j, c]
                    elif i >= rows - 1:
                        # Bottom border or overflow
                        new_image[:rows, j, c] = image[:, j, c]
                        new_image[rows, j, c] = image[rows - 1, j, c]
                    else:
                        # Normal case
                        new_image[:i+1, j, c] = image[:i+1, j, c]
                        new_image[i+1, j, c] = image[i, j, c]  # Duplicate the seam pixel
                        new_image[i+2:, j, c] = image[i+1:, j, c] # Copy the pixels below the seam

        else:
            print("Error: orientation must be 0 (vertical) or 1 (horizontal)")
        return new_image

    def upsize(self,image: np.ndarray, num_seams: int, method: str = 'l1', orientation: str = 'vertical') -> np.ndarray:
        """
        Upsize the image by adding num_seams seams.
        """
        history = self.collect_seams(num_seams, method, orientation)
        new_image = image.copy()
        if orientation == 'vertical':
            décalage_indice =  np.zeros((image.shape[0],), dtype=int)
        else : 
            décalage_indice =  np.zeros((image.shape[1],), dtype=int)


        for _ in range(num_seams):
            seam = history.pop() + décalage_indice
            new_image = self.add_seam(new_image, seam, orientation)

            for i in range(len(décalage_indice)):
                if orientation == 'vertical':
                    if i < len(seam) and seam[i] < new_image.shape[1] - 1: # If the seam is not at the last column
                        décalage_indice[i] += 1
                else:
                    if i < len(seam) and seam[i] < new_image.shape[0] - 1: # If the seam is not at the last row
                        décalage_indice[i] += 1


        return new_image
    

    def collect_seams(self, num_seams:int, method:str='l1', orientation:str='vertical'):
        """
        Collects num_seams seams from the image in the specified orientation.
        This method computes the energy map, finds the seams, and removes them from the image.
        Returns a list of seams in the order they were removed.
        """
        img_copy = self.image.copy()
        history = []
        for _ in range(num_seams):
            energy = self.compute_energy(method, img_copy)
            seam = self.find_seam(img_copy, energy, orientation)
            history.append(seam)
            img_copy = self.remove_seam(img_copy, seam, orientation)
        #  We stock the seams in history in the extract order
        return history

    def remove_object(self,image: np.ndarray, mask: np.ndarray, method: str = 'l1', orientation: str = 'vertical') -> np.ndarray:
        """
        Removes seams from self.image based on the mask.
        The idea is to assign a very low energy to all pixels in the masked area, and keep removing seams as long as there are pixels left in the mask.
        """

        new_image1 = image.copy()
        while np.any(mask == 255): # While there are still pixels in the mask (255 means the pixel is part of the object to remove)
            energy = self.compute_energy(method, new_image1)
            energy[mask == 255] = -1e6  # Set the energy of the object's pixels to a very low value to ensure the algorithm removes seams through these points
            seam = self.find_seam(new_image1, energy, orientation)
            new_image1 = self.remove_seam(new_image1, seam, orientation)
            mask = self.remove_seam(mask,seam,orientation)

        # The object has been removed, now scale up to return to the original image dimensions

        rows,cols,_ = image.shape
        new_rows, new_cols,_ = new_image1.shape

        num_seams = (rows - new_rows) + (cols - new_cols)

        new_image2 = self.upsize(new_image1, num_seams, method, orientation) 

        

        return new_image1,new_image2