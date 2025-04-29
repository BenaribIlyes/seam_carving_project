# Import necessary libraries
import cv2 # type: ignore
import numpy as np
import matplotlib.pyplot as plt


class SeamCarver:
    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.float32)
        self.history = []  # To store removed seams

    def compute_energy(self, method='l1') -> np.ndarray:
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
            # Exemple de calcul d'une énergie globale d'entropie de l'histogramme
            hist = cv2.calcHist([gray], [0], None, [256], [0,256]).ravel()
            p = hist / hist.sum()
            p = p[p>0]
            H = -(p * np.log2(p)).sum()
            # on remplit la carte d'énergie avec cette valeur constante
            energy = np.full_like(gray, H, dtype=np.float32)

        elif method == 'HoG': # Compute the HoG energy map eHoG(x,y) = (|Ix|+|Iy|) / max(HoG_local(x,y))
            #  Compute the grad X/Y, magnitude & degree orientation [0,180)
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            mag = np.hypot(gx, gy)
            ang = (np.degrees(np.arctan2(gy, gx)) + 360) % 180

            # To prepare the bin (angle intervals) and the win_size (size of the local window to see the orientation of the histigram)
            # Practicle rule: for an image of less than 500 pixels on the smallest side use 8 bins
            # Practicle rule: win_size = alpha  * min(rows,cols) with apha between 0.01 and 0.05
            nbins = 12 # Default Value about 22.5° 
            win_size = 15
            bin_edges = np.linspace(0, 180, nbins+1)
            half = win_size//2
            rows,cols = gray.shape
            eHoG = np.zeros((rows,cols), dtype=np.float64)
    
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
            energy = eHoG
    
        else:
            raise ValueError(f"compute_energy: méthode inconnue '{method}'")


        return energy

    def find_seam(self,image, orientation='vertical') -> np.ndarray:
        """
        Find the minimal-energy seam in the given orientation.
        Returns an array of indices (row -> col) for vertical, or (col -> row) for horizontal.
        """
        energy = self.compute_energy() # Compute the energy of the image, here we do it for all but doing it localy will be better (for only around the seams left)
        if orientation == 'horizontal':
            energy = energy.T
        rows, cols = energy.shape
        seam = np.zeros((rows,), dtype=int) # Initialize the seam array
        dp = np.zeros((rows, cols), dtype=np.float32) # Initialize the dynamic programming table
        dp[0] = energy[0] # The first row is just the energy of the first col
        # backtrack = np.zeros_like(dp, dtype=int)

        if orientation == 'vertical':
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

    def remove_seam(self,image, seam: np.ndarray):
        """
        Remove the seam from the image.
        """
        rows, cols, _ = image.shape # Get the dimensions of the image
        new_image = np.zeros((rows, cols - 1, 3), dtype=image.dtype) # Create a new image with one less column

        for i in range(rows):
            j = seam[i]
            new_image[i, :j] = image[i, :j] # Copy the pixels to the left of the seam
            new_image[i, j:] = image[i, j + 1:] # Copy the pixels to the right of the seam

        return new_image       

    def seam_carve(self, num_seams: int, method='l1'):
        """
        Découpe num_seams coutures verticales dans self.image.
        Retourne l'image retarotée.
        """
        for _ in range(num_seams):
            # 1) Calculer l'énergie de l'image **actuelle**
            energy = self.compute_energy(method)

            # 2) Trouver le seam sur cette énergie
            seam = self.find_seam(self.image, energy, orientation='vertical')

            # 3) Sauvegarder si besoin
            self.history.append(seam)

            # 4) Supprimer ce seam de self.image
            self.image = self.remove_seam(self.image, seam, orientation='vertical')

        return self.image



    #def insert(self, num_seams: int, orientation='vertical', method='sobel'):
     #
     #   """
      #  Insert seams by duplicating previously removed paths in reverse order.
       # """
        #seams = [s for s, ori in self.history if ori == orientation][:num_seams]
        #for seam in reversed(seams):
        #    self._insert_single_seam(seam, orientation)

    #def _insert_single_seam(self, seam: np.ndarray, orientation='vertical'):
        # Implement seam insertion by averaging neighbor pixels...
    #    pass
