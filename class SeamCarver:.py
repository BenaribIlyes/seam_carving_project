# Import necessary libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt

class SeamCarver:
    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.float32)
        self.history = []  # to store removed seams

    def compute_energy(self, method='sobel') -> np.ndarray:
        """
        Compute the energy map of the current image.
        Supported methods: 'sobel', 'l1', 'l2', 'entropy', 'hog', ...
        """
        gray = cv2.cvtColor(self.image.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        if method == 'sobel':
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            energy = np.abs(gx) + np.abs(gy)
        elif method == 'l2':
            gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
            gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
            energy = np.hypot(gx, gy)
        # ... add other methods
        return energy

    def find_seam(self, energy: np.ndarray, orientation='vertical') -> np.ndarray:
        """
        Find the minimal-energy seam in the given orientation.
        Returns an array of indices (row -> col) for vertical, or (col -> row) for horizontal.
        """
        if orientation == 'horizontal':
            energy = energy.T
        rows, cols = energy.shape
        dp = energy.copy()
        backtrack = np.zeros_like(dp, dtype=int)
        # Dynamic programming to fill dp and backtrack...
        # ...
        seam = np.zeros(rows, dtype=int)
        # Backtracking seam path...
        # ...
        if orientation == 'horizontal':
            seam = seam  # interpreted as (col -> row)
        return seam

    def remove_seam(self, seam: np.ndarray, orientation='vertical'):
        """
        Remove the specified seam from the image.
        """
        H, W, _ = self.image.shape
        if orientation == 'horizontal':
            self.image = self.image.transpose(1, 0, 2)
        mask = np.ones((H, W), dtype=bool)
        for i in range(H):
            mask[i, seam[i]] = False
        self.image = self.image[mask].reshape((H, W-1, 3))
        if orientation == 'horizontal':
            self.image = self.image.transpose(1, 0, 2)

    def carve(self, num_seams: int, orientation='vertical', method='sobel'):
        """
        Carve out the specified number of seams.
        """
        for _ in range(num_seams):
            energy = self.compute_energy(method=method)
            seam = self.find_seam(energy, orientation=orientation)
            self.history.append((seam, orientation))
            self.remove_seam(seam, orientation)

    def insert(self, num_seams: int, orientation='vertical', method='sobel'):
        """
        Insert seams by duplicating previously removed paths in reverse order.
        """
        seams = [s for s, ori in self.history if ori == orientation][:num_seams]
        for seam in reversed(seams):
            self._insert_single_seam(seam, orientation)

    def _insert_single_seam(self, seam: np.ndarray, orientation='vertical'):
        # Implement seam insertion by averaging neighbor pixels...
        pass