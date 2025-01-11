import numpy as np

class NumpyMultiMacenkoNormalizer:
    def __init__(self, norm_mode='avg-post'):
        self.norm_mode = norm_mode
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])
        
    def __convert_rgb2od(self, I, Io, beta):
        if I.ndim == 2:
            I = I[:, :, np.newaxis]
        elif I.ndim != 3:
            raise ValueError("Input image must have 2 or 3 dimensions")
        
        I = np.transpose(I, (1, 2, 0))
        OD = -np.log((I.reshape((-1, I.shape[-1])).astype(float) + 1) / Io)
        ODhat = OD[~np.any(OD < beta, axis=1)]
        return OD, ODhat

    def __find_phi_bounds(self, ODhat, eigvecs, alpha):
        if ODhat.size == 0:
            raise ValueError("ODhat is empty, cannot compute phi bounds")
        
        That = np.dot(ODhat, eigvecs)
        phi = np.arctan2(That[:, 1], That[:, 0])

        minPhi = np.percentile(phi, alpha)
        maxPhi = np.percentile(phi, 100 - alpha)

        return minPhi, maxPhi

    def __find_HE_from_bounds(self, eigvecs, minPhi, maxPhi):
        vMin = np.dot(eigvecs, np.stack((np.cos(minPhi), np.sin(minPhi)))).reshape(-1, 1)
        vMax = np.dot(eigvecs, np.stack((np.cos(maxPhi), np.sin(maxPhi)))).reshape(-1, 1)

        HE = np.where(vMin[0] > vMax[0], np.concatenate((vMin, vMax), axis=1), np.concatenate((vMax, vMin), axis=1))

        return HE

    def __find_HE(self, ODhat, eigvecs, alpha):
        minPhi, maxPhi = self.__find_phi_bounds(ODhat, eigvecs, alpha)
        return self.__find_HE_from_bounds(eigvecs, minPhi, maxPhi)

    def __find_concentration(self, OD, HE):
        Y = OD.T
        return np.linalg.lstsq(HE, Y, rcond=None)[0]

    def __compute_matrices_single(self, I, Io, alpha, beta):
        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)
        
        if ODhat.size == 0:
            raise ValueError("ODhat is empty, cannot compute matrices")

        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        eigvecs = eigvecs[:, [1, 2]]

        HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)
        maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, Is, Io=240, alpha=1, beta=0.15):
        if self.norm_mode == 'avg-post':
            HEs, _, maxCs = zip(*(
                self.__compute_matrices_single(I, Io, alpha, beta)
                for I in Is
            ))

            self.HERef = np.mean(np.stack(HEs), axis=0)
            self.maxCRef = np.mean(np.stack(maxCs), axis=0)
        elif self.norm_mode == 'concat':
            ODs, ODhats = zip(*(
                self.__convert_rgb2od(I, Io, beta)
                for I in Is
            ))
            OD = np.concatenate(ODs, axis=0)
            ODhat = np.concatenate(ODhats, axis=0)

            eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
            eigvecs = eigvecs[:, [1, 2]]

            HE = self.__find_HE(ODhat, eigvecs, alpha)

            C = self.__find_concentration(OD, HE)
            maxCs = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
            self.HERef = HE
            self.maxCRef = maxCs
        elif self.norm_mode == 'avg-pre':
            ODs, ODhats = zip(*(
                self.__convert_rgb2od(I, Io, beta)
                for I in Is
            ))
            
            eigvecs = np.mean(np.stack([np.linalg.eigh(np.cov(ODhat.T))[1][:, [1, 2]] for ODhat in ODhats]), axis=0)

            OD = np.concatenate(ODs, axis=0)
            ODhat = np.concatenate(ODhats, axis=0)
            
            HE = self.__find_HE(ODhat, eigvecs, alpha)

            C = self.__find_concentration(OD, HE)
            maxCs = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
            self.HERef = HE
            self.maxCRef = maxCs
        elif self.norm_mode == 'fixed-single' or self.norm_mode == 'stochastic-single':
            # single img
            self.HERef, _, self.maxCRef = self.__compute_matrices_single(Is[0], Io, alpha, beta)
        else:
            raise ValueError("Unknown norm mode")

    def normalize(self, I, Io=240, alpha=1, beta=0.15, stains=True):
        c, h, w = I.shape

        HE, C, maxC = self.__compute_matrices_single(I, Io, alpha, beta)
        C = (self.maxCRef / maxC).reshape(-1, 1) * C

        Inorm = Io * np.exp(-np.dot(self.HERef, C))
        Inorm[Inorm > 255] = 255
        Inorm = np.transpose(Inorm, (1, 0)).reshape(h, w, c).astype(int)

        H, E = None, None

        if stains:
            H = Io * np.exp(np.dot(-self.HERef[:, 0].reshape(-1, 1), C[0, :].reshape(1, -1)))
            H[H > 255] = 255
            H = np.transpose(H, (1, 0)).reshape(h, w, c).astype(int)

            E = Io * np.exp(np.dot(-self.HERef[:, 1].reshape(-1, 1), C[1, :].reshape(1, -1)))
            E[E > 255] = 255
            E = np.transpose(E, (1, 0)).reshape(h, w, c).astype(int)

        return Inorm, H, E
