import numpy as np

class SafeModel:
    def __init__(self, alpha=1.0, eps=1e-8, eps2=1e-8, maxit=1e5):
        self.alpha = alpha
        self.eps = eps
        self.eps2 = eps2
        self.maxit = int(maxit)
        
        # Outputs after fitting
        self.beta = None
        self.gamma = None
        self.lambda_ = None
        self.npass_beta = None
        self.npass_gamma = None
        self.jerr = None

    def _fista_vectorized(self, x, y, k, ulam, int_b, int_g, ind_p):
        nobs, nvars = x.shape
        nlam = len(ulam)

        beta = np.zeros((nvars, nlam))
        gamma = np.zeros((nvars, nlam))
        npass_beta = np.zeros(nlam, dtype=int)
        npass_gamma = np.zeros(nlam, dtype=int)
        jerr = 0
        rnobs = float(nobs)

        for l in range(nlam):
            betavec = int_b.copy()
            gammavec = int_g.copy()
            oldbeta = int_b.copy()
            oldgamma = int_g.copy()

            xibeta = x @ oldbeta
            oldz = (x.T @ (np.exp(xibeta) - k)) / rnobs

            xigamma = x @ oldgamma
            oldr = (-x.T @ (y * self.alpha * np.exp(-xigamma)) + x.T @ (k * self.alpha)) / rnobs

            told = 1.0
            eta = 0.5

            while True:
                xibeta = x @ betavec
                zvec = (x.T @ (np.exp(xibeta) - k)) / rnobs
                hold = (np.exp(xibeta) - k * xibeta) / rnobs

                olddif = betavec - oldbeta
                diffz = zvec - oldz
                denom = np.dot(olddif, diffz)
                ar = np.dot(olddif, olddif) / denom if denom != 0 else 1.0
                aval = 100.0 / np.linalg.norm(zvec)
                ar = max(ar, aval)

                n = 1
                while True:
                    lms = (eta ** n) * ar
                    wvec = betavec - lms * zvec
                    indvec = (gammavec * wvec <= 0).astype(int)
                    vvec = np.abs(wvec) - ind_p * lms * ulam[l] * np.abs(gammavec) * indvec
                    betanew = np.sign(wvec) * np.where(vvec > 0, vvec, 0.0)

                    xibeta_new = x @ betanew
                    hnew = (np.exp(xibeta_new) - k * xibeta_new) / rnobs

                    difbeta = betanew - betavec
                    bound = hold.sum() + np.dot(difbeta, zvec) + 0.5 * (1.0 / lms) * np.sum(difbeta ** 2)
                    bound = hnew.sum() - bound

                    if bound > 0:
                        n += 1
                    else:
                        break

                tnew = 0.5 + 0.5 * np.sqrt(1.0 + 4.0 * told ** 2)
                mul = 1.0 + (told - 1.0) / tnew
                told = tnew
                oldbeta = betavec.copy()
                oldz = zvec.copy()
                betavec += mul * (betanew - oldbeta)

                difbeta = betavec - oldbeta
                npass_beta[l] += 1
                if np.max(difbeta ** 2) < self.eps * mul * mul:
                    break
                if np.sum(npass_beta) > self.maxit:
                    break

            beta[:, l] = betavec
            if np.sum(npass_beta) > self.maxit:
                jerr = l
                break

            # === Gamma loop ===
            nold = 1.0
            while True:
                xigamma = x @ gammavec
                rvec = (-x.T @ (y * self.alpha * np.exp(-xigamma)) + x.T @ (k * self.alpha)) / rnobs
                gold = (y * self.alpha * np.exp(-xigamma) + k * self.alpha * xigamma) / rnobs

                oldgamdif = gammavec - oldgamma
                difr = rvec - oldr
                denom = np.dot(oldgamdif, difr)
                cr = np.dot(oldgamdif, oldgamdif) / denom if denom != 0 else 1.0
                cval = 1.0 / np.linalg.norm(rvec)
                cr = max(cr, cval)

                n = 1
                while True:
                    dms = (eta ** n) * cr
                    wvec = gammavec - dms * rvec
                    indvec = (betavec * wvec <= 0).astype(int)
                    vvec = np.abs(wvec) - ind_p * dms * ulam[l] * np.abs(betavec) * indvec
                    gammanew = np.sign(wvec) * np.where(vvec > 0, vvec, 0.0)

                    xigamma_new = x @ gammanew
                    gnew = (y * self.alpha * np.exp(-xigamma_new) + k * self.alpha * xigamma_new) / rnobs

                    difgam = gammanew - gammavec
                    bound = gold.sum() + np.dot(difgam, rvec) + 0.5 * (1.0 / dms) * np.sum(difgam ** 2)
                    bound = gnew.sum() - bound

                    if bound > 0:
                        n += 1
                    else:
                        break

                nnew = 0.5 + 0.5 * np.sqrt(1.0 + 4.0 * nold ** 2)
                mul = 1.0 + (nold - 1.0) / nnew
                nold = nnew
                oldgamma = gammavec.copy()
                oldr = rvec.copy()
                gammavec += mul * (gammanew - oldgamma)

                xigamma = x @ gammavec
                difgam = gammavec - oldgamma
                npass_gamma[l] += 1
                if np.max(difgam ** 2) < self.eps2 * mul * mul:
                    break
                if np.sum(npass_gamma) > self.maxit:
                    break

            gamma[:, l] = gammavec
            if np.sum(npass_gamma) > self.maxit:
                jerr = l
                break

        return beta, gamma, npass_beta, npass_gamma, jerr

    def _initialize(self, x, y, k):
        nobs, nvars = x.shape
        ind_p = np.zeros(nvars)
        int_beta = np.zeros(nvars)
        int_gamma = np.zeros(nvars)
        lambda_ = np.array([0.0])

        beta, gamma, _, _, _ = self._fista_vectorized(
            x, y, k,
            ulam=lambda_,
            int_b=int_beta,
            int_g=int_gamma,
            ind_p=ind_p
        )

        return beta[:, 0], gamma[:, 0]

    def fit(self, x, y, k, lambda_, ind_p=None, int_beta=None, int_gamma=None):
        x = np.asarray(x)
        y = np.asarray(y).flatten()
        k = np.asarray(k)
        lambda_ = np.asarray(lambda_)

        nobs, nvars = x.shape
        if y.shape[0] != nobs:
            raise ValueError("x and y must have the same number of observations")

        if ind_p is None:
            ind_p = np.zeros(nvars)

        if int_beta is None or int_gamma is None:
            int_beta, int_gamma = self._initialize(x, y, k)

        ulam = np.sort(lambda_)[::-1]

        beta, gamma, npass_beta, npass_gamma, jerr = self._fista_vectorized(
            x, y, k, ulam, int_beta, int_gamma, ind_p
        )

        self.beta = beta
        self.gamma = gamma
        self.lambda_ = ulam
        self.npass_beta = npass_beta
        self.npass_gamma = npass_gamma
        self.jerr = jerr
        return self
    
    def coef(self, s=None):
        beta = self.beta
        gamma = self.gamma
        if s is not None:
            lamlist = self._lambda_interp(self.lambda_, s)
            beta = beta[:, lamlist['left']] * lamlist['frac'] + \
                   beta[:, lamlist['right']] * (1 - lamlist['frac'])
            gamma = gamma[:, lamlist['left']] * lamlist['frac'] + \
                    gamma[:, lamlist['right']] * (1 - lamlist['frac'])
        return {'beta': beta, 'gamma': gamma}
        
    def predict(self, x, s=None):
        x = np.asarray(x)
        coefs = self.coef(s)
        khat = np.exp(x @ coefs['beta'])
        muhat = np.exp(x @ coefs['gamma'])
        return khat * muhat

    def _lambda_interp(self, lambda_seq, s):
        lambda_seq = np.asarray(lambda_seq)
        s = np.atleast_1d(s)

        if len(lambda_seq) == 1:
            left = np.zeros_like(s, dtype=int)
            right = np.zeros_like(s, dtype=int)
            frac = np.ones_like(s, dtype=float)
        else:
            s = np.clip(s, lambda_seq.min(), lambda_seq.max())
            k = len(lambda_seq)
            lam_max, lam_min = lambda_seq[0], lambda_seq[-1]
            scale = lam_max - lam_min
            sfrac = (lam_max - s) / scale
            lambda_scaled = (lam_max - lambda_seq) / scale
            coord = np.interp(sfrac, lambda_scaled, np.arange(k))
            left = np.floor(coord).astype(int)
            right = np.ceil(coord).astype(int)
            left = np.clip(left, 0, k - 1)
            right = np.clip(right, 0, k - 1)
            denom = lambda_seq[left] - lambda_seq[right]
            with np.errstate(divide='ignore', invalid='ignore'):
                frac = (s - lambda_seq[right]) / denom
                frac[left == right] = 1.0

        return {'left': left, 'right': right, 'frac': frac}
        
    def eccv(self, x, y, k, lambda_, ind_p, rep=24, nfolds=5):
        x = np.asarray(x)
        y = np.asarray(y).flatten()
        nobs = x.shape[0]
        if len(y) != nobs:
            raise ValueError("x and y must have the same number of observations")
        if nfolds < 3:
            raise ValueError("nfolds must be >= 3")

        foldid_mat = np.empty((nobs, rep), dtype=int)
        base = np.tile(np.arange(nfolds), int(np.ceil(nobs / nfolds)))[:nobs]
        for r in range(rep):
            foldid_mat[:, r] = np.random.permutation(base)

        int_beta, int_gamma = self._initialize(x, y, k)

        votes = np.zeros(len(lambda_), dtype=int)
        ncvmat = np.full((rep, len(lambda_)), np.nan)

        for r in range(rep):
            foldid = foldid_mat[:, r]
            outlist = []
            for j in range(nfolds):
                train_idx = foldid != j
                model = SafeModel(alpha=self.alpha, eps=self.eps, eps2=self.eps2, maxit=self.maxit)
                model.fit(x[train_idx], y[train_idx], k[train_idx], lambda_,
                          ind_p=ind_p, int_beta=int_beta, int_gamma=int_gamma)
                outlist.append(model)

            cvres = self._cvpath(outlist, x, y, lambda_, foldid)
            cvm = cvres['cvm']
            best = np.argmin(cvm)
            votes[best] += 1
            ncvmat[r, :] = cvm

        vote_l = np.argmax(votes)
        return {
            'lambda.min': lambda_[vote_l],
            'ncvmat': ncvmat,
            'lambda': lambda_
        }

    def _cvpath(self, outlist, x, y, lambda_, foldid):
        nfolds = int(np.max(foldid)) + 1
        nlam = len(lambda_)
        nobs = x.shape[0]
        predmat = np.full((nobs, nlam), np.nan)

        for i in range(nfolds):
            which = foldid == i
            preds = outlist[i].predict(x[which])
            predmat[which, :] = preds

        cvraw = (y[:, None] - predmat) ** 2
        n_per_lam = np.sum(~np.isnan(predmat), axis=0)
        cvm = np.apply_along_axis(lambda col: np.mean(col[~np.isnan(col)]), axis=0, arr=cvraw)

        scaled = (cvraw - cvm[None, :]) ** 2
        cvsd = np.sqrt(np.nanmean(scaled, axis=0) / (n_per_lam - 1))

        return {'cvm': cvm, 'cvsd': cvsd}