import warnings

import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
import numpy as np  # pyright: ignore[reportMissingImports]
import pandas as pd  # pyright: ignore[reportMissingImports]
from scipy.stats import rankdata  # pyright: ignore[reportMissingImports]


class MonACP:
    """Analyse en Composantes Principales (ACP)."""

    _METHODES = ("normed", "non_normed", "spearman")

    def __init__(self, method="normed"):
        """Initialisation de l'ACP.

        :param method: 'normed' (Pearson - par défaut), 'non_normed'
        (Covariance), ou 'spearman' (Rangs)
        """
        if method not in self._METHODES:
            raise ValueError(
                f"Méthode inconnue : {method!r}. "
                "Choisir 'normed', 'non_normed' ou 'spearman'."
            )
        self.method = method

        # Métadonnées du DataFrame pour utiliser plus tard lors des projections
        self.feature_names_ = None  # Noms des colonnes numériques retenues
        self.n_samples_ = None  # Nombre d'individus vus au fit()
        self.ignored_columns_ = []  # Colonnes non numériques ignorées

        # Paramètres statistiques appris uniquement pendant le fit()
        self.mean_ = None
        self.std_ = None
        self.X_train_ = None  # Données d'entraînement (utile pour Spearman)

        # Résultats mathématiques de l'ACP
        self.matrix_ = None  # DataFrame (p, p) : Matrice de corr/cov M
        self.eigenvalues_ = None  # (p,) : Valeurs propres triées
        self.eigenvectors_ = None  # (p, p) : Vecteurs propres triés
        self.explained_variance_ratio_ = None  # (p,) : % d'info par axe
        self.cumulative_variance_ratio_ = None  # (p,) : % cumulé

        self.fitted_method_ = None  # Méthode utilisée lors du dernier fit()
        self.is_fitted_ = False

    def _check_is_fitted(self):
        """Vérifie que fit() a bien été appelé avant d'utiliser le modèle."""
        if not self.is_fitted_:
            raise RuntimeError(
                "Erreur : L'ACP n'est pas encore entraînée ! Appelez .fit(X) d'abord."
            )
        if self.method != self.fitted_method_:
            raise RuntimeError(
                "Erreur : La méthode a changé depuis le fit(). "
                "Relancez .fit(X) avant d'utiliser le modèle."
            )

    def _validate_n_axes(self, n_axes):
        """Vérifie que le nombre d'axes demandé est valide."""
        n_variables = len(self.feature_names_ or [])
        if not isinstance(n_axes, (int, np.integer)) or not (
            1 <= n_axes <= n_variables
        ):
            raise ValueError(
                f"n_axes doit être un entier compris entre 1 et {n_variables} "
                f"(reçu : {n_axes!r})."
            )

    def _validate_input(self, X, fitting=False):
        """Gère les DataFrames, listes ou tableaux numpy, filtre le texte et vérifie les NaN.

        Renvoie : (X_numpy, index_lignes)
        """
        # 1. Si ce n'est pas un DataFrame (ex: liste ou numpy array), on le convertit
        est_dataframe = isinstance(X, pd.DataFrame)
        if not est_dataframe:
            X = pd.DataFrame(np.asarray(X))

        if fitting:
            # Noms par défaut uniquement pour les données qui n'étaient pas un DataFrame
            if not est_dataframe:
                X.columns = [f"Var_{i + 1}" for i in range(X.shape[1])]
                X.index = [f"Indiv_{i + 1}" for i in range(X.shape[0])]

            if X.columns.duplicated().any():
                doublons = X.columns[X.columns.duplicated()].tolist()
                raise ValueError(f"Erreur : Noms de colonnes dupliqués : {doublons}.")

            # 2. Gestion des colonnes numériques vs texte
            X_num = X.select_dtypes(include=[np.number])
            ignored_cols = [col for col in X.columns if col not in X_num.columns]
            self.ignored_columns_ = ignored_cols
            if ignored_cols:
                warnings.warn(
                    "Colonnes non-numériques ignorées automatiquement : "
                    f"{ignored_cols}",
                    stacklevel=3,
                )

            if X_num.shape[1] == 0:
                raise ValueError(
                    "Erreur : Aucune colonne numérique trouvée dans les données !"
                )

            # On sauvegarde les noms des colonnes retenues et leur nombre
            self.feature_names_ = list(X_num.columns)
            self.n_samples_ = X_num.shape[0]

        else:
            self._check_is_fitted()

            if X.shape[1] == 0:
                raise ValueError("Erreur : X ne contient aucune colonne.")

            # Cas d'un numpy array / d'une liste : les colonnes sont un RangeIndex.
            # On les renomme avec les noms appris lors du fit().
            if not est_dataframe:
                if X.shape[1] != len(self.feature_names_):
                    raise ValueError(
                        f"Erreur : X doit contenir {len(self.feature_names_)} "
                        f"colonnes, mais il en contient {X.shape[1]}."
                    )
                X.columns = self.feature_names_
            elif isinstance(X.columns, pd.RangeIndex) and len(X.columns) == len(
                self.feature_names_
            ):
                X.columns = self.feature_names_

            if X.columns.duplicated().any():
                raise ValueError("Erreur : X contient des noms de colonnes dupliqués.")

            colonnes_manquantes = [
                col for col in self.feature_names_ if col not in X.columns
            ]
            if colonnes_manquantes:
                raise ValueError(
                    "Erreur : Colonnes manquantes dans X : "
                    f"{colonnes_manquantes}. "
                    f"Colonnes attendues : {self.feature_names_}."
                )

            # En transform(), on reprend exactement les colonnes retenues au fit()
            X_num = X.loc[:, self.feature_names_]

        # 3. Vérification des valeurs manquantes (NaN)
        if X_num.isna().to_numpy().any():
            raise ValueError(
                "Erreur : Les données contiennent des valeurs manquantes (NaN). "
                "Nettoyez-les avant l'ACP !"
            )

        return X_num.to_numpy(dtype=float), X_num.index

    def _prepare_data(self, X_array, fitting=False):
        """Étape 0 : Prépare la matrice Z.

        Si fitting=True : calcule et sauvegarde mean_ et std_. Si fitting=False
        : réutilise mean_ et std_ sauvegardés lors du fit() !
        """
        if not fitting and self.method != self.fitted_method_:
            raise RuntimeError(
                "Erreur : La méthode a changé depuis le fit(). "
                "Relancez .fit(X) avant .transform(X)."
            )

        if self.method == "non_normed":
            if fitting:
                self.mean_ = X_array.mean(axis=0)
            Z = X_array - self.mean_

        elif self.method == "normed":
            if fitting:
                self.mean_ = X_array.mean(axis=0)
                self.std_ = X_array.std(axis=0)
                if np.any(self.std_ == 0):
                    raise ValueError(
                        "Erreur : Une ou plusieurs colonnes sont constantes "
                        "(écart-type = 0) !"
                    )
            Z = (X_array - self.mean_) / self.std_

        elif self.method == "spearman":
            if fitting:
                self.X_train_ = X_array.copy()
                R = rankdata(X_array, method="average", axis=0)
                self.mean_ = R.mean(axis=0)
                self.std_ = R.std(axis=0)
                if np.any(self.std_ == 0):
                    raise ValueError(
                        "Erreur : Une ou plusieurs colonnes ont des rangs "
                        "constants (écart-type = 0) !"
                    )
            else:
                if self.X_train_ is None:
                    raise RuntimeError(
                        "Erreur : Les données d'entraînement Spearman sont "
                        "absentes. Relancez .fit(X)."
                    )
                # Rang moyen de chaque nouvelle valeur par rapport aux données
                # du fit() : nombre de valeurs inférieures + (nombre d'égales + 1) / 2.
                n_nouveaux, p = X_array.shape
                R = np.empty((n_nouveaux, p), dtype=float)
                for j in range(p):
                    train_j = self.X_train_[:, j]
                    nouveau_j = X_array[:, j]
                    inferieurs = np.sum(train_j[None, :] < nouveau_j[:, None], axis=1)
                    egaux = np.sum(train_j[None, :] == nouveau_j[:, None], axis=1)
                    R[:, j] = inferieurs + (egaux + 1) / 2.0
            Z = (R - self.mean_) / self.std_

        else:
            raise ValueError(
                "Méthode inconnue ! Choisir 'normed', 'non_normed' ou 'spearman'."
            )

        return Z

    def fit(self, X):
        """Entraîne l'ACP sur X (calcule tous les axes)."""
        # On repart d'un modèle propre : si le fit échoue, is_fitted_ reste False.
        self.is_fitted_ = False
        self.fitted_method_ = None
        self.feature_names_ = None
        self.n_samples_ = None
        self.ignored_columns_ = []
        self.mean_ = None
        self.std_ = None
        self.X_train_ = None
        self.matrix_ = None
        self.eigenvalues_ = None
        self.eigenvectors_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None

        # Validation et extraction (fitting=True car on apprend sur ces données)
        X_array, _ = self._validate_input(X, fitting=True)
        n = X_array.shape[0]
        if n < 2:
            raise ValueError("Erreur : L'ACP nécessite au moins 2 individus (lignes).")

        # Étape 0 : Préparation de Z avec fitting=True
        Z = self._prepare_data(X_array, fitting=True)

        # Étape 1 : Calcul de la matrice M (taille p, p)
        M_array = (1 / n) * (Z.T @ Z)
        # On la stocke sous forme d'un DataFrame avec les noms des variables !
        self.matrix_ = pd.DataFrame(
            M_array, index=self.feature_names_, columns=self.feature_names_
        )

        # Étape 2 : Diagonalisation et tri
        # puis les trier par ordre décroissant (np.argsort(eigenvalues)[::-1])
        eigenvalues, eigenvectors = np.linalg.eigh(M_array)
        idx = np.argsort(eigenvalues)[::-1]
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvectors[:, idx]

        # Étape 3 : Pourcentages d'inertie
        inertie_totale = self.eigenvalues_.sum()
        if inertie_totale <= 0:
            raise ValueError(
                "Erreur : Inertie totale nulle (données constantes) ; "
                "l'ACP est impossible."
            )
        # On garde les valeurs exactes et on arrondit seulement à l'affichage.
        self.explained_variance_ratio_ = self.eigenvalues_ / inertie_totale
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)

        self.fitted_method_ = self.method
        self.is_fitted_ = True
        return self

    def bilan(self):
        """Affiche le tableau d'aide à la décision pour choisir le nombre d'axes."""
        self._check_is_fitted()
        seuil_kaiser = np.mean(self.eigenvalues_)
        critere_kaiser = self.method in ("normed", "spearman")
        print(
            f"\n================ BILAN DE L'ACP ({self.method.upper()}) ================"
        )
        print(f"Variables actives ({len(self.feature_names_)}) : {self.feature_names_}")
        if critere_kaiser:
            print(
                f"Critère de Kaiser (Val. Propre > {seuil_kaiser:.2f}) -> "
                f"Conserver {np.sum(self.eigenvalues_ > seuil_kaiser)} axe(s)."
            )
        else:
            print(
                f"Moyenne des valeurs propres ({seuil_kaiser:.2f}) -> "
                f"Conserver {np.sum(self.eigenvalues_ > seuil_kaiser)} axe(s)."
            )

        print("-" * 60)
        titre_critere = "Kaiser ?" if critere_kaiser else "Axe ?"
        print(
            f"{'Axe':<8} | {'Val. Propre':<12} | {'% Info':<10} | "
            f"{'% Cumulé':<10} | {titre_critere}"
        )
        print("-" * 60)
        for i in range(len(self.eigenvalues_)):
            val = self.eigenvalues_[i]
            pct = self.explained_variance_ratio_[i] * 100
            cum = self.cumulative_variance_ratio_[i] * 100
            kaiser_ok = "OUI" if val > seuil_kaiser else "NON"
            print(
                f"Axe {i + 1:<4} | {val:<12.4f} | {pct:<9.2f}% | "
                f"{cum:<9.2f}% | {kaiser_ok}"
            )
        print("-" * 60)

    def get_coord_variables(self, n_axes=2):
        """Renvoie un DataFrame avec les coordonnées des variables."""
        self._check_is_fitted()
        self._validate_n_axes(n_axes)
        u_k = self.eigenvectors_[:, :n_axes]
        # On borne à 0 pour éviter un sqrt() d'une valeur propre légèrement négative
        # due aux erreurs d'arrondi numérique.
        lambda_k = np.clip(self.eigenvalues_[:n_axes], 0, None)

        coord_var = u_k * np.sqrt(lambda_k)
        axes_names = [f"Axe_{i + 1}" for i in range(n_axes)]
        return pd.DataFrame(coord_var, index=self.feature_names_, columns=axes_names)

    def transform(self, X, n_axes=2):
        """Projette les individus sur les 'n_axes' choisis et renvoie un DataFrame."""
        self._check_is_fitted()
        self._validate_n_axes(n_axes)

        # Validation et préparation avec fitting=False
        # (on réutilise mean_ et std_ du fit !)
        X_array, index_individus = self._validate_input(X, fitting=False)
        Z = self._prepare_data(X_array, fitting=False)

        u_k = self.eigenvectors_[:, :n_axes]
        coord_indiv = Z @ u_k

        axes_names = [f"Axe_{i + 1}" for i in range(n_axes)]
        return pd.DataFrame(coord_indiv, index=index_individus, columns=axes_names)

    def tracer_acp(self, df_individus, df_variables):
        """Trace la carte des individus et le cercle des corrélations."""
        self._check_is_fitted()

        for nom, df in (
            ("df_individus", df_individus),
            ("df_variables", df_variables),
        ):
            if not isinstance(df, pd.DataFrame):
                raise TypeError(f"Erreur : {nom} doit être un DataFrame.")
            colonnes_manquantes = [
                col for col in ("Axe_1", "Axe_2") if col not in df.columns
            ]
            if colonnes_manquantes:
                raise ValueError(
                    f"Erreur : {nom} doit contenir les colonnes Axe_1 et Axe_2. "
                    f"Colonnes manquantes : {colonnes_manquantes}."
                )

        pct_axe1 = self.explained_variance_ratio_[0] * 100
        pct_axe2 = self.explained_variance_ratio_[1] * 100

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Graphique 1 : Carte des Individus
        ax1.scatter(
            df_individus["Axe_1"],
            df_individus["Axe_2"],
            color="royalblue",
            s=80,
            zorder=3,
        )
        for nom_client, row in df_individus.iterrows():
            ax1.annotate(
                nom_client,
                (row["Axe_1"], row["Axe_2"]),
                fontsize=8,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax1.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax1.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax1.set_title("Carte des Individus (Projection)")
        ax1.set_xlabel(f"Axe 1 ({pct_axe1:.1f}%)")
        ax1.set_ylabel(f"Axe 2 ({pct_axe2:.1f}%)")
        ax1.grid(True, alpha=0.3)

        # Graphique 2 : Cercle des corrélations (pour les variables)
        # Si normée ou spearman, on trace le fameux cercle de rayon 1 !
        if self.method in ("normed", "spearman"):
            cercle = plt.Circle(
                (0, 0),
                1,
                color="crimson",
                fill=False,
                linestyle="--",
                linewidth=1.5,
            )
            ax2.add_patch(cercle)
            ax2.set_xlim(-1.1, 1.1)
            ax2.set_ylim(-1.1, 1.1)
            ax2.set_aspect("equal")

        for nom_var, row in df_variables.iterrows():
            ax2.arrow(
                0,
                0,
                row["Axe_1"],
                row["Axe_2"],
                head_width=0.04,
                head_length=0.04,
                fc="crimson",
                ec="crimson",
                length_includes_head=True,
            )
            ax2.annotate(
                nom_var,
                (row["Axe_1"], row["Axe_2"]),
                color="crimson",
                weight="bold",
                fontsize=9,
                xytext=(5, 5),
                textcoords="offset points",
            )

        ax2.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax2.axvline(0, color="grey", linestyle="--", linewidth=0.8)
        ax2.set_title(f"Graphique des Variables ({self.method})")
        ax2.set_xlabel(f"Axe 1 ({pct_axe1:.1f}%)")
        ax2.set_ylabel(f"Axe 2 ({pct_axe2:.1f}%)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        return fig
