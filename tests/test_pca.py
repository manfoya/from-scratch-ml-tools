"""Tests unitaires pour la classe MonACP."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from ml_tools.pca import MonACP


def jeu_de_donnees_synthetique(n=40, graine=42):
    """Crée un jeu de données reproductible de n lignes et 6 variables."""
    rng = np.random.default_rng(graine)
    age = rng.integers(22, 61, n).astype(float)
    revenu = 12 + 0.9 * age + rng.normal(0, 8, n)
    anciennete = np.clip(0.1 * age + rng.normal(0, 2, n), 0, None)
    nb_achats = np.clip(
        2 + 0.08 * revenu + 0.7 * anciennete + rng.normal(0, 3, n), 0, None
    )
    panier_moyen = 25 + 0.9 * revenu + 2.0 * nb_achats + rng.normal(0, 15, n)
    satisfaction = np.clip(
        5 + 0.04 * revenu + 0.15 * anciennete + rng.normal(0, 1.2, n), 1, 10
    )
    return pd.DataFrame(
        {
            "age": age,
            "revenu": revenu,
            "anciennete": anciennete,
            "nb_achats": nb_achats,
            "panier_moyen": panier_moyen,
            "satisfaction": satisfaction,
        },
        index=[f"Client_{i:02d}" for i in range(1, n + 1)],
    )


@pytest.fixture(scope="module")
def donnees():
    return jeu_de_donnees_synthetique()


@pytest.fixture()
def acp_normed(donnees):
    return MonACP(method="normed").fit(donnees)


def test_init_rejette_une_methode_inconnue():
    with pytest.raises(ValueError, match="Méthode inconnue"):
        MonACP(method="inconnue")


def test_fit_normed_calcule_les_resultats(donnees):
    acp = MonACP(method="normed").fit(donnees)

    assert isinstance(acp.matrix_, pd.DataFrame)
    assert acp.matrix_.shape == (6, 6)
    assert acp.eigenvalues_.shape == (6,)
    assert acp.eigenvectors_.shape == (6, 6)
    assert np.all(np.diff(acp.eigenvalues_) <= 0)
    assert acp.explained_variance_ratio_.sum() == pytest.approx(1.0)
    assert acp.cumulative_variance_ratio_[-1] == pytest.approx(1.0)
    assert np.all(np.diff(acp.cumulative_variance_ratio_) >= 0)
    assert acp.cumulative_variance_ratio_.shape == (6,)
    assert acp.feature_names_ == list(donnees.columns)
    assert acp.n_samples_ == len(donnees)
    assert acp.fitted_method_ == "normed"
    assert acp.is_fitted_


def test_fit_retourne_self(donnees):
    acp = MonACP()
    assert acp.fit(donnees) is acp


def test_fit_refuse_moins_de_deux_individus(donnees):
    with pytest.raises(ValueError, match="au moins 2 individus"):
        MonACP().fit(donnees.iloc[:1])


def test_fit_refuse_les_colonnes_dupliquees():
    df = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=["x", "x"])

    with pytest.raises(ValueError, match="dupliqués"):
        MonACP().fit(df)


def test_fit_refuse_aucune_colonne_numerique():
    df = pd.DataFrame({"a": ["x", "y"], "b": ["z", "w"]})

    with (
        pytest.warns(UserWarning, match="non-numériques"),
        pytest.raises(ValueError, match="Aucune colonne numérique"),
    ):
        MonACP().fit(df)


def test_fit_ignore_les_colonnes_non_numeriques(donnees):
    df = donnees.copy()
    df["ville"] = ["Paris", "Lyon"] * (len(df) // 2)

    with pytest.warns(UserWarning, match="ville"):
        acp = MonACP().fit(df)

    assert "ville" not in acp.feature_names_
    assert acp.ignored_columns_ == ["ville"]


def test_normed_refuse_une_colonne_constante(donnees):
    df = donnees.copy()
    df["constante"] = 1.0

    with pytest.raises(ValueError, match="constantes"):
        MonACP(method="normed").fit(df)


def test_non_normed_refuse_une_inertie_nulle():
    df = pd.DataFrame({"x": [1.0, 1.0, 1.0], "y": [2.0, 2.0, 2.0]})

    with pytest.raises(ValueError, match="Inertie totale nulle"):
        MonACP(method="non_normed").fit(df)


def test_transform_retrouve_les_coordonnees_du_fit(acp_normed, donnees):
    attendu = acp_normed.transform(donnees)

    X = donnees.to_numpy(dtype=float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0)
    calcul = Z @ acp_normed.eigenvectors_[:, :2]

    np.testing.assert_allclose(attendu.to_numpy(), calcul, rtol=1e-12, atol=1e-12)
    assert list(attendu.columns) == ["Axe_1", "Axe_2"]
    assert list(attendu.index) == list(donnees.index)


def test_transform_accepte_un_tableau_numpy(acp_normed, donnees):
    resultat = acp_normed.transform(donnees.to_numpy(), n_axes=3)

    assert resultat.shape == (len(donnees), 3)
    assert list(resultat.columns) == ["Axe_1", "Axe_2", "Axe_3"]


def test_transform_refuse_un_mauvais_nombre_de_colonnes(acp_normed):
    with pytest.raises(ValueError, match="colonnes"):
        acp_normed.transform(np.zeros((3, 2)))


def test_transform_refuse_des_colonnes_manquantes(acp_normed, donnees):
    df = donnees.rename(columns={"age": "age_2"})

    with pytest.raises(ValueError, match="Colonnes manquantes"):
        acp_normed.transform(df)


def test_transform_refuse_les_nan(acp_normed, donnees):
    df = donnees.copy()
    df.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="valeurs manquantes"):
        acp_normed.transform(df)


def test_transform_avant_fit_leve_une_erreur(donnees):
    with pytest.raises(RuntimeError, match="pas encore entraînée"):
        MonACP().transform(donnees)


def test_changement_de_methode_apres_fit_leve_une_erreur(acp_normed, donnees):
    acp_normed.method = "spearman"

    with pytest.raises(RuntimeError, match="méthode a changé"):
        acp_normed.transform(donnees)

    with pytest.raises(RuntimeError, match="méthode a changé"):
        acp_normed.bilan()


@pytest.mark.parametrize("n_axes", [0, -1, 7, 2.5])
def test_n_axes_invalides(acp_normed, donnees, n_axes):
    with pytest.raises(ValueError, match="n_axes doit être"):
        acp_normed.get_coord_variables(n_axes=n_axes)

    with pytest.raises(ValueError, match="n_axes doit être"):
        acp_normed.transform(donnees, n_axes=n_axes)


def test_get_coord_variables(acp_normed):
    coord = acp_normed.get_coord_variables(n_axes=2)

    assert coord.shape == (6, 2)
    assert list(coord.columns) == ["Axe_1", "Axe_2"]
    assert list(coord.index) == acp_normed.feature_names_
    # Pour une ACP normée, les variables restent dans le cercle unité.
    assert (coord.pow(2).sum(axis=1) <= 1 + 1e-9).all()


def test_get_coord_variables_borne_les_valeurs_propres_negatives(acp_normed):
    acp_normed.eigenvalues_ = acp_normed.eigenvalues_.copy()
    acp_normed.eigenvalues_[0] = -1e-16

    coord = acp_normed.get_coord_variables(n_axes=1)

    assert not coord.isna().any().any()
    assert (coord == 0).all().all()


def test_spearman_retrouve_les_rangs_du_fit():
    df = pd.DataFrame({"a": [1.0, 2.0, 2.0, 4.0, 5.0], "b": [5.0, 4.0, 3.0, 2.0, 1.0]})
    acp = MonACP(method="spearman").fit(df)
    coords = acp.transform(df)

    rangs = rankdata(df.to_numpy(), method="average", axis=0)
    Z = (rangs - rangs.mean(axis=0)) / rangs.std(axis=0)
    attendu = pd.DataFrame(
        Z @ acp.eigenvectors_[:, :2],
        index=df.index,
        columns=["Axe_1", "Axe_2"],
    )

    np.testing.assert_allclose(
        coords.to_numpy(), attendu.to_numpy(), rtol=1e-12, atol=1e-12
    )


def test_spearman_nouvel_individu_utilise_les_rangs_du_fit():
    df = pd.DataFrame({"x": [1.0, 2.0, 2.0, 4.0, 5.0]})
    acp = MonACP(method="spearman").fit(df)
    coords_train = acp.transform(df, n_axes=1)["Axe_1"]

    # Une valeur égale à une valeur d'entraînement doit retrouver le même rang.
    coord_egal = acp.transform(pd.DataFrame({"x": [2.0]}, index=["nouveau"]), n_axes=1)[
        "Axe_1"
    ].iloc[0]
    assert coord_egal == pytest.approx(coords_train.iloc[1], abs=1e-12)

    # Une valeur intermédiaire doit recevoir le rang moyen exact :
    # 3 valeurs inférieures à 3 et aucune valeur égale -> rang 3.5.
    coord_entre = acp.transform(
        pd.DataFrame({"x": [3.0]}, index=["nouveau"]), n_axes=1
    )["Axe_1"].iloc[0]
    rangs_train = rankdata(df["x"], method="average")
    Z_attendu = (3.5 - rangs_train.mean()) / rangs_train.std()
    attendu = Z_attendu * acp.eigenvectors_[0, 0]
    assert coord_entre == pytest.approx(attendu, abs=1e-12)


def test_tracer_acp_retourne_une_figure(acp_normed, donnees):
    fig = acp_normed.tracer_acp(
        acp_normed.transform(donnees), acp_normed.get_coord_variables()
    )

    assert fig is not None
    plt.close(fig)


def test_tracer_acp_refuse_des_colonnes_manquantes(acp_normed, donnees):
    df_individus = acp_normed.transform(donnees).rename(columns={"Axe_1": "x"})

    with pytest.raises(ValueError, match="Axe_1 et Axe_2"):
        acp_normed.tracer_acp(df_individus, acp_normed.get_coord_variables())


def test_bilan_avant_fit_leve_une_erreur():
    with pytest.raises(RuntimeError, match="pas encore entraînée"):
        MonACP().bilan()
