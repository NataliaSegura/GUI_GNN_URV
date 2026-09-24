import numpy as np
from rdkit import Chem


ATOM_TYPES = ["C", "N", "O", "S", "H", "P", "F", "Cl", "Br", "I"]

ATOM_TYPES_DTA = ["C","N","O","S","F","Si","P","Cl","Br","Mg","Na","Ca","Fe","As",
                  "Al","I","B","V","K","Tl","Yb","Sb","Sn","Ag","Pd","Co","Se","Ti",
                  "Zn","H","Li","Ge","Cu","Au","Ni","Cd","In","Mn","Zr","Cr","Pt",
                  "Hg","Pb","Unknown"]

HBD_PATTERN = Chem.MolFromSmarts('[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]')

# Aceptor: N u O con pares libres disponibles (definición estándar de Lipinski)
HBA_PATTERN = Chem.MolFromSmarts(
    '[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]')

# Patrón para enlace rotable: Enlace simple (-), no en anillo (!@), entre átomos no terminales (!D1)
FLEXIBILITY_BOND_PATTERN = Chem.MolFromSmarts('[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]')

def atom_type_onehot(symbol):
    vec = [0] * len(ATOM_TYPES)
    if symbol in ATOM_TYPES:
        vec[ATOM_TYPES.index(symbol)] = 1
    return vec


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    features = np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_TYPES)
        + one_of_k_encoding(atom.GetDegree(), [i for i in range(0, 11, 1)])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [i for i in range(0, 11, 1)])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [i for i in range(0, 11, 1)])
        + [atom.GetIsAromatic()],
        dtype=np.float32,
    )

    feature_sum = features.sum()

    if feature_sum != 0:
        features /= feature_sum

    return features


def atom_features_dta(atom):
    features = np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), ATOM_TYPES_DTA)
        + one_of_k_encoding(atom.GetDegree(), [i for i in range(0, 11, 1)])
        + one_of_k_encoding_unk(atom.GetTotalNumHs(), [i for i in range(0, 11, 1)])
        + one_of_k_encoding_unk(atom.GetImplicitValence(), [i for i in range(0, 11, 1)])
        + [atom.GetIsAromatic()],
        dtype=np.float32,
    )

    feature_sum = features.sum()

    if feature_sum != 0:
        features /= feature_sum

    return features
