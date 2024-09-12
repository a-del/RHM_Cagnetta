import numpy as np
import torch
import argparse

from datasets import RandomHierarchyModel
from datasets.hierarchical import number2base
from datasets.hierarchical import number2base as dec2base
from datasets.random_hierarchy_model import RandomHierarchyModel as RandomHierarchyModel2


seed = 999


parser = argparse.ArgumentParser()
args = parser.parse_args()


def figure_stuff(args, n_samples=None, max_samples=50000):
    m = args.m
    s = args.s
    num_layers = args.num_layers
    nc = args.num_classe

    Pmax = m ** ((s ** num_layers - 1) // (s - 1)) * nc

    if n_samples is None:
        n_samples = Pmax
    n_samples = min(Pmax, n_samples, max_samples)

    rhm = RandomHierarchyModel(num_features=args.num_features,
                               m=args.m, num_layers=args.num_layers, num_classes=args.num_classes, input_format='long',
                               seed=seed, train=True, transform=None, testsize=0, max_dataset_size=n_samples)
    base_indices = rhm.samples_indices

    d = Pmax // nc
    y = base_indices.div(d, rounding_mode='floor')
    samples_indices = base_indices % d

    indices = [y]
    for l in range(num_layers):
        d //= m ** (s ** l)
        layer_indices = samples_indices.div(d, rounding_mode='floor')
        samples_indices = samples_indices % d
        indices.append(layer_indices)

    seqs = torch.stack(indices).t()
    id_to_seq = dict(zip(samples_indices, seqs))


# Now, a function the generates sequences for substitues and replacements (READ!)
def substitution(seqs, l):
    # should act on rules, not on seq!!
    pass


def find_relationship(idx1, idx2, L, dcts, m, s, nc):
    """
    idx1: index of a sample
    idx2: index of a sample
    L: number of layers
    dcts: list of dict per layer.
    """

    Pmax = m ** ((s ** L - 1) // (s - 1)) * nc
    # x = paths[-1].reshape(num_classes, *sum([(m, s) for _ in range(num_layers)], ()))  # [nc, m, s, m, s, ...]

    groups_size = Pmax // nc
    y1 = idx1 // groups_size
    y2 = idx2 //groups_size
    spl1 = idx1 % groups_size
    spl2 = idx2 % groups_size

    same_until_now = y1 == y2

    # indices = []
    for l in range(L):
        if not same_until_now:
            dcts[l][(idx1, idx2)] = ""
            continue

        groups_size //= m ** (s ** l)
        layer_indices_1 = spl1 // groups_size
        layer_indices_2 = spl2 // groups_size

        rules1 = number2base(layer_indices_1, m, string_length=s ** l)  # len(samples_indices), s ** l
        rules2 = number2base(layer_indices_2, m, string_length=s ** l)  # len(samples_indices), s ** l
        rules1 = (
            rules1[:, None]  # len(samples_indices), 1, s ** l
            .repeat(1, s ** (L - l - 1), 1)  # len(samples_indices), s**(num_layers - l - 1), s ** l
            .permute(0, 2, 1)  # len(samples_indices), s ** l, s**(num_layers - l - 1)
            .flatten(1)  # len(samples_indices), s ** (num_layers - 1)
        )
        rules2 = (
            rules2[:, None]  # len(samples_indices), 1, s ** l
            .repeat(1, s ** (L - l - 1), 1)  # len(samples_indices), s**(num_layers - l - 1), s ** l
            .permute(0, 2, 1)  # len(samples_indices), s ** l, s**(num_layers - l - 1)
            .flatten(1)  # len(samples_indices), s ** (num_layers - 1)
        )

        # indices.append(rules)

        same_until_now = same_until_now and all(spl1 == spl2)

        spl1 = spl1 % groups_size
        spl2 = spl2 % groups_size

    # yi = y[:, None].repeat(1, s ** (L - 1))
    #
    # x = x[tuple([yi, *indices])].flatten(1)

    return x, y






#### Francesco's code ####

def sample_synonyms_from_rules(samples, rules, n, m, s, L, level, positions):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and the sampled indices.

    Args:
        samples: A tensor of size [batch_size, I], with I from 0 to max_data-1, containing the indices of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).
        level: The level of the representations to replace (int in 1,...,L+1).
        positions: The list of position of the representations to replace (ints in 1,...,s**(L-level)).

    Returns:
        A set of inputs synonymic to the output of sample_data_from_rules.
    """

    max_data = n * m ** ((s**L-1)//(s-1))
    data_per_hl = max_data // n # div by num_classes to get number of data per class

    high_level = samples.div(data_per_hl, rounding_mode='floor') # div by data_per_hl to get class index (run in range(n))
    low_level = samples % data_per_hl # compute remainder (run in range(data_per_hl))

    labels = high_level # labels are the classes (features of highest level)
    features = labels # init input features as labels (rep. size 1)
    size = 1

    for l in range(L):

        choices = m**(size)
        data_per_hl = data_per_hl // choices # div by num_choices to get number of data per high-level feature

        high_level = low_level.div( data_per_hl, rounding_mode='floor') # div by data_per_hl to get the index of the low-level representation (1 index in range(m**size))
        high_level = dec2base(high_level, m, size) # convert to base m (size indices in range(m), squeeze needed if index already in base m)
        if l==(L-level): # randomly change low=level representations at required positions
            for pos in positions:
                high_level[:,pos] = torch.randint(m, (high_level.size(0),))
        high_level = high_level.squeeze()

        features = rules[l][features, high_level]     # apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
        features = features.flatten(start_dim=1) # flatten to tensor of size (batch_size, size*s)
        size *= s # rep. size increases by s at each level

        low_level = low_level % data_per_hl # compute remainder (run in range(data_per_hl))

    return features


def sample_noisy_from_rules(samples, rules, n, m, s, L, v, level, positions):
    """
    Create data of the Random Hierarchy Model starting from a set of rules and the sampled indices.

    Args:
        samples: A tensor of size [batch_size, I], with I from 0 to max_data-1, containing the indices of the data to be sampled.
        rules: A dictionary containing the rules for each level of the hierarchy.
        n: The number of classes (int).
        m: The number of synonymic lower-level representations (multiplicity, int).
        s: The size of lower-level representations (int).
        L: The number of levels in the hierarchy (int).
        v: The number of features (int).
        level: The level (-1) of the features to replace (int in 1,...,L-1).
        positions: The list of position of the features to replace (ints in 1,...,s**(L-level)).

    Returns:
        A set of corrupted versions of the ouput of sample_data_from_rules.
    """

    max_data = n * m ** ((s**L-1)//(s-1))
    data_per_hl = max_data // n # div by num_classes to get number of data per class

    high_level = samples.div(data_per_hl, rounding_mode='floor') # div by data_per_hl to get class index (run in range(n))
    low_level = samples % data_per_hl # compute remainder (run in range(data_per_hl))

    labels = high_level # labels are the classes (features of highest level)
    features = labels # init input features as labels (rep. size 1)
    size = 1

    for l in range(L):

        choices = m**(size)
        data_per_hl = data_per_hl // choices # div by num_choices to get number of data per high-level feature

        high_level = low_level.div( data_per_hl, rounding_mode='floor') # div by data_per_hl to get the index of the low-level representation (1 index in range(m**size))
        high_level = dec2base(high_level, m, size) # convert to base m (size indices in range(m), squeeze needed if index already in base m)
        if l==(L-level): # randomly change features low-level representations at required positions
            for pos in positions:
                features[:,pos] = torch.randint(v, (features.size(0),))
                high_level[:,pos] = torch.randint(m, (high_level.size(0),))
        high_level = high_level.squeeze()

        features = rules[l][features, high_level]     # apply l-th rule to expand to get features at the lower level (tensor of size (batch_size, size, s))
        features = features.flatten(start_dim=1)   # flatten to tensor of size (batch_size, size*s)
        size *= s # rep. size increases by s at each level

        low_level = low_level % data_per_hl   # compute remainder (run in range(data_per_hl))

    return features






#### my code again ####

def create_datasets(args, n_versions=5, max_ds_size=20000):
    n = args.num_features
    m = args.m
    s = args.s
    L = args.num_layers
    nc = args.num_classes

    seed1 = 0
    rhm = RandomHierarchyModel2(num_features=n, num_synonyms=m, num_layers=L,
                               num_classes=nc, tuple_size=s, seed_rules=seed1, seed_sample=seed1,
                               input_format="long", max_dataset_size=max_ds_size)
    base_features = rhm.features - 1   # for some reason, format="long" adds +1 to features
    base_sample_indices = rhm.samples
    rules = rhm.rules

    syns = {lt: [] for lt in range(1, L+1)}
    nois = {lt: [] for lt in range(1, L)}
    for k in range(n_versions):
        # seed2 = 0+k
        # rng = np.random.default_rng(seed2)

        for l in range(1, L+1):
            pos = max(0, s**(L-l)-2)   # penultimate
            # pos = rng.integers(0, s**(L-l))
            syns[l].append(sample_synonyms_from_rules(base_sample_indices, rules, n, m, s, L, l, positions=[pos]))
        # syns[l]: synonyms of base_features, obtained by replacing the encoding of a feature of level l.
        # Level is counted from bottom (final encoding), i.e. level l=1 is the lowest hidden variable; only the final
        # encoding layer is changed. List of these synonyms per version.

        for l in range(1, L):   # todo? could implement sample_noisy_from_rules for level=0
            pos = max(0, s**(L-l)-2)   # penultimate
            # pos = rng.integers(0, s**(L-l))
            nois[l].append(sample_noisy_from_rules(base_sample_indices, rules, nc, m, s, L, n, l, positions=[pos]))
        # nois[l]: noisy version of base_features, obtained by replacing a feature of level l.
        # Level is counted from bottom (final encoding), i.e. level l=1 is the lowest hidden variable; only the lowest
        # hidden variable and the final encoding layer is changed.

    return base_features, syns, nois


def create_encodings_datasets(model, base_features, synsd, noisd, bs=400):
    # todo? could optimize and not compute and store several times features that are found multiple times across
    #  base_features, syns[:], and nois[:]
    base_encs = {}
    for l, encs in get_encodings(model, base_features, bs).items():
        base_encs[l] = encs

    syns_per_ltransfo = {}
    for ltransfo, syns in synsd.items():
        syns_per_ltransfo[ltransfo] = {l: [] for l in range(model.num_layers)}
        for synonyms in syns:
            for l, encs in get_encodings(model, synonyms, bs).items():
                syns_per_ltransfo[ltransfo][l].append(get_encodings(model, synonyms, bs))
    nois_per_ltransfo = {}
    for ltransfo, nois in noisd.items():
        nois_per_ltransfo[ltransfo] = {l: [] for l in range(model.num_layers)}
        for noisy in nois:
            for l, encs in get_encodings(model, noisy, bs).items():
                nois_per_ltransfo[ltransfo][l].append(encs)
    return base_encs, syns_per_ltransfo, nois_per_ltransfo


def get_encodings(model, features, bs=400):
    N = len(features)
    features_1hot = ...(features)   # TODO
    all_encs_per_layer = {l: [] for l in range(model.num_layers)}
    for k in range(int(np.ceil(N, bs))):
        y, outs, pres = model(features_1hot[k * bs:(k + 1 * bs)])
        # pres: list of shape (bs, chans, len) for each layer
        for l, pre in enumerate(pres):
            all_encs_per_layer[l].append(pre)
    for l in all_encs_per_layer:
        all_encs_per_layer[l] = torch.cat(all_encs_per_layer[l])
    return all_encs_per_layer


def compute_sensitivity(base_encs, other_encs):
    nv = len(other_encs)

    mean = base_encs.mean(0, keepdim=True)
    std = base_encs.std(0, keepdim=True)
    centered_base = (base_encs - mean) / std

    base = centered_base.repeat(nv, 1, 1, 1)   # nv, nb_data, chans, len
    others = torch.stack(other_encs, dim=0)   # nv, nb_data, chans, len
    others = (others - mean) / std

    sensitivity = base*others.sum(dim=2)  # dot product over chans   # nv, nb_data, len
    sensitivity = sensitivity.mean()
    return sensitivity



def full_algo():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # TODO load model...
    model = None
    # TODO set layerwise = True

    base_features, syns, nois = create_datasets(args)
    # base features: nb_data, len
    # syns and nois: dict l_transfo -> list, for each version, of transformed features (each: nb_data, len)
    base_encs, syns_encs, nois_encs = create_encodings_datasets(model, base_features, syns, nois)
    # base_encs: lenc -> encodings (nb_data, len)
    # syns_encs and nois_encs: dict l_transfo -> [lencs -> list for each version, of transformed encs (each (nb_data, len))]

    syn_sensitivity = {ltransfo:
                           {lenc:
                                compute_sensitivity(base_encs[lenc], syns_encs[ltransfo][lenc])
                            for lenc in syns_encs[ltransfo]}
                       for ltransfo in syns_encs}
    noise_sensitivity = {ltransfo:
                           {lenc:
                                compute_sensitivity(base_encs[lenc], nois_encs[ltransfo][lenc])
                            for lenc in nois_encs[ltransfo]}
                       for ltransfo in nois_encs}

    # TODO need this for different models trained with different Ps
    









# load a trained model (or train it?)

# compute encodings of all pts of dataset
# standardise encodings


# we need to associate the encodings with the whole coding structure of the RHM
#  could store sampled paths, xs and ys together in rhm?


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Hierarchical dataset #
    parser.add_argument("--num_features", type=int, default=4)  # v
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--s", type=int, default=2)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_classes", type=int, default=3)

    args = parser.parse_args()
    create_datasets(args, max_ds_size=1000)