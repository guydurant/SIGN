from openbabel import pybel, OBMol
from preprocess_pdbbind import (
    pocket_atom_num_from_mol2,
    gen_pocket_graph,
    dist_filter,
    distance_matrix,
    cons_lig_pock_graph_with_spatial_context,
)
from featurizer import Featurizer
from tqdm import tqdm
import os
from pymol import cmd
import numpy as np
import pandas as pd
import pickle
from argparse import ArgumentParser
import paddle
import paddle.nn.functional as F
from pgl.utils.data import Dataloader, Dataset
from sklearn.model_selection import train_test_split
from dataset import ComplexDataset, collate_fn
from model import SIGN
from utils import rmse, mae, sd, pearson
import time
from train import evaluate, setup_seed

paddle.seed(123)


def make_protein_pocket_file(protein_file, ligand_file, key, csv_file):
    if not os.path.exists(f'data/scratch/{csv_file.split("/")[-1].split(".")[0]}'):
        os.mkdir(f'data/scratch/{csv_file.split("/")[-1].split(".")[0]}')
    cmd.load(protein_file, "protein")
    # cmd.load(f'data/scratch/{protein_file.split("/")[-1].split(".")[0]}_charged.mol2', 'protein')
    cmd.load(ligand_file, "ligand")
    cmd.select("pocket", "byres (ligand around 6)")
    cmd.save(
        f'data/scratch/{csv_file.split("/")[-1].split(".")[0]}/{key}_pocket.mol2',
        "pocket",
    )
    cmd.delete("all")
    return None


def calculate_charges_for_pocket(key):
    os.system(
        f'echo "open data/scratch/{key}_pocket.pdb \n addh \n addcharge \n  save tmp.mol2 \n exit" | chimerax --nogui'
    )
    # Do not use TIP3P atom types, pybel cannot read them
    os.system(
        f"sed 's/H\.t3p/H    /' tmp.mol2 | sed 's/O\.t3p/O\.3  /' > data/scratch/{key}_pocket.mol2"
    )


def pocket_atom_num_from_mol2(key, csv_file):
    n = 0
    with open(
        f"data/scratch/{csv_file.split('/')[-1].split('.')[0]}/{key}_pocket.mol2"
    ) as f:
        for line in f:
            if "<TRIPOS>ATOM" in line:
                break
        for line in f:
            cont = line.split()
            if "<TRIPOS>BOND" in line or cont[7] == "HOH":
                break
            n += int(cont[5][0] != "H")
    return n


def load_csv(csv_file, data_dir):
    df = pd.read_csv(csv_file)
    protein_files = [
        os.path.join(data_dir, protein_file) for protein_file in df["protein"]
    ]
    ligand_files = [os.path.join(data_dir, ligand_file) for ligand_file in df["ligand"]]
    keys = df["key"]
    pks = df["pk"]
    return protein_files, ligand_files, keys, pks


def gen_feature(protein_file, ligand_file, key, csv_file, featurizer):
    charge_idx = featurizer.FEATURE_NAMES.index("partialcharge")
    # calculate_charges(ligand_file)
    # ligand = next(pybel.readfile('mol2', f'data/scratch/{ligand_file.split("/")[-1].split(".")[0]}_charged.mol2'))
    ligand = next(pybel.readfile("sdf", ligand_file))
    ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)
    # calculate_charges(protein_file)
    make_protein_pocket_file(protein_file, ligand_file, key, csv_file)
    # calculate_charges_for_pocket(key)
    try:
        pocket = next(
            pybel.readfile(
                "mol2",
                f'data/scratch/{csv_file.split("/")[-1].split(".")[0]}/{key}_pocket.mol2',
            )
        )
        pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
    except StopIteration:
        pocket_coords, pocket_features = [[]], []
        pocket = pybel.Molecule(OBMol())
    node_num = pocket_atom_num_from_mol2(key, csv_file)
    pocket_coords = pocket_coords[:node_num]
    pocket_features = pocket_features[:node_num]
    # try:
    # assert (ligand_features[:, charge_idx] != 0).any()
    # assert (pocket_features[:, charge_idx] != 0).any()
    assert (ligand_features[:, :9].sum(1) != 0).all()
    # except:
    # print('Error in feature generation for ', key)
    lig_atoms, pock_atoms = [], []
    for i, atom in enumerate(ligand):
        if atom.atomicnum > 1:
            lig_atoms.append(atom.atomicnum)
    for i, atom in enumerate(pocket):
        if atom.atomicnum > 1:
            pock_atoms.append(atom.atomicnum)
    # for x in pock_atoms[node_num:]:
    #     assert x == 8
    pock_atoms = pock_atoms[:node_num]
    assert len(lig_atoms) == len(ligand_features) and len(pock_atoms) == len(
        pocket_features
    )

    ligand_edges = gen_pocket_graph(ligand)
    pocket_edges = gen_pocket_graph(pocket)
    return {
        "lig_co": ligand_coords,
        "lig_fea": ligand_features,
        "lig_atoms": lig_atoms,
        "lig_eg": ligand_edges,
        "pock_co": pocket_coords,
        "pock_fea": pocket_features,
        "pock_atoms": pock_atoms,
        "pock_eg": pocket_edges,
    }


def pairwise_atomic_types(
    protein_files, ligand_files, keys, processed_dict, atom_types, atom_types_
):
    atom_keys = [(i, j) for i in atom_types_ for j in atom_types]
    for i in tqdm(range(len(keys))):
        ligand = next(pybel.readfile("sdf", ligand_files[i]))
        pocket = next(pybel.readfile("pdb", protein_files[i]))
        coords_lig = np.vstack([atom.coords for atom in ligand])
        coords_poc = np.vstack([atom.coords for atom in pocket])
        atom_map_lig = [atom.atomicnum for atom in ligand]
        atom_map_poc = [atom.atomicnum for atom in pocket]
        dm = distance_matrix(coords_lig, coords_poc)
        # print(coords_lig.shape, coords_poc.shape, dm.shape)
        ligs, pocks = dist_filter(dm, 12)
        # print(len(ligs),len(pocks))

        fea_dict = {k: 0 for k in atom_keys}
        for x, y in zip(ligs, pocks):
            x, y = atom_map_lig[x], atom_map_poc[y]
            if x not in atom_types or y not in atom_types_:
                continue
            fea_dict[(y, x)] += 1

        processed_dict[keys[i]]["type_pair"] = list(fea_dict.values())

    return processed_dict


def process_dataset(csv_file, data_dir, cutoff):
    protein_files, ligand_files, keys, pks = load_csv(csv_file, data_dir)
    # core_set_list = [x for x in os.listdir(core_path) if len(x) == 4]
    # refined_set_list = [x for x in os.listdir(refined_path) if len(x) == 4]

    # path = refined_path

    # atomic sets for long-range interactions
    atom_types = [6, 7, 8, 9, 15, 16, 17, 35, 53]
    atom_types_ = [6, 7, 8, 16]

    # atomic feature generation
    featurizer = Featurizer(save_molecule_codes=False)
    processed_dict = {}
    for i in tqdm(range(len(keys))):
        processed_dict[keys[i]] = gen_feature(
            protein_files[i], ligand_files[i], keys[i], csv_file, featurizer
        )
        os.system(f'rm data/scratch/{csv_file.split("/")[-1].split(".")[0]}/{keys[i]}*')

    # interaction features
    processed_dict = pairwise_atomic_types(
        protein_files, ligand_files, keys, processed_dict, atom_types, atom_types_
    )
    # load pka (binding affinity) data
    pk_dict = {key: pk for key, pk in zip(keys, pks)}
    data_dict = processed_dict
    for k, v in processed_dict.items():
        v["pk"] = pk_dict[k]
        data_dict[k] = v

    processed_id, processed_data, processed_pk = [], [], []

    for k, v in tqdm(data_dict.items()):
        ligand = (v["lig_fea"], v["lig_co"], v["lig_atoms"], v["lig_eg"])
        pocket = (v["pock_fea"], v["pock_co"], v["pock_atoms"], v["pock_eg"])
        graph = cons_lig_pock_graph_with_spatial_context(
            ligand,
            pocket,
            add_fea=3,
            theta=cutoff,
            keep_pock=False,
            pocket_spatial=True,
        )
        cofeat, pk = v["type_pair"], v["pk"]
        graph = list(graph) + [cofeat]
        processed_id.append(k)
        processed_data.append(graph)
        processed_pk.append(pk)

    # split train and valid
    # train_idxs, valid_idxs = random_split(len(refined_data), split_ratio=0.9, seed=2020, shuffle=True)
    combined_features_labels = (processed_data, processed_pk)
    with open(
        os.path.join(
            f'data/features/{csv_file.split("/")[-1].split(".")[0]}' + "_features.pkl"
        ),
        "wb",
    ) as f:
        pickle.dump(combined_features_labels, f)
    return None


def train_model(args, model, trn_loader, val_loader):
    # learning rate decay and optimizer
    epoch_step = len(trn_loader)
    boundaries = [
        i for i in range(args.dec_step, args.epochs * epoch_step, args.dec_step)
    ]
    values = [args.lr * args.lr_dec_rate**i for i in range(0, len(boundaries) + 1)]
    scheduler = paddle.optimizer.lr.PiecewiseDecay(
        boundaries=boundaries, values=values, verbose=False
    )
    optim = paddle.optimizer.Adam(
        learning_rate=scheduler, parameters=model.parameters()
    )
    # l1_loss = paddle.nn.loss.L1Loss(reduction='sum')

    rmse_val_best, res_tst_best = 1e9, ""
    running_log = ""
    print("Start training model...")
    for epoch in tqdm(range(1, args.epochs + 1)):
        sum_loss, sum_loss_inter = 0, 0
        model.train()
        start = time.time()
        for batch_data in trn_loader:
            a2a_g, b2a_g, b2b_gl, feats, types, counts, y = batch_data
            feats_hat, y_hat = model(a2a_g, b2a_g, b2b_gl, types, counts)

            # loss function
            loss = F.l1_loss(y_hat, y, reduction="sum")
            loss_inter = F.l1_loss(feats_hat, feats, reduction="sum")
            loss += args.lambda_ * loss_inter
            loss.backward()
            optim.step()
            optim.clear_grad()
            scheduler.step()

            sum_loss += loss
            sum_loss_inter += loss_inter

        end_trn = time.time()
        rmse_val, mae_val, sd_val, r_val = evaluate(model, val_loader)
        end_val = time.time()
        log = (
            "-----------------------------------------------------------------------\n"
        )
        log += "Epoch: %d, loss: %.4f, loss_b: %.4f, time: %.4f, val_time: %.4f.\n" % (
            epoch,
            sum_loss / (epoch_step * args.batch_size),
            sum_loss_inter / (epoch_step * args.batch_size),
            end_trn - start,
            end_val - end_trn,
        )
        log += "Val - RMSE: %.6f, MAE: %.6f, SD: %.6f, R: %.6f.\n" % (
            rmse_val,
            mae_val,
            sd_val,
            r_val,
        )
        # print(log)

        if rmse_val < rmse_val_best:
            rmse_val_best = rmse_val
            if args.save_model:
                obj = {"model": model.state_dict(), "epoch": epoch}
                # path = os.path.join(args.model_dir, 'saved_model')
                paddle.save(obj, f"data/models/{args.model_name}")
                # model.save(f'data/models/{args.model_name}')
                # pickle.dump(model, open(f'data/models/{args.model_name}.pkl', 'wb'))
        # f.close()

    # # f = open(os.path.join(args.model_dir, 'log.txt'), 'w')
    # f.write(running_log + res_tst_best)
    # f.close()


def predict(model, loader, csv_file, data_dir):
    test_index = pd.read_csv(csv_file)["key"].tolist()
    model.eval()
    y_hat_list = []
    y_list = []
    for batch_data in loader:
        a2a_g, b2a_g, b2b_gl, feats, types, counts, y = batch_data
        _, y_hat = model(a2a_g, b2a_g, b2b_gl, types, counts)
        y_hat_list += y_hat.tolist()
        y_list += y.tolist()

    y_hat = np.array(y_hat_list).reshape(
        -1,
    )
    y = np.array(y_list).reshape(
        -1,
    )
    return pd.DataFrame({"key": test_index, "pred": y_hat, "pk": y})


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--csv_file", type=str, default="train.csv")
    parser.add_argument("--val_csv_file", type=str)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--val_data_dir", type=str)
    parser.add_argument("--model_name", type=str, default="test")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--save_model", action="store_true", default=True)

    parser.add_argument("--lambda_", type=float, default=1.75)
    parser.add_argument("--feat_drop", type=float, default=0.2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_dec_rate", type=float, default=0.5)
    parser.add_argument("--dec_step", type=int, default=8000)
    parser.add_argument("--stop_epoch", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=300)

    parser.add_argument("--num_convs", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--infeat_dim", type=int, default=36)
    parser.add_argument("--dense_dims", type=str, default="128*4,128*2,128")

    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--cut_dist", type=float, default=5.0)
    parser.add_argument("--num_angle", type=int, default=6)
    parser.add_argument("--merge_b2b", type=str, default="cat")
    parser.add_argument("--merge_b2a", type=str, default="mean")
    args = parser.parse_args()
    # get as full path not relative path
    args.data_dir = os.path.abspath(args.data_dir)
    args.activation = F.relu
    args.dense_dims = [eval(dim) for dim in args.dense_dims.split(",")]
    if args.seed:
        setup_seed(args.seed)

    # if not os.path.isdir(args.model_dir):
    #     os.mkdir(args.model_dir)

    if args.cuda == "-1":
        print("Using CPU")
        paddle.set_device("cpu")
    else:
        print("Using GPU")
        paddle.set_device("gpu:%s" % args.cuda)
    if args.train:
        if not os.path.exists(
            f'data/features/{args.csv_file.split("/")[-1].split(".")[0]}_features.pkl'
        ):
            print("Extracting features...")
            process_dataset(args.csv_file, args.data_dir, args.cut_dist)
        if args.val_csv_file is not None:
            if not os.path.exists(
                f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.pkl'
            ):
                print("Extracting features...")
                process_dataset(args.val_csv_file, args.val_data_dir, args.cut_dist)
        trn_complex = ComplexDataset(
            "data/features",
            f"{args.csv_file.split('/')[-1].split('.')[0]}_features",
            args.cut_dist,
            args.num_angle,
        )
        if args.val_csv_file is not None:
            val_complex = ComplexDataset(
                "data/features",
                f"{args.val_csv_file.split('/')[-1].split('.')[0]}_features",
                args.cut_dist,
                args.num_angle,
            )
        else:
            # sample 1000 from train set
            train_data, val_data = train_test_split(
                trn_complex, test_size=1000, random_state=42
            )

            trn_complex = train_data
            val_complex = val_data

        trn_loader = Dataloader(
            trn_complex,
            args.batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=collate_fn,
        )
        val_loader = Dataloader(
            val_complex,
            args.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )

        model = SIGN(args)
        train_model(args, model, trn_loader, val_loader)
    if args.predict:
        if not os.path.exists(
            f'data/features/{args.val_csv_file.split("/")[-1].split(".")[0]}_features.pkl'
        ):
            print("Extracting features...")
            process_dataset(args.val_csv_file, args.val_data_dir, args.cut_dist)
        obj = paddle.load(f"data/models/{args.model_name}")
        # initiliase model with state_dict
        model = SIGN(args)
        model.set_state_dict(obj["model"])
        val_complex = ComplexDataset(
            "data/features",
            f"{args.val_csv_file.split('/')[-1].split('.')[0]}_features",
            args.cut_dist,
            args.num_angle,
        )
        val_loader = Dataloader(
            val_complex,
            args.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )
        df = predict(model, val_loader, args.val_csv_file, args.val_data_dir)
        df.to_csv(
            f'data/results/{args.model_name}_{args.val_csv_file.split("/")[-1]}',
            index=False,
        )
    if not args.train and not args.predict:
        print("Please specify either --train or --predict or both")
