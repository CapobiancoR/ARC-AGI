import json
import random
import torch
from copy import deepcopy

def rotate_tensor(tensor, k):
    # ruota di k*90° intorno agli assi (0,1)
    return torch.rot90(tensor, k=k, dims=(0,1))

def apply_permutation_tensor(tensor, perm_map):
    # perm_map è un Tensor di shape (10,) tale che perm_map[i] = nuovo valore per i
    return perm_map[tensor]

def augment_example(inp_mat, out_mat, num_perms=50, device='cpu'):
    inp = torch.tensor(inp_mat, device=device, dtype=torch.long)
    out = torch.tensor(out_mat, device=device, dtype=torch.long)
    augmented = []
    # genera 50 permutazioni casuali
    perms = [torch.randperm(10, device=device) for _ in range(num_perms)]
    # per ogni rotazione (1×90°, 2×90°, 3×90°)
    for k in [1,2,3]:
        inp_rot = rotate_tensor(inp, k)
        out_rot = rotate_tensor(out, k)
        # applica ciascuna permutazione
        for perm in perms:
            inp_aug = apply_permutation_tensor(inp_rot, perm)
            out_aug = apply_permutation_tensor(out_rot, perm)
            augmented.append({
                'input': inp_aug.cpu().tolist(),
                'output': out_aug.cpu().tolist()
            })
    return augmented

def augment_challenge(challenge, num_perms=50, device='cpu'):
    new_train = []
    for ex in challenge['train']:
        new_train.extend(
            augment_example(ex['input'], ex['output'], num_perms, device)
        )
    # opzionale: se vuoi includere anche gli originali, decommenta
    # new_train.extend(deepcopy(challenge['train']))
    return {
        'train': new_train,
        'test': deepcopy(challenge['test'])
    }

def main(input_path, output_path, num_perms=50):
    # seleziona device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    with open(input_path, 'r') as f:
        data = json.load(f)

    augmented = {}
    for key, challenge in data.items():
        print(f"Augmenting '{key}': {len(challenge['train'])} esempi → ", end='')
        aug = augment_challenge(challenge, num_perms, device)
        augmented[key] = aug
        print(f"{len(aug['train'])} esempi nel nuovo train")

    with open(output_path, 'w') as f:
        json.dump(augmented, f, indent=2)
    print(f"Salvato in '{output_path}'")

if __name__ == '__main__':
    # sostituisci con i percorsi corretti
    main(
        input_path='./dataset/arc-agi_training_challenges.json',
        output_path='./dataset/arc-agi_training_challenges_augmented.json',
        num_perms=50
    )
