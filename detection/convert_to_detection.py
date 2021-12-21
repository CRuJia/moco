import torch
import pickle as pkl

if __name__ == "__main__":
    model_path = "models/resnet50/xjb/checkpoint_0199.pth.tar"
    obj = torch.load(model_path, map_location="cpu")
    obj = obj["state_dict"]

    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("encoder_q."):
            continue
        if k.startswith("encoder_q.fc"):
            continue
        old_k = k
        k = k.replace("encoder_q.", "module.")

        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"state_dict": newmodel, "__author__": "MOCO", "matching_heuristics": True}
    file = "new_model.pkl"
    with open(file, "wb") as f:
        pkl.dump(res, f)
