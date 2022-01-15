import torch
import pickle as pkl
import argparse

parser = argparse.ArgumentParser(description="Convert model tools")
parser.add_argument(
    "--model", type=str, default="moco", help="coco or densecl convert model tools"
)
parser.add_argument("--model_path", help="original model_path")
parser.add_argument("--new_model_path", help="new model path")


def moco_convert_to_detection(model_path, new_model_path):
    obj = torch.load(model_path, map_location="cpu")
    obj = obj["state_dict"]
    ans = 0
    newmodel = {}
    for k, v in obj.items():
        if not k.startswith("encoder_q."):
            continue
        if k.startswith("encoder_q.fc"):
            continue
        old_k = k
        k = k.replace("encoder_q.", "module.")
        ans += 1
        print(old_k, "->", k)
        newmodel[k] = v.numpy()

    res = {"state_dict": newmodel, "__author__": "MOCO", "matching_heuristics": True}
    with open(new_model_path, "wb") as f:
        pkl.dump(res, f)
    print("orginal model path: %s" % model_path)
    print("new model path: %s" % new_model_path)


def densecl_convert_to_detection(model_path, new_model_path):
    obj = torch.load(model_path, map_location="cpu")
    obj = obj["state_dict"]
    newmodel = {}
    ans = 0
    for k, v in obj.items():
        if not k.startswith("backbone."):
            continue
        old_k = k
        k = k.replace("backbone.", "module.")
        if k.startswith("module.0"):
            k = k.replace("0", "conv1")
        elif k.startswith("module.1"):
            k = k.replace("1", "bn1")
        else:
            for t in [4, 5, 6, 7]:
                k = k.replace("module.{}".format(t), "module.layer{}".format(t - 3))
        print(old_k, "->", k)
        ans += 1
        newmodel[k] = v.numpy()
    res = {"state_dict": newmodel, "__author__": "MOCO", "matching_heuristics": True}
    with open(new_model_path, "wb") as f:
        pkl.dump(res, f)
    print("orginal model path: %s" % model_path)
    print("new model path: %s" % new_model_path)


def main():
    args = parser.parse_args()
    if args.model == "moco":
        moco_convert_to_detection(args.model_path, args.new_model_path)
    elif args.model == "densecl":
        densecl_convert_to_detection(args.model_path, args.new_model_path)
    else:
        raise ValueError("model must be moco or densecl")


if __name__ == "__main__":
    main()
    # densecl_convert_to_detection()
    # moco_convert_to_detection()
