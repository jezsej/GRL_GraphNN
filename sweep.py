import sys
import subprocess

def patch_and_forward():
    # sweep_to_hydra_map = {
    #     "--grl_lambda": "domain_adaptation.grl_lambda",
    #     "--domain_loss_weight": "domain_adaptation.domain_loss_weight",
    #     "--use_grl": "domain_adaptation.use_grl",
    #     "--model_name": "models.name",
    #     "--lr": "training.lr",
    #     "--seed": "training.seed",
    #     "--run_name": "logging.run_name",
    #     "--site": "dataset.site",
    # }

    # patched_args = []
    # for arg in sys.argv[1:]:
    #     matched = False
    #     for key, hydra_path in sweep_to_hydra_map.items():
    #         if arg.startswith(f"{key}="):
    #             value = arg.split("=", 1)[1]
    #             patched_args.append(f"{hydra_path}={value}")
    #             matched = True
    #             break
    #     if not matched:
    #         patched_args.append(arg)

    cmd = ["python", "run_loso.py"] + sys.argv[1:] #patched_args
    print("[INFO] Launching with patched args:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    patch_and_forward()