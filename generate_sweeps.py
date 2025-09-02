import os
from pathlib import Path
from string import Template

# -------- CONFIG --------
models = ["bnt"]
grl_flags = [True, False]
sites = ["NYU", "PITT", "USM", "KKI", "UM_1"]

# Update these paths as needed
template_path = Path("/Volumes/shared/ai4h/Shared/acq21js/code/abide/sweeps/sweep_template.yaml")
output_dir = Path("/Volumes/shared/ai4h/Shared/acq21js/code/abide/sweeps/agents")
output_dir.mkdir(exist_ok=True)

# Load template
with open(template_path, "r") as f:
    template = Template(f.read())

# Generate one YAML per model/site/GRL combination
for model in models:
    for grl in grl_flags:
        for site in sites:
            substitutions = {
                "MODEL": model,
                "USE_GRL": str(grl).lower(),
                "SITE": site,
                "env": "${env}"  # retain as a literal placeholder
            }

            filled = template.substitute(substitutions)
            filename = f"{model}_{'grl' if grl else 'base'}_{site}.yaml"
            (output_dir / filename).write_text(filled)

print(f"[INFO] Generated {len(models) * len(grl_flags) * len(sites)} sweep YAMLs in {output_dir}/")