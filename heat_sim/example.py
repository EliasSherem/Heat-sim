"""Prepare and optionally run a bounded, reproducible study from a real job."""
from dataclasses import asdict
from pathlib import Path
import argparse
import subprocess
import sys

from .build_solver import BuildConfig
from .job_import import load_job, save_imported_job
from .materials import ILLUSTRATIVE_ABSORPTIVITY
from .storage import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job",type=Path)
    parser.add_argument("--output",type=Path,default=Path("results/builds/reproducible-example"))
    parser.add_argument("--layers",type=int,default=3)
    parser.add_argument("--all-materials",action="store_true")
    parser.add_argument("--run",action="store_true")
    args=parser.parse_args()
    job=load_job(args.job,progress=lambda fraction,message:print(message,flush=True) if fraction==1 else None)
    if args.output.exists() and any(args.output.iterdir()):
        parser.error("Choose an empty output directory to preserve existing evidence")
    save_imported_job(job,args.output/"job")
    cfg=BuildConfig(alloy="17-4PH",substrate="17-4PH",absorptivity=.35,
        spacing_m=.0015,z_spacing_m=.00024,layer_start=0,layer_end=min(args.layers,len(job.layers)),
        recoat_time_s=3.,final_cooldown_s=.5,max_frames=35,frame_interval_s=.2,
        support_sections=(16,) if 16 in {int(p) for layer in job.layers for p in layer.process_ids} else ())
    request={"kind":"transfer","job_path":"job/job-summary.json","build_config":asdict(cfg),
        "options":{"baseline_material":"17-4PH","target_materials":["316L","Ti-6Al-4V","AlSi10Mg"] if args.all_materials else ["316L"],
            "absorptivities":ILLUSTRATIVE_ABSORPTIVITY,"evaluate_candidates":True,"power_scales":[1.],"speed_scales":[.9,1.,1.1]},
        "input_provenance":{"recoating":{"value_s":3.,"source":"User approximate estimate, including recoating"},
            "other_machine_inputs":"ILLUSTRATIVE ASSUMPTIONS; not a printer recommendation"}}
    path=args.output/"request.json";atomic_json(path,request)
    print(path.resolve(),flush=True)
    if args.run:
        subprocess.run([sys.executable,"-m","heat_sim.worker",str(path.resolve())],check=True)


if __name__=="__main__":main()
