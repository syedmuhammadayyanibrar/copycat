import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from copycat.data.generator import SyntheticDatasetGenerator

def main():
    dev_dir = os.path.join("data", "dev")
    held_out_dir = os.path.join("data", "held_out")
    os.makedirs(dev_dir, exist_ok=True)
    os.makedirs(held_out_dir, exist_ok=True)
    
    generator = SyntheticDatasetGenerator(seed=2026)
    bundle = generator.generate_bundle(
        n_creators=25,
        n_train=60,
        n_similar=25,
        n_novel=25,
    )
    
    dev_creators = {f"creator_{i:02d}": bundle.creators[f"creator_{i:02d}"] for i in range(20)}
    held_out_creators = {f"creator_{i:02d}": bundle.creators[f"creator_{i:02d}"] for i in range(20, 25)}
    
    dev_creator_ids = set(dev_creators.keys())
    held_out_creator_ids = set(held_out_creators.keys())
    
    dev_train = bundle.train_df[bundle.train_df["creator_id"].isin(dev_creator_ids)]
    dev_similar = bundle.similar_df[bundle.similar_df["creator_id"].isin(dev_creator_ids)]
    dev_novel = bundle.novel_df[bundle.novel_df["creator_id"].isin(dev_creator_ids)]
    
    dev_train.to_csv(os.path.join(dev_dir, "train.csv"), index=False)
    dev_similar.to_csv(os.path.join(dev_dir, "held_out_similar.csv"), index=False)
    dev_novel.to_csv(os.path.join(dev_dir, "held_out_novel.csv"), index=False)
    
    with open(os.path.join(dev_dir, "creators.json"), "w") as f:
        json.dump({k: v.to_dict() for k, v in dev_creators.items()}, f, indent=2)
        
    held_train = bundle.train_df[bundle.train_df["creator_id"].isin(held_out_creator_ids)]
    held_similar = bundle.similar_df[bundle.similar_df["creator_id"].isin(held_out_creator_ids)]
    held_novel = bundle.novel_df[bundle.novel_df["creator_id"].isin(held_out_creator_ids)]
    
    held_train.to_csv(os.path.join(held_out_dir, "train.csv"), index=False)
    held_similar.to_csv(os.path.join(held_out_dir, "held_out_similar.csv"), index=False)
    held_novel.to_csv(os.path.join(held_out_dir, "held_out_novel.csv"), index=False)
    
    with open(os.path.join(held_out_dir, "creators.json"), "w") as f:
        json.dump({k: v.to_dict() for k, v in held_out_creators.items()}, f, indent=2)
        
    print(f"Generated dev data: {len(dev_creators)} creators, {len(dev_train)} train, {len(dev_similar)} sim, {len(dev_novel)} nov")
    print(f"Generated held-out data: {len(held_out_creators)} creators, {len(held_train)} train, {len(held_similar)} sim, {len(held_novel)} nov")

if __name__ == "__main__":
    main()
