import json
import os
import dataclasses
from enum import Enum
from utils.benchmark_generator import BenchmarkGenerator, GeneratorConfig, InstanceSize

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, set):
            return list(o)
        if isinstance(o, Enum):
            return o.value
        # Handle dict keys that are integers (JSON requires string keys)
        if isinstance(o, dict):
            return {str(k): v for k, v in o.items()}
        return super().default(o)

def convert_dict_keys_to_str(obj):
    if isinstance(obj, dict):
        return {str(k): convert_dict_keys_to_str(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_dict_keys_to_str(item) for item in obj]
    elif dataclasses.is_dataclass(obj):
        d = dataclasses.asdict(obj)
        return convert_dict_keys_to_str(d)
    elif isinstance(obj, set):
        return list(obj)
    elif isinstance(obj, Enum):
        return obj.value
    else:
        return obj

def generate_large_benchmarks():
    os.makedirs("benchmarks/large", exist_ok=True)
    
    for i in range(1, 4):
        instance_id = f"SFJSSP_LARGE_{i:03d}"
        print(f"Generating {instance_id}...")
        
        config = GeneratorConfig(
            instance_id=instance_id, 
            size=InstanceSize.LARGE, 
            n_jobs=200, 
            n_machines=20, 
            n_workers=20, 
            seed=42 + i
        )
        
        gen = BenchmarkGenerator(config)
        instance = gen.generate()
        
        output_path = f"benchmarks/large/{instance_id}.json"
        
        # We use convert_dict_keys_to_str to ensure all int keys in dicts are strings,
        # sets are lists, and dataclasses are dicts.
        data = convert_dict_keys_to_str(instance)
        
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved {instance_id} to {output_path}")

if __name__ == "__main__":
    generate_large_benchmarks()