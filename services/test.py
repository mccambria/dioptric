# from utils import common
# from utils import tool_belt as tb
# config = common.get_config_dict()

# collection_mode = config[
#     "collection_mode_counter"
# ]  ## TODO: change to "collection_mode_counter" this line when set up in new computer
# collection_mode_str = collection_mode.name.lower()
# print(collection_mode_str)
# repo_path = common.get_repo_path()
# sequence_library_path = (
#     repo_path
#     / f"servers/timing/sequencelibrary/{self.name}/{collection_mode_str}"
# )

from pathlib import Path
from utils import common

config = common.get_config_dict()
collection_mode = config["collection_mode_counter"]
collection_mode_str = collection_mode.name.lower()

repo_path = common.get_repo_path()
name = "pulse_gen_SWAB_82"  # or PulseGenSwab82.name
sequence_library_path = (
    repo_path / f"servers/timing/sequencelibrary/{name}/{collection_mode_str}"
)

print("collection_mode:", collection_mode)
print("collection_mode_str:", collection_mode_str)
print("sequence_library_path:", sequence_library_path.resolve())
print("exists:", sequence_library_path.exists())
