from pathlib import Path

from boxsdk import BoxException, Client, JWTAuth

root_folder_id = "235146666549"

data_manager_folder = Path.cwd()

try:
    box_auth_file_name = "dioptric_box_authorization.json"
    box_auth = JWTAuth.from_settings_file(data_manager_folder / box_auth_file_name)
    box_client = Client(box_auth)
except Exception as exc:
    print(
        "\n"
        f"Make sure you have the Box authorization file for dioptric in your "
        f"checkout of the GitHub repo. It should live here: {data_manager_folder}. "
        f"Create the folder if it doesn't exist yet. The file, {box_auth_file_name}, "
        f"can be found in the nvdata folder of the Kolkowitz group Box account."
        "\n"
    )
    raise exc


def reauth_box():
    try:
        box_auth_file_name = "dioptric_box_authorization.json"
        box_auth = JWTAuth.from_settings_file(data_manager_folder / box_auth_file_name)
        global box_client
        box_client = Client(box_auth)
    except Exception as exc:
        print(
            "\n"
            f"Make sure you have the Box authorization file for dioptric in your "
            f"checkout of the GitHub repo. It should live here: {data_manager_folder}. "
            f"Create the folder if it doesn't exist yet. The file, {box_auth_file_name}, "
            f"can be found in the nvdata folder of the Kolkowitz group Box account."
            "\n"
        )
        raise exc


def main():
    # items = box_client.collection(root_folder_id).get_items()
    folder = box_client.folder(root_folder_id)
    items = folder.get_items()
    folder_info = folder.get()
    folder_name = folder_info.name.split(".")[0]
    curr_path = Path("G:\\")
    recurse_download(root_folder_id, curr_path)


def recurse_download(folder_id, current_path):
    try:
        start_folder = box_client.folder(folder_id)
        items = start_folder.get_items()
        folder_info = start_folder.get()
        folder_name = folder_info.name.split(".")[0]
        new_path = current_path / folder_name
        new_path.mkdir()
    except Exception as exc:
        reauth_box()
        start_folder = box_client.folder(folder_id)
        items = start_folder.get_items()
        folder_info = start_folder.get()
        folder_name = folder_info.name.split(".")[0]
        new_path = current_path / folder_name
        new_path.mkdir()

    for item in items:
        if item.type == "folder":
            # print("recursing")
            recurse_download(item.id, new_path)
        elif item.type == "file":
            try:
                box_file = box_client.file(item.id)
                file_content = box_file.content()

            except Exception as exec:
                reauth_box()
                box_file = box_client.file(item.id)
                file_content = box_file.content()

            file_path = new_path / item.get().name
            with file_path.open("wb+") as f:
                f.write(file_content)


if __name__ == "__main__":
    main()
