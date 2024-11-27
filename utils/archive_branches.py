# -*- coding: utf-8 -*-
"""Script for archiving branches. A branch should be archived if you're
done working with it so that it doesn't clutter the repo. Generally we want
a record of our work so we don't want to delete branches outright.

Archiving is done by tagging them as archive/<branchname>-<unix_timestamp> and
deleting them both locally and remotely.

Created on June 10th, 2019

@author: mccambria
"""

import time
from pathlib import Path

from git import Repo


def _parse_branch_response(response):
    """Raw git commands return strings that we need to parse.
    The form is [* <current_branch>, <branch>, <branch>, ...]
    """
    branches = response.split("\n")
    branches = [el.strip() for el in branches]
    return branches


def main(repo_path, branches_to_archive, skip_merged_check=False):
    # Get the repo and the remote origin
    repo = Repo(repo_path)
    repo_git = repo.git
    origin = repo.remotes.origin

    # Get fully merged branches
    merged_branches = _parse_branch_response(repo_git.branch("--merged", "master"))
    merged_branches = [
        branch
        for branch in merged_branches
        if branch != "" and "*" not in branch and not branch == "master"
    ]
    print("Merged branches:")
    print(merged_branches)

    # Get all local branches
    local_branches = _parse_branch_response(repo_git.branch("-l"))
    local_branches = [
        branch
        for branch in local_branches
        if branch != "" and "*" not in branch and not branch == "master"
    ]
    print("\nLocal branches:")
    print(local_branches)

    print()

    for branch in branches_to_archive:
        do_archive = True
        tag_created = False
        tag_pushed = False
        remote_deleted = False
        local_deleted = False
        if branch == "master":
            print("I'm sorry Dave. I'm afraid I can't archive the master branch.")
            do_archive = False
        elif branch not in local_branches:
            print(
                f"Branch {branch} does not exist locally for this repo. "
                "Switch to it first before merging."
            )
            do_archive = False
        elif (branch not in merged_branches) and not skip_merged_check:
            print(
                f"Branch {branch} is not fully merged with master. "
                "If you wish to archive anyway, set skip_merged_check to True."
            )
            do_archive = False

        # Move on to the next branch if we can't archive this one
        if not do_archive:
            print(f"Skipping branch {branch}.")
            print()
            continue

        # Create the tag
        inst = int(time.time())  # Add a timestamp to the tagged branch
        tagged_name = f"{branch}-{inst}"
        try:
            repo_git.tag(rf"archive/{tagged_name}", branch)
            tag_created = True
        except Exception as exc:
            print(f"Failed to create tag for branch {branch}")
            print(exc)

        # Push the tag
        if tag_created:
            try:
                origin.push(f"archive/{tagged_name}")
                tag_pushed = True
            except Exception as exc:
                print(f"Failed to push tag {tagged_name}")
                print(exc)

        # Delete remote and local branches
        if tag_created and tag_pushed:
            try:
                origin.push(f":{branch}")
                remote_deleted = True
            except Exception as exc:
                print(f"Failed to delete remote branch {branch}")
                print(exc)
            try:
                repo_git.branch("-D", branch)
                local_deleted = True
            except Exception as exc:
                print(f"Failed to delete local branch {branch}")
                print(exc)

        if tag_created and tag_pushed and remote_deleted and local_deleted:
            print(f"Branch {branch} successfully archived.")
        print()


if __name__ == "__main__":
    # Path to your local checkout of the repo
    repo_path = Path.home() / "Documents/GitHub/dioptric"

    # List of branches to archive
    branches_to_archive = [
        "branch",
    ]

    skip_merged_check = True

    main(repo_path, branches_to_archive, skip_merged_check)
