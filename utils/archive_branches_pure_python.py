# -*- coding: utf-8 -*-
"""
This script will archive branches. A branch should be archived if you're
done working with it so that it doesn't clutter the repo. Generally we want
a record of our work so we don't want to delete branches outright.

Archiving is done by tagging them as archive/<branchname> and removing them
both locally and remotely. Before each operation, the user is asked for
confirmation.

Created on Mon Jun 10 08:02:12 2019

@author: mccambria
"""


# %% Imports


import time
from git import Repo


# %% Input parameters

# List of branch names
branches_to_archive = []


# %% Functions


def parse_string_array(string_array):
    pass


# %% Run the file


# Get the repo
repo_path = 'C:\\Users\\kolkowitz\\Documents\\' \
    'GitHub\\kolkowitz-nv-experiment-v1.0'
repo = Repo(repo_path)
repo_git = repo.git


# Get fully merged branches
merged_branches = repo_git.branch('--merged', 'master')
merged_branches = [branch for branch in merged_branches
                   if branch != '' and not '*' in branch and not branch == 'master']
print('Merged branches:')
print(merged_branches)

# Get all local branches
local_branches = [line.strip() for line in
                  repo_git.branch('-l').n]
local_branches  = [branch for branch in local_branches
                   if branch != '' and not '*' in branch and not branch == 'master']
print('\nLocal branches:')
print(local_branches)

# Confirm that the branch should be deleted
archived_branches = []
tagged_branches = []
for branch in branches_to_archive:
    if branch == 'master':
        print("I'm sorry Dave. I'm afraid I can't archive the master branch.")
        continue
    elif branch not in local_branches:
        print('Branch {} does not exist locally for this repo. Skipping.'.format(branch))
        continue
    elif branch not in merged_branches:
        msg = 'Branch {} is not fully merged with master. Archive anyway?'
    else:
        msg = 'Archive branch {}?'
    if input(msg.format(branch)).startswith('y'):
        # Add a timestamp to the tagged branch
        inst = int(time.time())
        tagged_name = '{}-{}'.format(branch, inst)
        repo_git.tag('archive/{} {}', tagged_name, branch)
        archived_branches.append(branch)
        tagged_branches.append(tagged_name)
        
if archived_branches == []:
    exit('No branches archived.')

# Push archive tags to remote
for branch in tagged_branches:
    repo_git.push('origin', 'archive/{}'.format(branch))

# Delete remote branches
for branch in archived_branches:
    repo_git.push('origin', ':{}'.format(branch))

# Delete local branches
for branch in archived_branches:
    repo_git.push('branch', '-D', branch)
