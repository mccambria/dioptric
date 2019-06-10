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

from pyshellout import get, out, confirm
import time

input_branches = []

while True:
    result = input('Enter a branch to archive or blank to proceed: ')
    if result == '':
        break
    input_branches.append(result)

# Get fully merged branches
merged_branches = [line.strip() for line in
                   get(r'git branch --merged master').n]
merged_branches = [branch for branch in merged_branches
                   if branch != '' and not '*' in branch and not branch == 'master']

# Get all local branches
local_branches = [line.strip() for line in
                   get(r'git branch -l').n]
local_branches  = [branch for branch in local_branches
                   if branch != '' and not '*' in branch and not branch == 'master']

# Confirm that the branch should be deleted
archived_branches = []
tagged_branches = []
for branch in input_branches:
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
    if confirm(msg, branch):
        # Add a timestamp to the tagged branch
        inst = int(time.time())
        tagged_name = '{}-{}'.format(branch, inst)
        out('git tag archive/{} {}', tagged_name, branch)
        archived_branches.append(branch)
        tagged_branches.append(tagged_name)
        
if archived_branches == []:
    exit('No branches archived.')

# Push archive tags to remote
for branch in tagged_branches:
    out('git push origin archive/{}', branch)

# Delete remote branches
for branch in archived_branches:
    out('git push origin :{}', branch)

# Delete local branches
for branch in archived_branches:
    out('git branch -D {}', branch)
    
input('Press enter to close.')
