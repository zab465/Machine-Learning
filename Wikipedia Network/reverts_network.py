# Module to make a network out of the wikipedia data on Romania
# Uses external module math and datetime for the operations 

from datetime import datetime
from datetime import timedelta

import numpy as np

def read_data(file_path, skipped_lines=1):
    """
    Function reads the data from the file and returns a list of lists.
    It skips the first line by default.
    """
    # initialize the list of all edits
    i = 1
    all_edits = []

    for line in open(file_path, 'r', encoding='utf-8'):
        # skip the first line of the file
        if i <= skipped_lines:
            i += 1
            continue
        # split the line into the title, timestamp, revert, version and user on tab
        title, timestamp, revert, version, user = line.split("\t")

        # strip the title and timestamp of any extra spaces
        timestamp = timestamp.strip()

        # convert the timestamp to datetime object so that dates can be compared
        time = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        title = title.rstrip()
        user = user.replace('\n', "")

        # create a list of the title, timestamp, revert, version and user and append it to the list of all edits
        ls = [title, time, int(revert), int(version), user]
        all_edits.append(ls)

    return all_edits

def create_user_info(all_edits):
    """
    Function returns a dictionary of edit counts and a dictionary user_info_dict 
    which contains the time stamps of each edit for each user.
    Takes for argument the list of all edits.
    """
    # create a list of the time and user for each edit
    time_user_list = [[item[1], item[4]] for item in all_edits]
    
    # initialize the dictionaries to store the edit counts and the time stamps of each edit for each user
    edit_counts = {}
    user_info_dict = {}
    unique_users = set()

    # iterate through the list of time and user for each edit
    for time, user in time_user_list:
        # if the user is not in the users set, then append it to the set & initalize the edit count to 0
        if user not in unique_users:
            unique_users.add(user)
            edit_counts[user] = 0
            # initialize the list of time stamps for each user
            user_info_dict[user] = []

        # increment the edit count for the user
        # append the time stamp to the list of time stamps for the user
        edit_counts[user] += 1
        user_info_dict[user].append(time)

    return edit_counts, user_info_dict

def get_seniority_before_revert(user, revert_time, user_info_dict):
    """
    Function returns the seniority of a user before the given revert_time.
    Calculates the seniority by taking the log10 of the edit count 
    of the user before the revert_time.
    Takes for argument the user, revert time, and the dictionary of user_info_dict.
    """
    # Filter user edits before the revert_time
    user_edits_before_revert = [edit_time for edit_time in user_info_dict[user] if edit_time < revert_time]

    # If the last edit time before the revert is equal to the revert_time, decrement the edit count
    # This will exclude the current revert from the seniority calculation 
    if user_edits_before_revert and user_edits_before_revert[-1] == revert_time:
        user_edits_before_revert.pop()

    # save edit count as the length of the list of user edits before the revert
    edit_count = len(user_edits_before_revert)

    # If the edit count is 0, the seniority is log1 = 0
    if edit_count == 0:
        return 0

    # Calculate the seniority as log10 of the edit count
    seniority = np.log10(edit_count)
    return seniority


def create_revert_network(all_edits, user_info_dict):
    """
    Function returns a list of dictionaries containing the 
    reverter, reverted, time stamp, reverter seniority and reverted seniority.
    This is a network of reverts.
    Takes for argument the list of all edits and the dictionary of user_info_dict.
    Seniority is provided before the revert.
    """
    # List to store revert dictionaries
    network = []
    # Set to keep track of unique users involved in reverts
    unique_users = set()
    # Iterate through all edits in reverse order (since the file is in reverse chronological order)
    for i in range(len(all_edits) - 1, -1, -1):  
        # Check if the edit is a revert using column "revert"
        if all_edits[i][2] == 1:
            # User that "comits" the revert
            reverter_user = all_edits[i][4]
            # Time stamp of the revert is column 1
            time_stamp = all_edits[i][1]
            # Initialize the found_reverted as False and the index j as 0
            found_reverted = False
            j = 0

            # Iterate to find the reverted user
            while found_reverted is False:
                j += 1
                # Check if the version of the next edit is the same as the one where revert == 1
                if i + j < len(all_edits) and all_edits[i][3] == all_edits[i + j][3]:  
                    # If the version is the same, then the user of the next edit is the reverted user
                    reverted_user = all_edits[i + j - 1][4]
                    # Set found_reverted to True
                    found_reverted = True

                    # Check that the reverted user is not the same as the reverter user
                    # We don't accept self-reverts
                    if (
                        reverted_user != reverter_user
                        and all_edits[i][0] == all_edits[i + j][0]
                    ):
                        # If the reverted user is not the same as the reverter user then get the seniority of both users
                        # Call the function get_seniority_before_revert
                        reverter_seniority = get_seniority_before_revert(reverter_user, time_stamp, user_info_dict)
                        reverted_seniority = get_seniority_before_revert(reverted_user, time_stamp, user_info_dict)

                        # Add the reverted and reverter to the unique users set to keep track of all users involved in reverts
                        # Add the reverter, reverted, time stamp, reverter seniority and reverted seniority to the network
                        unique_users.add(reverter_user)
                        unique_users.add(reverted_user)
                        network.append(
                            {
                                "Reverter": reverter_user,
                                "Reverted": reverted_user,
                                "Time stamp": time_stamp,
                                "Reverter Seniority": reverter_seniority,
                                "Reverted Seniority": reverted_seniority,
                            }
                        )
    return unique_users, network


def count_and_label_AB_BA(network):
    """
    Function that counts the number of AB-BA event sequences in the network. 
    Takes a list of dictionaries as input and adds the 'AB-BA' key to edges, indicating 
    whether they are involved in an AB-BA sequence or not.
    Edges can be considered a "response" for multiple sequences, but not as a start node. 
    """
    # Sorting the network by time stamp
    # so that the first revert is always the first revert in the network
    # this is more accurate to calculate the AB-BA sequences because ordering the edges by time stamp will 
    # help keep track of where the edge belongs in the sequence
    network_sorted = network.copy()
    network_sorted.sort(key=lambda x: x['Time stamp'])

    # Starting the counter
    ab_ba_sequences = 0

    # Initialize a set to keep track of edges involved in AB-BA sequences
    ab_ba_edges = set()

    # Iterate through the network 
    for i in range(len(network_sorted)):
        # Flag to track if the current i edge is part of any AB-BA sequence
        found_ab_ba = False
        
        for j in range(i + 1, len(network_sorted)):
            # If time diff > 24 hours, then not AB-BA sequence
            # break loop if time diff > 24 hours
            if network_sorted[j]['Time stamp'] - network_sorted[i]['Time stamp'] > timedelta(hours=24):
                break
            # Check if the reverter of i is the reverted of j and the other way around
            if network_sorted[i]['Reverter'] == network_sorted[j]['Reverted'] and network_sorted[i]['Reverted'] == network_sorted[j]['Reverter']:
                # If this is the first occurrence for the current i edge, increment the counter
                if not found_ab_ba:
                    ab_ba_sequences += 1
                    found_ab_ba = True

                # Add only the current i and j edges to the set of AB-BA edges
                ab_ba_edges.add((network_sorted[i]['Reverter'], network_sorted[i]['Reverted'], network_sorted[i]['Time stamp']))
                ab_ba_edges.add((network_sorted[j]['Reverter'], network_sorted[j]['Reverted'], network_sorted[j]['Time stamp']))

    # Update 'AB-BA' key for the edges directly involved in AB-BA sequences
    for edge in network_sorted:
        edge_tuple = (edge['Reverter'], edge['Reverted'], edge['Time stamp'])
        edge['AB-BA'] = edge_tuple in ab_ba_edges

    return ab_ba_sequences, network_sorted

def get_seniority_diffs(network_sorted):
    # Initialize lists to store the absolute differences in seniority
    ab_ba_diffs = []
    other_diffs = []

    # Iterate over the 'network' list
    for edge in network_sorted:
        # Calculate the absolute difference in seniority using abs
        diff = abs(edge['Reverter Seniority'] - edge['Reverted Seniority'])

        # Check if the revert is part of an AB-BA sequence
        if edge.get('AB-BA', True):
            ab_ba_diffs.append(diff)
        else:
            other_diffs.append(diff)
    
    return ab_ba_diffs, other_diffs