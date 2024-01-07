import re
import os

"""
Directory structure is:

batch_number
     |-- files_PDK_batch_larcv (and log)
        |-- data
           |-- files_PDK_batch_larcv_grp{number from 0 to 35}
              |-- data
                 |-- files_PDK_batch_larcv_{number from 0 to 19}
                    |-- data
                       |-- file for plane0
                       |-- file for plane1
                       |-- file for plane2
"""

init_dir = '/mnt/lustre/helios-home/gartmann/tar_data/signal/pdk/PDK'
batch_range = range(0, 124)

for batch in batch_range:
    # Create file for writing the simplified log
    init_log = os.path.join(init_dir, f'{batch}', 'evtinfo.log')
    simplified_log = os.path.join(init_dir, f'{batch}', 'evtinfo_reduced.log')

    if not os.path.exists(simplified_log):
        with open(simplified_log, 'w') as f:
            f.write('')

    with open(init_log, 'r') as f:
        lines = f.readlines()

        # Until the line before the last line
        for line in lines[:-1]:
            if line.startswith('       %MSG-w'):
                continue
            # If the line starts with 5 spaces followed by a number with up to 4 digits, write to simplified log number only
            elif re.match(r'     \d', line) or re.match(r'    \d\d', line) or re.match(r'   \d\d\d', line):
                
                # The number is bracketed by space from the left and tab from the right
                number = re.search(r' (\d+)	', line).group(1)
                next_line = lines[lines.index(line) + 1]
                if re.match(r'       K', next_line):
                    
                    # Write number, space, line with K decay mode
                    with open(simplified_log, 'a') as f:
                        f.write(number + ' ' + next_line[7:])
                else:
                    # Write number only
                    with open(simplified_log, 'a') as f:
                        f.write(number + '\n')
                        
