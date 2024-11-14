import os
import json

def update_json_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # Read the JSON file
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Update the specified keys
                data['nb_repetitions'] = 10
                data['nb_epochs'] = 50
                data['percentage_samples_keep'] = 1
                
                # Write the updated data back to the file
                with open(file_path, 'w') as f:
                    json.dump(data, f, indent=4)
                
                print(f"Updated: {file_path}")

# Specify the directory path
directory_path = './parameters_files/'

# Call the function to update JSON files
update_json_files(directory_path)

print("All JSON files have been updated.")
