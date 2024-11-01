if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

import os
import sys
import time

try :
    import torch
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
except ModuleNotFoundError as Err:
    missing_module = str(Err).replace('No module named ', '')
    missing_module = missing_module.replace("'", '')
    sys.exit(f'No module named {missing_module} try : pip install {missing_module}')

try :
    from Imports.InferenceDataloader import HAR_Inference_DataSet
    from Imports.Functions import model_exist
    from Imports.Models.MoViNet.config import _C as config
    from Imports.Models.fusion import FusionModel
except ModuleNotFoundError :
    sys.exit('Missing Import folder, make sure you are in the right directory')

# Modifiable variables
action_to_idx = {'down': 0, 'grab': 1, 'walk': 2}   # Action to index mapping
root_directory = 'Temporary Data'                   # Directory where temporary folders are stored
time_for_prediction = 5                             # Time we wait for each prediction
prediction_threshold = 3                            # how much prediction we need to activate

# If there is no model to load, we stop
if not model_exist() :
    sys.exit("No model to load")
try :
    if not os.listdir(root_directory) :sys.exit('No data to make prediction on, launch GetData.py first')
except FileNotFoundError :
    sys.exit('No data to make prediction on, launch GetData.py first')

idx_to_action = {v: k for k, v in action_to_idx.items()}    # We invert the dictionary to have the action with the index
tracking = []

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = HAR_Inference_DataSet(root_dir=root_directory, transform=transform)

ModelToLoad_Path = os.path.join('Model to Load',os.listdir('./Model to Load')[0])
ModelName = os.listdir('./Model to Load')[0]
if ModelName.endswith('.pt') :
    ModelName = ModelName.replace('.pt','')
else :
    ModelName = ModelName.replace('.pht','')
print(f"Loading {ModelName}")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}\n")
LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

model = FusionModel(config.MODEL.MoViNetA0, num_classes=3, lstm_input_size=12, lstm_hidden_size=512, lstm_num_layers=2)
model.load_state_dict(torch.load(ModelToLoad_Path, weights_only = True, map_location=device))
model.to(device)
model.eval()

try :
    for action in action_to_idx:
        tracking.append(0) # We create a variable in the list for each action
    old_sample = ''
    first_sample = ''
    last_action = 'Down'    # So we cannot start with down
    Motor_activation_counter = 0
    while True:
        walk_counter = 0
        grab_counter = 0
        down_counter = 0
        Motor_activation_counter += 1
        print('')
        Start_Time = time.time()
        Current_Time = 0
        while Current_Time - Start_Time < time_for_prediction :
            print(LINE_UP, end=LINE_CLEAR)

            while old_sample == dataset.SampleNumber :
                time.sleep(0.001)
                dataset = HAR_Inference_DataSet(root_dir=root_directory, transform=transform)
            old_sample = dataset.SampleNumber
            if first_sample == '' : first_sample = old_sample

            loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
            with torch.no_grad():
                for video_frames, imu_data in loader:
                    video_frames, imu_data = video_frames.to(device), imu_data.to(device)
                    outputs = model(video_frames, imu_data)
                    predicted = torch.argmax(model(video_frames, imu_data)).item()
                    tracking[predicted] += 1

            if predicted == 0 :
                down_counter += 1
            elif predicted == 1:
                grab_counter += 1
            elif predicted == 2:
                walk_counter += 1
            else :
                sys.exit('Error in Prediction, Predicted value out of range')

            Current_Time = time.time()
            print (f'walk : {walk_counter},  grab : {grab_counter},  down : {down_counter}')
        print(LINE_UP, end=LINE_CLEAR)


        if grab_counter > prediction_threshold and last_action != 'Grab' :
            last_action = 'Grab'
            print(f'Action {Motor_activation_counter} is {last_action}')

            # Met l'action moteur grab ici


        elif down_counter > prediction_threshold and last_action != 'Down':
            last_action = 'Down'
            print(f'Action {Motor_activation_counter} is {last_action}')

            # Met l'action moteur down ici



        else :
            print(f'Action {Motor_activation_counter} is Walk')

            # Si il y a une action moteur pour walk, tu la mets ici (bloquer le torque par exemple)







except KeyboardInterrupt:
    num_of_predictions = 0
    for i in tracking :
        num_of_predictions += i
    num_first = int(first_sample.replace('Sample_',''))
    num_last = int(old_sample.replace('Sample_',''))

    if num_of_predictions > 1 : end_text = 's'
    else : end_text = ''
    print(f'\nThere were a total of {num_of_predictions} prediction{end_text}, with {(num_last-num_first+1)-num_of_predictions} missed')
    for action, i in action_to_idx.items() :
        print(f'{tracking[i]} for {action}')
except FileNotFoundError:
    print("Samples folder got deleted")
    num_of_predictions = 0
    for i in tracking :
        num_of_predictions += i
    num_first = int(first_sample.replace('Sample_',''))
    num_last = int(old_sample.replace('Sample_',''))

    if num_of_predictions > 1 : end_text = 's'
    else : end_text = ''
    print(f'\nThere were a total of {num_of_predictions} prediction{end_text}, with {(num_last-num_first+1)-num_of_predictions} missed')
    for action, i in action_to_idx.items() :
        print(f'{tracking[i]} for {action}')



print('\nProgramme Stopped\n')