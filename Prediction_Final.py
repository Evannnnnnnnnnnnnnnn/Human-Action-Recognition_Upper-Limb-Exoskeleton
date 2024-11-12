if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

# ----   # Modifiable variables   ----
action_to_idx = {'down': 0, 'grab': 1, 'walk': 2}   # Action to index mapping
root_directory = 'Temporary Data'                   # Directory where temporary folders are stored
prediction_threshold = 3                            # how much prediction we need to activate
# ------------------------------------

#TODO testing to see if it works

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
    from Imports.Functions import model_exist, all_the_same
    from Imports.Models.MoViNet.config import _C as config
    from Imports.Models.fusion import FusionModel
except ModuleNotFoundError :
    sys.exit('Missing Import folder, make sure you are in the right directory')




LINE_UP = '\033[1A'
LINE_CLEAR = '\x1b[2K'

def make_prediction(Dataset) -> int :
    Loader = DataLoader(Dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
    with torch.no_grad():
        for video_frames, imu_data in Loader:
            video_frames, imu_data = video_frames.to(device), imu_data.to(device)
            predicted = torch.argmax(model(video_frames, imu_data))
    return predicted.item()

if not model_exist() :
    sys.exit("No model to load") # If there is no model to load, we stop

try :
    if not os.listdir(root_directory) : raise FileNotFoundError # If there is a folder but it's empty, we raise an error
except FileNotFoundError :
    sys.exit('No data to make prediction on, launch GetData.py first')

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
idx_to_action = {v: k for k, v in action_to_idx.items()}    # We invert the dictionary to have the action with the index
tracking = []

model = FusionModel(config.MODEL.MoViNetA0, num_classes=3, lstm_input_size=12, lstm_hidden_size=512, lstm_num_layers=2)
model.load_state_dict(torch.load(ModelToLoad_Path, weights_only = True, map_location=device))
model.to(device)
model.eval()

try : # Main Loop
    print(f'\033cProgramme running   ctrl + C to stop\n\nLoading {ModelName}\nUsing {device}\n\n')
    sample_num = ''
    first_sample_num = ''
    Motor_activation_counter = 0

    for action in action_to_idx:
        tracking.append(0) # We create a variable in the list for each action

    prediction_save = [] # prediction_save[-1] is the newest prediction, and prediction_save[-prediction_threshold] is the oldest saved
    for i in range(prediction_threshold) :
        prediction_save.append('')
    
    while True:
        while sample_num == dataset.SampleNumber : # We check for new sample every millisecond
            time.sleep(0.001)
            dataset = HAR_Inference_DataSet(root_dir=root_directory, transform=transform)
        sample_num = dataset.SampleNumber
        if first_sample_num == '' : first_sample_num = sample_num # We get the number of the first sample

        try :
            prediction = make_prediction(dataset)
        except :
            if sample_num == first_sample_num : print(LINE_UP, end=LINE_CLEAR)
            print(f'Error on {sample_num}\n')
        tracking[prediction] += 1

        for i in range(prediction_threshold,1,-1) :
            prediction_save[-i] = prediction_save[-i+1]
        prediction_save[-1] = idx_to_action.get(prediction.item())

        if all_the_same(prediction_save)[0] :
            Motor_activation_counter += 1


            if prediction_save[-1] == 'Grab' and last_action != 'Grab' :
                last_action = 'Grab'
                print(f'Action {Motor_activation_counter} is {last_action}\n')

                # Motor action


            elif prediction_save[-1] == 'Down' and last_action != 'Down':
                last_action = 'Down'
                print(f'Action {Motor_activation_counter} is {last_action}\n')

                # Motor action



            else :
                print(f'Action {Motor_activation_counter} is Walk\n')

                # Torque Lock
        else :
            raise NotImplementedError #TODO print that there were n similar action an the action, or if none then none


        print(LINE_UP, end=LINE_CLEAR)
        print(f'{sample_num} : {idx_to_action.get(prediction)}')


except KeyboardInterrupt:
    pass

except FileNotFoundError:
    print("Samples folder got deleted")
    
num_of_predictions = 0
for i in tracking :
    num_of_predictions += i
num_first = int(first_sample_num.replace('Sample_',''))
num_last = int(sample_num.replace('Sample_',''))

if num_of_predictions > 1 : end_text = 's'
else : end_text = ''
print(f'\nThere were a total of {num_of_predictions} prediction{end_text}, with {(num_last-num_first+1)-num_of_predictions} missed')
for action, i in action_to_idx.items() :
    print(f'{tracking[i]} for {action}')



if __name__ == "__main__" :
    print('\nProgramme Stopped\n')
