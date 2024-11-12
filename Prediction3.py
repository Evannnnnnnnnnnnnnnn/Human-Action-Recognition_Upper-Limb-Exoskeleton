if __name__ == "__main__" :
    print("\033cStarting ...\n") # Clear Terminal

# ----   # Modifiable variables   ----
action_to_idx = {'down': 0, 'grab': 1, 'walk': 2}   # Action to index mapping
root_directory = 'Temporary Data'                   # Directory where temporary folders are stored
prediction_threshold = 2                            # how much prediction we need to activate
# ------------------------------------

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

# If there is no model to load, we stop
model_list = model_exist()
if not model_list : 
    sys.exit("No model to load")
elif len(model_list) == 1 :
    ModelName = model_list[0]
    ModelToLoad_Path = os.path.join('Model to Load',ModelName)
else :
    print("\nPlease chose which model to load :")
    for i,item in enumerate(model_list) : 
        print (f'{i+1}\t:\t{item}')
    try :
        num = int(input("\nModel to load : "))
        if num > len(model_list) or num <= 0 :
            raise ValueError
    except ValueError :
        sys.exit("Incorrect number")
    except KeyboardInterrupt :
        sys.exit('\n\nProgeamme Stopped')
    ModelName = model_list[num-1]
    ModelToLoad_Path = os.path.join('Model to Load',ModelName)


try :
    if not os.listdir(root_directory) :sys.exit('No data to make prediction on, launch GetData.py first')
except FileNotFoundError :
    sys.exit('No data to make prediction on, launch GetData.py first')

idx_to_action = {v: k for k, v in action_to_idx.items()}    # We invert the dictionary to have the action with the index
tracking = []

transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = HAR_Inference_DataSet(root_dir=root_directory, transform=transform)

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
    print(f'\033cProgramme running   ctrl + C to stop\n\nLoading {ModelName}\nUsing {device}\n\n')
    prediction_save = [] # prediction_save[-1] is the newest prediction, and prediction_save[-prediction_threshold] is the oldest saved
    for i in range(prediction_threshold) :
        prediction_save.append('')
    Motor_activation_counter = 0
    sample_num = ''
    first_sample_num = ''
    for action in action_to_idx:
        tracking.append(0) # We create a variable in the list for each action

    while True:
        while sample_num == dataset.SampleNumber :
            time.sleep(0.001)
            dataset = HAR_Inference_DataSet(root_dir=root_directory, transform=transform)
        sample_num = dataset.SampleNumber
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        with torch.no_grad():
            for video_frames, imu_data in loader:
                video_frames, imu_data = video_frames.to(device), imu_data.to(device)
                outputs = model(video_frames, imu_data)
                prediction = torch.argmax(model(video_frames, imu_data))
                tracking[prediction] += 1

        if first_sample_num == '' : first_sample_num = sample_num



        for i in range(prediction_threshold,1,-1) :
            prediction_save[-i] = prediction_save[-i+1]
        prediction_save[-1] = idx_to_action.get(prediction.item())

        print(LINE_UP, end=LINE_CLEAR)
        if all_the_same(prediction_save) :
            Motor_activation_counter += 1


            if prediction_save[-1] == 'Grab' :
                last_action = 'Grab'
                print(f'Action {Motor_activation_counter} is {last_action}\n')

                # Met l'action moteur grab ici


            elif prediction_save[-1] == 'Down':
                last_action = 'Down'
                print(f'Action {Motor_activation_counter} is {last_action}\n')

                # Met l'action moteur down ici



            else :
                print(f'Action {Motor_activation_counter} is Walk\n')

                # Si il y a une action moteur pour walk, tu la mets ici (bloquer le torque par exemple)




        else :
            print(f'Ignoring {sample_num} : {prediction_save[-1]}')


except KeyboardInterrupt:
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



print('\nProgramme Stopped\n')
