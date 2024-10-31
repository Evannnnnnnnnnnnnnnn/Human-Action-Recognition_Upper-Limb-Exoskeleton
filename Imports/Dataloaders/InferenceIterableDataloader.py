if __name__ == "__main__" :
    print("\033c") # Clear Terminal
    import time
    Start_Time = time.time()

try :
    import torch
    import pandas as pd
    from PIL import Image
    from torch.utils.data import Dataset
except ModuleNotFoundError as Err :
    missing_module = str(Err).replace('No module named ','')
    missing_module = missing_module.replace("'",'')
    match missing_module:
        case 'PIL' :
            sys.exit(f'No module named {missing_module} try : pip install pillow')
        case _ :
            sys.exit(f'No module named {missing_module} try : pip install {missing_module}')


testdata = [0, 9, 2, 3, 4, 5, 6, 7, 10, 5, 10, 11, 12, 13, 14, 15, 16]


class HARIterableDataset (IterableDataset):
    def __init__(self, path, sequence_length):
        self.path = path
        self.sequence_length = sequence_length
        self.action_to_idx = {'down': 0, 'grab': 1, 'walk': 2} # Define the mapping from actions to indices



    def get_data_test(self):
        FramesPath = [os.path.join(self.path, f) for f in os.listdir(self.path) if f.endswith('.jpg')]
        FramesPath.sort(key=lambda x: int(re.search(r'\d+', os.path.basename(x)).group()))


        IMUPath = os.path.join(self.path, 'imu.csv')







    def __iter__(self):
        return





MainTestFolder = 'Test Inference'
if os.path.exists(MainTestFolder) :
    Dir = os.listdir(MainTestFolder)
else :
    os.makedirs(MainTestFolder)
    print(f'Please put a piece of the dataset in the "{MainTestFolder}" folder')
    sys.exit(0)
SamplePath = os.path.join(MainTestFolder, Dir[0])


if __name__ == '__main__' :
    print(f'Testing for {SamplePath}')


TestFrames, TestIMU = HARIterableDataset(SamplePath).get_data_test()

print(TestIMU)







