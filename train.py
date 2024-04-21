"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""

from torchvision.datasets import Cityscapes
import torchvision.transforms.v2 as transforms
import torch.optim.lr_scheduler as lr_scheduler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR

#import segmentation_models_pytorch as smp

import wandb

from model import Model
from argparse import ArgumentParser
import utils

#import albumentations as A
#import cv2
#from albumentations.pytorch import ToTensorV2

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser
    
class EarlyStopper:
    def __init__(self, epoch_start=10, diff_patience=10, diff_min_delta=0.2, diff_lim = 0.35, val_patience=10, val_min_delta=0.1, val_lim = 0.3):
        self.epoch_start = epoch_start
        self.diff_patience = diff_patience
        self.diff_min_delta = diff_min_delta
        self.diff_lim = diff_lim
        self.val_patience = val_patience
        self.val_min_delta = val_min_delta
        self.val_lim = val_lim
        self.counter_val = 0
        self.counter_diff = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, epoch, running_loss, validation_loss):
        # Activate early stopping after #num epochs, when the validation and training errors are ~ same. 
        if epoch > self.epoch_start:
            # Check if the current validation error is better, if true save it as new min_validation
            if validation_loss < self.min_validation_loss:
                self.min_validation_loss = validation_loss
                self.counter_val = 0
                # Check if validation loss is below certain threshold and stop if true
                if self.min_validation_loss < self.val_lim:
                    print("The validation loss is below the threshold.\n")
                    return True
            # Check if the validation does not improve over epochs, and stop if true
            elif validation_loss > (self.min_validation_loss + self.val_min_delta):
                self.counter_val += 1
                if self.counter_val >= self.val_patience:
                    print("The validation loss is not decreasing for multiple epochs.\n")
                    return True
                
            # Check if the training error is greater than validation error, if true reset    
            if running_loss > self.min_validation_loss:
                self.counter_diff = 0
            # Check if the training error differce from validation by certain limit, and stop if true 
            elif running_loss < (self.min_validation_loss - self.diff_lim):
                print("The validation loss is too far away from validation error.\n")
                return True
            # Check if the difference betweem the training error and validation error does not decrease, and stop if true
            elif running_loss < (self.min_validation_loss - self.diff_min_delta):
                self.counter_diff += 1
                if self.counter_diff >= self.diff_patience:
                    print("The validation loss is far away from validation error for multiple epochs.\n")
                    return True 
        return False

'''
class ImageDataset:
    def __init__(self, images_filepaths, transform=None):
        self.images_filepaths = images_filepaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform is not None:
            image = self.transform(image=image)["image"]
        return image
'''

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
        
        
def main(args):
    """define your model, trainingsloop optimitzer etc. here"""
    
    '''
    train_transform = A.Compose([
        A.RandomResizedCrop(224,224),
        A.HorizontalFlip(p=0.5),
        #A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=False, p=0.5),
        A.RandomBrightnessContrast (p=0.5),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), always_apply=False, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=0.5),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    '''
    size = 256
    transform_train = transforms.Compose([
    transforms.Resize((size, size*2), interpolation=transforms.InterpolationMode.LANCZOS),
    #transforms.ColorJitter(contrast=0.5),
    #transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), 
    ])
    
    target_transforms = transforms.Compose([
        transforms.Resize((size, size*2)),
        transforms.ToTensor(),
    ])
    # Data loading and applying transformation
    dataset_train = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transform=transform_train, target_transform=target_transforms)
    #dataset_val = Cityscapes(args.data_path, split='val', mode='fine', target_type='semantic', transform=transform_val, target_transform=target_transforms)
    #dataset_train = ImageDataset(images_filepaths=args.data_path, transform=train_transform)

    # Split training set into training and validation sets
    split = 0.8
    boundary = round(split*round(len(dataset_train)))
    train_dataset = torch.utils.data.Subset(dataset_train, range(boundary))
    val_dataset = torch.utils.data.Subset(dataset_train, range(boundary, len(dataset_train)))
    
    # Create data loaders
    trainloader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=False)
    validationloader = DataLoader(val_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=False)
    
    # Set W&B access
    #wandb.login(key='85a151911d232f1466f01b9e6b10930c70bee464',relogin=True)
    #wandb.init(
    #    project="CNN_ctyscp_U-net",
    #)

    # Set the device 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # I no gpu, use cpu

    #model = Model(num_classes=19, size_img=size).cuda()
    model = Model(block=ResidualBlock, blocks = [2, 2, 2, 2], in_channels = 3, classes = 19, power = 6).to(device)
    #model = smp.Unet(encoder_name='efficientnet-b1', in_channels=3, classes=19, activation=None).to(DEVICE)
    #model = Model(in_channels=3, num_classes=19, initial_power=6).cuda()
    
    # Initialize the model
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            # torch.nn.init.constant_(m.weight, 1) # Sets tensor m.weights to value of 1
            torch.nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))  # Uses xavier weights initialization (Var[s] = Var[x], since Var[w] = 1/n)
            if m.bias is not None:
                m.bias.data.zero_()
        if isinstance(m, nn.Linear):
            torch.nn.init.constant_(m.weight, 1) 
            if m.bias is not None:
                m.bias.data.zero_()
    # Apply the weights and biases before training
    model.apply(init_weights)
    # Set epochs and learning rate
    num_epochs = 110
    lr = 0.01 # 0.01 for 20
    step_size = 35
    # define optimizer and loss function (don't forget to ignore class index 255)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr) #optim.SGD(model.parameters(), lr=lr, momentum=0.5)
    # Set scheduler to change the learning rate over number of epochs
    scheduler = StepLR(optimizer, step_size=step_size, gamma=0.1)
    # Early stopping
    early_stopper = EarlyStopper(epoch_start=10, diff_patience=10, diff_min_delta=0.5, diff_lim = 0.5, val_patience=10, val_min_delta=1, val_lim = 0.3)
    
    # training/validation loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        val_loss = 0.0
        for i, data in enumerate(trainloader):
            inputs, labels = data[0].to(device), (data[1]*255).to(device).long()
            labels = utils.map_id_to_train_id(labels)#.to(device) 
            #labels = labels.argmax(dim=1)
            labels=labels.squeeze(1)
            optimizer.zero_grad()

            #print(labels.unique())
            #print(labels.shape)
            optimizer.zero_grad()

            outputs = model(inputs)
            labels = labels.squeeze(1)

            loss = criterion(outputs, labels)
            v=epoch + 1
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            #wandb.log({'loss': running_loss/(i+1)})
            #wandb.log({'train_Iteration': i})
            #wandb.log({'Train_Epoch':float(v)})
            
            #print(f'Epoch {epoch + 1}, Iteration [{i}/{len(trainloader)}], Loss: {running_loss/(i+1)}')
        
        with torch.no_grad():
            model.eval()
            
            for i, data in enumerate(validationloader):
                inputs, labels = data[0].to(device), (data[1]*255).to(device).long()
                labels = utils.map_id_to_train_id(labels)#.to(device)
                #labels = labels.argmax(dim=1)
                labels=labels.squeeze(1)
                outputs = model(inputs)

                loss = criterion(outputs, labels)
                v=epoch + 1
                #wandb.log({'loss': running_loss,'Epoch':float(v)})

                val_loss += loss.item()
                #wandb.log({'test_loss': val_loss/(i+1)})
                #wandb.log({'test_Iteration': i})
                #wandb.log({'TEST_Epoch':float(v)})

                #print(f'TEST: Epoch {epoch + 1}, Iteration [{i}/{len(validationloader)}], Loss: {val_loss/(i+1)}')

        # One epoch finished, display the summary
        #clear_output()
        print(f'Finished Train epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}')
        print(f'Finished TEST epoch [{epoch + 1}/{num_epochs}], Loss: {val_loss / len(validationloader):.4f}')
        # Check if stopping is required
        if early_stopper.early_stop(epoch, running_loss, val_loss):             
            break
        # Change the learning rate if certain number of epochs is achieved
        if (epoch < 65):
          scheduler.step()
        if (v%step_size == 0):
          print(scheduler.get_last_lr(), " New Learning Rate")

    # save model
    torch.save(model.state_dict(), '/gpfs/home3/scur0778/FinalAssignment/submit/stored_models/ResNmodel_notEff_Adam_nodp-6-ESnew_085_256-8-8_001-35-01.pth')
    #model.load_state_dict(torch.load(PATH))

    # visualize some results
    pass


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
