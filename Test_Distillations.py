#Import libraries
import torch
from torch import nn
import torch.nn.functional as F
from fastai.vision.all import *
from uniplot import plot
 
#Sample Tensors Taken from Student Training to validate distillation loss function
def testDistillation(loss_function):
    #Pass in sample tensors to determine whether the forward and backward methods are working syntactically
    passTensors(loss_function)
    #Generate small neural network and backpropogate to determine whether backward is correctly calculating gradients (using FastAI and MNIST)
    mnist_dataset_url = untar_data(URLs.MNIST)
    #Create datablock and dataloader
    mnist_datablock = DataBlock(
            blocks = (ImageBlock, CategoryBlock),
            get_items = get_image_files,
            splitter = RandomSplitter(valid_pct = 0.2, seed = 32),
            get_y = parent_label,
            batch_tfms = aug_transforms(mult = 2.0, do_flip = False))
    training_dataloader = mnist_datablock.dataloaders(mnist_dataset_url/"training", batch_size = 64, shuffle = True)
    #Create test model architecture
    test_model_architecture = nn.Sequential(nn.Flatten(),
                                            nn.Linear(28 * 28 * 3, 500),
                                            nn.ReLU(),
                                            nn.Linear(500, 300),
                                            nn.ReLU(),
                                            nn.Linear(300, 200),
                                            nn.ReLU(),
                                            nn.Linear(200, 10))
    #Create teacher model architecture
    teacher_model_architecture = nn.Sequential(nn.Flatten(),
                                               nn.Linear(28 * 28 * 3, 500),
                                               nn.ReLU(),
                                               nn.Linear(500, 300),
                                               nn.ReLU(),
                                               nn.Linear(300, 200),
                                               nn.ReLU(),
                                               nn.Linear(200, 10))
    #Create learner object
    test_learner = Learner(training_dataloader, test_model_architecture, metrics = ['accuracy', 'error_rate'])
    #Find optimal learning rate
    # optimal_LR = test_learner.lr_find()[0]
    #print(optimal_LR)
    test_learner.opt = Adam(test_learner.parameters(), lr = 0.001)
    #Import teacher model (from CustomMaxout.py where a ReLU model was trained on MNIST as a part of a comparison test)
    #The teacher model has the same architecture as the student
    teacher_learner = Learner(training_dataloader, teacher_model_architecture, metrics = ['accuracy', 'error_rate'])
    teacher_learner.load('TEST_TEACHER')

    #Training loop for test learner - only test on 1 epoch (change as needed)
    for epoch in range(1):
        #Distillation loss
        ds_losses = []
        #Accuracy
        accuracies = []
        for batch_idx, batch_data in enumerate(test_learner.dls.train, 0):
            #Get inputs and labels
            inputs, labels = batch_data
            #Set grad to zero
            test_learner.zero_grad()
            #Generate predictions
            student_preds = test_learner.model(inputs)
            teacher_preds = teacher_learner.model(inputs)
            #Find loss
            loss_func = loss_function.apply
            loss = loss_func(student_preds, teacher_preds, labels)
            #Compute gradients
            loss.backward()
            #Take step
            test_learner.opt.step()
            #Add to batch losses
            ds_losses.append(loss.item())
            #Find maximum predictions (the actual classes predicted by the student)
            true_preds, true_pred_labels = torch.max(student_preds, dim = 1)
            #Compute accuracy - squeeze both tensors
            true_pred_labels.squeeze_()
            labels.squeeze_()
            acc = 100 * torch.eq(torch.tensor(true_pred_labels), torch.tensor(labels)).sum().item() / 64
            #Append for graphing
            accuracies.append(acc)
            #Print metrics for minibatch
            print('MINIBATCH [{}]: DIST LOSS {} ACCURACY {}%'.format(batch_idx, loss, round(acc, 2)))
            if batch_idx == 749:
                plot(ds_losses, title = "LOSSES", color = True, lines = True, legend_labels = ["DIST LOSSES"])
                plot(accuracies, title = "ACCURACY", color = True, lines = True, legend_labels = ["ACCURACY"])
                avg_batch_loss = sum(ds_losses) / len(ds_losses)
                avg_acc = sum(accuracies)/len(accuracies)
                print("TRAINING EPOCH [{}] LOSS: {} ACCURACY: {}".format(epoch, avg_batch_loss, avg_acc))    
                break                                     

def passTensors(loss_function):
    #Initialize sample student, teacher, and true_y tensors with the same sizes as what will be used in the BAN (64 * 20)
    bs = 20
    t1 = torch.rand(bs, 20, requires_grad = True)
    t2 = torch.rand(bs, 20, requires_grad = True)
    t3 = torch.randint(low = 0, high = 19, size = (bs,))
    #Create loss object
    loss_func = loss_function.apply
    #Calculate loss
    loss = loss_func(t1, t2, t3)
    #Calculate gradients from loss
    loss.retain_grad()
    loss.backward()
    print('LOSS: ', loss)
    print('GRADIENTS: ', loss.grad)