#Import libraries
import torch
from torch import nn
from fastai.vision.all import *
 
#Sample Tensors Taken from Student Training to validate distillation loss function
def testDistillation(loss_function, n_args):
    #Pass in sample tensors to determine whether the forward and backward methods are working syntactically
    passTensors(loss_function, n_args)
    #Generate small neural network and backpropogate to determine whether backward is correctly calculating gradients (using FastAI and MNIST)
    mnist_dataset_url = untar_data(URLs.MNIST)
    #Create datablock and dataloader
    mnist_datablock = DataBlock(
            blocks = (ImageBlock, CategoryBlock),
            get_items = get_image_files,
            splitter = RandomSplitter(valid_pct = 0.2, seed = 32),
            get_y = parent_label,
            batch_tfms = aug_transforms(mult = 2.0, do_flip = False))
    training_dataloader = mnist_datablock.dataloaders(mnist_dataset_url/"training", batch_size = 10, shuffle = True)
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
    test_learner.opt = Adam(test_learner.parameters(), lr = 0.00000001)
    #Import teacher model (from CustomMaxout.py where a ReLU model was trained on MNIST as a part of a comparison test)
    #The teacher model has the same architecture as the student
    teacher_learner = Learner(training_dataloader, teacher_model_architecture, metrics = ['accuracy', 'error_rate'])
    teacher_learner.load('TEST_TEACHER')

    #Training loop for test learner 
    for epoch in range(1):
        batch_losses = []
        for batch_idx, batch_data in enumerate(test_learner.dls.train, 0):
            #Get inputs and labels
            inputs, labels = batch_data
            #Set grad to zero
            test_learner.zero_grad()
            #Generate predictions
            student_preds = test_learner.model(inputs)
            # print('STUDENT TRAINING PREDS: \n', student_preds)
            teacher_preds = teacher_learner.model(inputs)
            # print('TEACHER TRAINING PREDS \n', teacher_preds)
            #Find loss
            loss_func = loss_function.apply
            if n_args == 2:
                loss = loss_func(student_preds, teacher_preds)
            elif n_args == 3:
                loss = loss_func(student_preds, teacher_preds, labels)
            #Compute gradients
            loss.backward()
            #Take step
            test_learner.opt.step()
            #Add to batch losses
            batch_losses.append(loss)
            #If on the final epoch:
            print('MINIBATCH [{}] LOSS: {}'.format(batch_idx, loss))
            if batch_idx == 63:
                avg_batch_loss = sum(batch_losses) / len(batch_losses)
                print("TRAINING EPOCH [{}] LOSS: {}".format(epoch, avg_batch_loss))                                         

def passTensors(loss_function, n_args):
    #Initialize sample student, teacher, and true_y tensors with the same sizes as what will be used in the BAN (64 * 20)
    bs = 20
    t1 = torch.rand(bs, 20, requires_grad = True)
    t2 = torch.rand(bs, 20, requires_grad = True)
    t3 = torch.randint(low = 0, high = 19, size = (bs,))
    # print("TENSOR 1: \n", t1)
    # print("TENSOR 2: \n", t2)
    #Create loss object
    loss_func = loss_function.apply
    #Calculate loss - as this method is also used to test the DKPP loss, check the number of required args
    if n_args == 2: loss = loss_func(t1, t2) 
    else: loss = loss_func(t1, t2, t3)
    #Calculate gradients from loss
    loss.retain_grad()
    loss.backward()
    print('LOSS: ', loss)
    print('GRADIENTS: ', loss.grad)