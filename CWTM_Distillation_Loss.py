import torch
import torch.nn as nn
from torch.autograd import Function
from fastai.vision.all import *

#CWTM (Confidence Weighted by Teacher Max) Distillation Loss - subclass of torch.autograd.Function, all methods are static
class CWTM_DistillationLoss(Function):

    #Define forward method (where we compute the loss) - take in the student predictions, teacher predictions, and true predictions for each model
    @staticmethod
    def forward(ctx, s_preds, t_preds, true_preds):
        #Save both prediction tensors into context object for gradient computations
        #We want to save the NORMALIZED (softmax activated) versions of each tensor as opposed to the raw probabilities - normalize the values first, and then save
        softmax = torch.nn.Softmax(dim = 1) #Perform across each row (each row sums to 1)
        #The true predictions are already encoded and do not requre softmax
        s_soft_preds = softmax(s_preds)
        t_soft_preds = softmax(t_preds)
        #True predictions are already encoded; they do not require softmax
        #Perform the softmax activation in the process
        ctx.save_for_backward(s_soft_preds, t_soft_preds, true_preds)
        #Use Cross Entropy Loss - PyTorch does NOT support cross entropy between two probability distribution (and instead requires labels), so we must re-implement it
        #We could alternatively use Kulback Leibler loss 
        #Implement (- \sum(q(x) * log(p(x)))) where q(x) is the predicted (student) distribution and p(x) is the "true" (teacher) distribution
        #Take the sum for each row, and then find the mean of all the column sums - add small epsillon to prevent exploding gradients (divergence)
        loss = - torch.sum(torch.mul(t_soft_preds, torch.log(s_soft_preds))).mean() + 10e-10
        # print('STUDENT PREDICTIONS: \n', s_preds)
        # print('SOFTMAX STUDENT PREDICTIONS: \n', s_soft_preds)
        # print('TEACHER PREDICTIONS: \n', t_preds)
        # print('SOFTMAX TEACHER PREDICTIONS: \n', t_soft_preds)
        # print('TRUE PREDICTIONS: \n', true_preds)
        return loss

    #Define backward method (where the gradient of the loss is computed)
    @staticmethod
    def backward(ctx, grad_output):
        #Implement Equation 10 from the BAN paper https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf
        #Obtain labels from the saved tensors
        s_smax_preds, t_smax_preds, true_preds = ctx.saved_tensors
        #Find the probabilities of the predicted teacher classes, as well as the class predicted
        t_preds, t_pred_classes = torch.max(t_smax_preds, dim = 1)
        #For the student, find the probabilities located at the index of the true labels (i.e. what probability the student had for the correct answer)
        #Use the true classes to accomplish this - unsqueeze the true classes to (batch size, 1) such that .gather can be applied
        true_preds.unsqueeze_(1)
        s_true_label_preds = torch.gather(s_smax_preds, dim = 0, index = true_preds)
        #Find the difference between the STUDENT predicted labels and the GROUND TRUTH predicted labels
        #Subtract by 1, as the true label will always be one (before weighting via teacher)
        diff = torch.sub(s_true_label_preds, 1.0)
        #Convert to vector (remove extra dimension) to allow for element-wise multiplication
        diff.squeeze_()
        #Find the SUM of all the teacher probabilities within the minibatch
        t_label_sum = torch.sum(t_preds, dim = 0)
        #Divide each element in t_pred_labels by the total teacher sum
        weight_tensor = torch.divide(t_preds, t_label_sum)
        #Multiply the weight tensor by the gradients to get the final gradient update, normalize by batch size (first element in the tensor)
        batch_size = s_true_label_preds.shape[0]
        grad_input = torch.mul((1 / batch_size), torch.mul(weight_tensor, diff))
        #As we took the MAX predictions (we subtracted the probabilities of the true predictions), we now have a vector as a gradient.
        #Simply expand the vector to the original input size - each prediction per row should have the same gradient (no dark knowledge)
        #So, each class should have the same gradient - unsqueeze the gradient input to convert to matrix
        grad_input.unsqueeze_(dim = 1)
        grad_input = grad_input.repeat((1, s_smax_preds.shape[1]))
        #Return gradient to update student parameters - neither the teacher nor the true preds must have their gradients updated (return None)
        return grad_input, None, None

#Sample Tensors Taken from Student Training to validate distillation loss function
def test(loss_function, n_args):
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
    training_dataloader = mnist_datablock.dataloaders(mnist_dataset_url/"training", batch_size = 64)
    training_dataloader_2 = mnist_datablock.dataloaders(mnist_dataset_url/"training", batch_size = 64)
    #Create test model (simple 4 layer network)
    test_model_architecture = nn.Sequential(nn.Flatten(),
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
    test_learner.opt = Adam(test_learner.parameters(), lr = 0.0001)
    #Import teacher model (from CustomMaxout.py where a ReLU model was trained on MNIST as a part of a comparison test)
    #The teacher model has the same architecture as the student
    teacher_learner = Learner(training_dataloader_2, test_model_architecture, metrics = ['accuracy', 'error_rate'])
    teacher_learner.load('TEST_TEACHER')
    #Training loop for test learner
    for epoch in range(5):
        batch_losses = []
        for batch_idx, batch_data in enumerate(test_learner.dls.train, 0):
            #Get inputs and labels
            inputs, labels = batch_data
            print(inputs)
            inputs_2 = torch.tensor(inputs)
            #Set grad to zero
            test_learner.zero_grad()
            #Generate predictions
            student_preds = test_learner.model(inputs)
            print(inputs)
            teacher_preds = teacher_learner.model(inputs_2)
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
    t1 = torch.rand(64, 20, requires_grad = True)
    t2 = torch.rand(64, 20)
    t3 = torch.randint(low = 0, high = 19, size = (64,))
    print("TENSOR 1: \n", t1)
    print("TENSOR 2: \n", t2)
    #Create loss object
    loss_func = loss_function.apply
    #Calculate loss - as this method is also used to test the DKPP loss, check the number of required args
    if n_args == 2: loss = loss_func(t1, t2) 
    else: loss = loss_func(t1, t1, t3)
    #Calculate gradients from loss
    loss.retain_grad()
    loss.backward()
    print('LOSS: ', loss)
    print('GRADIENTS: ', loss.grad)

#Test of the same sized tensors to determine if the loss becomes zero
def sampleTest(loss_function, n_args):
    t1 = torch.tensor([[2., 3., 4., 1., 4., 1.]], requires_grad = True)
    t2 = torch.tensor([[2., 3., 4., 1., 4., 1.]], requires_grad = True)
    t3 = torch.tensor([[2, 3, 4, 1, 4, 1]])
    loss_func = loss_function.apply
    if n_args == 2: loss = loss_func(t1, t2)
    else: loss = loss_func(t1, t2, t3)
    print('LOSS: ', loss)
    loss.retain_grad()
    loss.backward()
    print('GRADIENTS: ', loss.grad)

#If the script is run directly from the terminal, perform the test
if __name__ == "__main__":
    test(CWTM_DistillationLoss, n_args = 3)