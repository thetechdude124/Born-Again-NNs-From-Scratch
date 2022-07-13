import torch
from torch.autograd import Function

#CWTM (Confidence Weighted by Teacher Max) Distillation Loss - subclass of torch.autograd.Function, all methods are static
class CWTM_DistillationLoss(Function):

    #Define forward method (where we compute the loss) - take in the student predictions, teacher predictions, and true predictions for each model
    @staticmethod
    def forward(ctx, s_preds, t_preds, true_preds):
        #Save both prediction tensors into context object for gradient computations
        #We want to save the NORMALIZED (softmax activated) versions of each tensor as opposed to the raw probabilities - normalize the values first, and then save
        softmax = torch.nn.Softmax(dim = 1) #Perform across each row (each row sums to 1)
        #True predictions are already encoded; they do not require softmax
        ctx.save_for_backward(softmax(s_preds), softmax(t_preds), true_preds)
        #Use Cross Entropy Loss - the neural networks used in this project are not softmax activated (raw logits), and the softmax conversion done above does not affect the actual tensors
        loss = torch.nn.BCEWithLogitsLoss()
        return loss(s_preds, t_preds)

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
        grad_input = torch.mul((1 / batch_size),torch.mul(weight_tensor, diff))
        #As we took the MAX predictions (we subtracted the probabilities of the true predictions), we now have a vector as a gradient.
        #Simply expand the vector to the original input size - each prediction per row should have the same gradient (no dark knowledge)
        #So, each class should have the same gradient - unsqueeze the gradient input to convert to matrix
        grad_input.unsqueeze_(dim = 1)
        grad_input = grad_input.repeat((1, s_smax_preds.shape[1]))
        #Return gradient to update student parameters - neither the teacher nor the true preds must have their gradients updated (return None)
        return grad_input, None, None

#Sample Tensors Taken from Student Training to validate distillation loss function
def test(loss_function, n_args):
    #Initialize sample student, teacher, and true_y tensors with the same sizes as what will be used in the BAN (64 * 20)
    t1 = torch.rand(64, 20, requires_grad = True)
    t2 = torch.rand(64, 20)
    t3 = torch.randint(low = 0, high = 19, size = (64,))
    #Create loss object
    loss_func = loss_function.apply
    #Calculate loss - as this method is also used to test the DKPP loss, check the number of required args
    if n_args == 2: loss = loss_func(t1, t2) 
    else: loss = loss_func(t1, t2, t3)
    #Calculate gradients from loss
    loss.backward()
    print('LOSS: ', loss)

#If the script is run directly from the terminal, perform the test
if __name__ == "__main__":
    test(CWTM_DistillationLoss, n_args = 3)