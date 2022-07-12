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
        ctx.save_for_backward(softmax(s_preds), softmax(t_preds), softmax(true_preds))
        #Use Cross Entropy Loss - the neural networks used in this project are not softmax activated (raw logits), and the softmax conversion done above does not affect the actual tensors
        loss = torch.nn.BCEWithLogitsLoss()
        return loss(s_preds, t_preds)

    #Define backward method (where the gradient of the loss is computed)
    @staticmethod
    def backward(ctx, grad_output):
        #Implement Equation 10 from the BAN paper https://proceedings.mlr.press/v80/furlanello18a/furlanello18a.pdf
        #Obtain labels from the saved tensors
        s_smax_preds, t_smax_preds, true_preds = ctx.saved_tensors
        #Find the predicted labels from both the student and the teacher for that sample (discard the indicies tensor)
        s_pred_labels, idx = torch.max(s_smax_preds, dim = 1)
        t_pred_labels, idx = torch.max(t_smax_preds, dim = 1)
        #Find the difference between the STUDENT predicted labels and the GROUND TRUTH predicted labels
        print(true_preds.size)
        print(true_preds)
        print(s_pred_labels.size)
        diff = torch.sub(s_pred_labels, true_preds)
        #Find the SUM of all the teacher labels - perform this column wise
        t_label_sum = torch.cumsum(t_pred_labels, dim = 0)
        #Divide each element in s_pred_labels by the total teacher sum
        weight_tensor = torch.divide(t_pred_labels, t_label_sum)
        #Multiply the weight tensor by the gradients to get the final gradient update, normalize by batch size (first element in the tensor)
        batch_size = s_pred_labels.shape[0]
        grad_input = batch_size * torch.mul(weight_tensor, diff)
        #Return gradient to update student parameters - neither the teacher nor the true preds must have their gradients updated (return None)
        return grad_input, None, None

#Sample Tensors Taken from Student Training to validate distillation loss function
def test(loss_function, n_args):
    #Initialize sample student, teacher, and true_y tensors
    t1 = torch.tensor([[ 0.2712,  0.4951,  2.3495, -1.3418, -4.1593, -2.5714],
                        [-2.0363, -3.5144,  0.8292, 0.7601, -2.6227, -0.4574],
                        [-0.7060,  0.9497, -1.9310, -1.2268,  1.0167,  0.1409],
                        [-0.6878,  2.1271,  2.0423, -0.8579,  2.5472,  2.2425],
                        [-0.6267, -0.0535,  2.2955, 1.4334, -0.2030,  0.0673],
                        [-1.0541, -0.4715, -2.4049, -0.2177,  2.7034, -0.0544]], requires_grad = True)
    t2 = torch.tensor([[ 1.2977,  3.1175,  4.8884, -5.7194, -0.2108, -2.4929],
                        [-0.5247, -2.2806, -2.0431, 2.7392, -0.0386,  6.6598],
                        [ 3.5381, -0.0925,  2.9286, -0.5404,  0.4483, -0.5553],
                        [-0.7808,  2.6653,  0.5740, -0.8034,  1.0488, -3.3808],
                        [-0.3185,  1.4326, -1.1891, 9.1551, -0.6319, -7.0673],
                        [-0.4056,  0.8678,  0.2392, -2.7831, -2.6583, -1.8217]], requires_grad = False)
    t3 = torch.tensor([[ 1., 0., 0., 0., 0., 0.],
                        [ 0., 1., 0., 0., 0., 0.],
                        [ 0., 0., 1., 0., 0., 0.],
                        [ 1., 0., 0., 0., 0., 0.],
                        [ 0., 0., 0., 0., 1., 0.],
                        [ 0., 1., 0., 0., 0., 0.]])
    #Create loss object
    loss_func = loss_function.apply
    #Calculate loss - as this method is also used to test the DKPP loss, check the number of required args
    if n_args == 2: loss = loss_func(t1, t2) 
    else: loss = loss_func(t1, t2, t3)
    #Calculate gradients from loss
    print('LOSS: ', loss)

#If the script is run directly from the terminal, perform the test
if __name__ == "__main__":
    test(CWTM_DistillationLoss, n_args = 3)